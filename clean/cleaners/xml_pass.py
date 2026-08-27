"""
Raw XML member catch-all pass for Office ZIP outputs.

Structured cleaners rewrite what their libraries expose — openpyxl cell
values, python-docx paragraphs, python-pptx shapes — but Office files
carry text in members those APIs never visit: drawing/text-box XML
(xl/drawings/*, w:txbxContent), chart XML, SmartArt, comments, legacy
VML. Entities there survived every re-clean (seen live: 'MAKROLON' and
a fax number persisting across docx/xlsx/xlsm despite correct patterns).

This pass reopens the CLEANED zip and runs mapper replacement over the
raw text of every XML-ish member. It is XML-safe because:
- placeholders are plain ASCII with no markup characters;
- entity patterns cannot cross tag boundaries (their flexible
  separators match whitespace only, never '<' or '>');
- the anonymizer's variants include XML-escaped spellings (&amp;,
  &#NNN;), so escaped text nodes are matched as themselves.

It is a CATCH-ALL, not a replacement for the structured cleaners: text
split across adjacent runs (<w:t>MAKRO</w:t><w:t>LON</w:t>) still
requires the structured pass; this closes the whole-string-in-one-node
gap (text boxes, charts, drawings).
"""

from __future__ import annotations

import bisect
import logging
import os
import re
import zipfile
from pathlib import Path

_logger = logging.getLogger(__name__)

_XML_SUFFIXES = ('.xml', '.rels', '.vml')

# OOXML text nodes: <w:t>, <a:t>, <t>, <xdr:t>... Rich-text formatting
# SPLITS one visible string across adjacent runs (<t>MAKRO</t><t>LON</t>),
# which no single-node text replacement can match — while the LLM-scan
# extraction (which strips tags with no separator, by design, for split
# CJK names) rejoins and reports exactly those strings. Seen live:
# MAKROLON / 'Design App UI/UX' surviving every re-clean of an xlsm.
_TEXT_NODE_RE = re.compile(
    r'(<(?:[\w.-]+:)?t(?:\s[^>]*)?>)'      # opening <t ...>
    r'((?:(?!</(?:[\w.-]+:)?t>).)*?)'      # text content (non-greedy)
    r'(</(?:[\w.-]+:)?t>)',                # closing </t>
    re.DOTALL)

# Tags that mark a NEW visible container (cell, row, paragraph, shared
# string, table cell). A gap containing one of these becomes a newline
# barrier in the joined view, so entities cannot phantom-bridge two
# unrelated cells ('917' + '13184849' in adjacent cells must not read
# as one fax number).
_CONTAINER_BOUNDARY_RE = re.compile(
    r'<(?:/)?(?:[\w.-]+:)?(?:si|c|row|p|tr|tc|sp|txBody)[\s>/]')


def _replace_across_text_runs(xml_text: str, mapper, source: str) -> str:
    """Replace entities whose text is SPLIT across adjacent text runs.

    Joins run contents (newline barriers at container boundaries), finds
    boundary-aware pattern matches in the joined view, and splices each
    cross-node match back: the placeholder lands in the first touched
    node, the covered tails of subsequent nodes are emptied. Only
    markup-free node CONTENT is edited, so the XML stays valid.
    Single-node matches are left to the plain replacement pass.
    """
    nodes = list(_TEXT_NODE_RE.finditer(xml_text))
    if len(nodes) < 2:
        return xml_text

    segments = [m.group(2) for m in nodes]
    # Joined view with per-gap barriers
    joined_parts = []
    bounds = []          # start offset of each segment in the joined view
    offset = 0
    for i, segment in enumerate(segments):
        if i > 0:
            gap = xml_text[nodes[i - 1].end():nodes[i].start()]
            if _CONTAINER_BOUNDARY_RE.search(gap):
                joined_parts.append('\n')
                offset += 1
        bounds.append(offset)
        joined_parts.append(segment)
        offset += len(segment)
    joined = ''.join(joined_parts)
    joined_lower = joined.lower()

    from ..anonymizer import NON_TEXT_ENTITY_TYPES

    def node_of(pos: int) -> int:
        return bisect.bisect_right(bounds, pos) - 1

    # Collect cross-node match spans (non-overlapping, first-come)
    spans = []
    taken = []
    for mapping in mapper.mappings:
        if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
            continue
        needles = mapper.prefilter_needles(
            mapping.original, mapping.entity_type)
        if needles and not any(n in joined_lower for n in needles):
            continue
        pattern = mapper._build_pattern_cached(
            mapping.original, mapping.entity_type)
        if pattern is None:
            continue
        for match in pattern.finditer(joined):
            start, end = match.start(), match.end()
            if node_of(start) == node_of(max(start, end - 1)):
                continue  # single-node: plain pass owns it
            if any(s < end and e > start for s, e in taken):
                continue
            taken.append((start, end))
            spans.append((start, end, mapping))
    if not spans:
        return xml_text

    # Per-node edit lists: (local_start, local_end, replacement)
    edits = {}
    for start, end, mapping in spans:
        first = node_of(start)
        last = node_of(end - 1)
        for i in range(first, last + 1):
            seg_start = bounds[i]
            seg_end = seg_start + len(segments[i])
            lo = max(start, seg_start) - seg_start
            hi = min(end, seg_end) - seg_start
            if hi <= lo:
                continue
            replacement = mapping.placeholder if i == first else ''
            edits.setdefault(i, []).append((lo, hi, replacement))
        mapping.occurrence_count += 1
        callback = getattr(mapper, '_tracker_callback', None)
        if callback is not None:
            callback(mapping.original, mapping.placeholder, source)
        _logger.info(
            "Cross-run replacement: %r -> %s in %s",
            mapping.original, mapping.placeholder, source)

    # Apply edits (reverse order within each node), splice back (reverse
    # node order so earlier match offsets stay valid).
    new_segments = list(segments)
    for i, node_edits in edits.items():
        segment = new_segments[i]
        for lo, hi, replacement in sorted(node_edits, reverse=True):
            segment = segment[:lo] + replacement + segment[hi:]
        new_segments[i] = segment
    result = xml_text
    for i in range(len(nodes) - 1, -1, -1):
        if new_segments[i] != segments[i]:
            m = nodes[i]
            result = (result[:m.start(2)] + new_segments[i]
                      + result[m.end(2):])
    return result


def scrub_zip_xml_members(zip_path: Path, mapper,
                          source_label: str = '') -> bool:
    """Replace mapped entities in every XML member of an Office zip.

    Rewrites zip_path in place (via a temp file) only when something
    changed. Returns True on success (including no-op); False on any
    failure — callers must fail closed.
    """
    label = source_label or zip_path.name
    try:
        with zipfile.ZipFile(zip_path, 'r') as zin:
            infos = zin.infolist()
            members = {}
            changed = False
            for info in infos:
                data = zin.read(info.filename)
                if info.filename.lower().endswith(_XML_SUFFIXES):
                    try:
                        text = data.decode('utf-8')
                    except UnicodeDecodeError:
                        members[info.filename] = data
                        continue
                    member_label = f'{label}::{info.filename}'
                    new_text = mapper.replace_in_text(
                        text, source=member_label)
                    new_text = _replace_across_text_runs(
                        new_text, mapper, member_label)
                    if new_text != text:
                        changed = True
                        _logger.info(
                            "XML catch-all pass replaced entities in "
                            "%s::%s", label, info.filename)
                        data = new_text.encode('utf-8')
                members[info.filename] = data

        if not changed:
            return True

        tmp_path = zip_path.with_name(zip_path.name + '.xmlpass.tmp')
        try:
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                for info in infos:
                    zout.writestr(info.filename, members[info.filename])
            os.replace(tmp_path, zip_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return True

    except Exception as e:
        _logger.error(
            "XML catch-all pass failed for %s: %s (failing closed)",
            label, e)
        return False
