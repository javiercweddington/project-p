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

import logging
import os
import zipfile
from pathlib import Path

_logger = logging.getLogger(__name__)

_XML_SUFFIXES = ('.xml', '.rels', '.vml')


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
                    new_text = mapper.replace_in_text(
                        text, source=f'{label}::{info.filename}')
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
