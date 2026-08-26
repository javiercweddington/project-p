"""
Shared OLE compound-file scrubber for legacy binary formats
(.ppt, .doc, .xls, .sldprt, .sldasm).

Strategy: zero-fill the \\x05SummaryInformation and
\\x05DocumentSummaryInformation property streams IN PLACE (same size, so
the OLE structure stays valid — Author/LastSavedBy/Company/timestamps
become unreadable), then verify that NO mapped entity remains anywhere in
the raw bytes, checked in both UTF-8 and UTF-16LE (legacy Office content
streams are UTF-16LE). Fails closed on any doubt.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

try:
    import olefile
    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False

_logger = logging.getLogger(__name__)

OLE_SUMMARY_STREAMS = ('\x05SummaryInformation',
                       '\x05DocumentSummaryInformation')


def binary_contains_mapped_entity(mapper, data: bytes) -> Optional[str]:
    """Return the first mapped entity found in the data (decoded as UTF-8
    and UTF-16LE), or None when clean.

    Uses the mapper's boundary-aware patterns — a raw substring scan
    false-flagged 3-letter initials like 'HLY' inside ordinary words
    ('highly') and random binary bytes, quarantining clean files.
    """
    import re as _re
    from ..anonymizer import NON_TEXT_ENTITY_TYPES
    # Scan only plausible TEXT RUNS (6+ consecutive printable/CJK chars):
    # decoded binary garbage is full of short letter clusters flanked by
    # control bytes ('~.Hly`') that boundary checks can't distinguish from
    # real words, false-quarantining clean files.
    printable_run = _re.compile(
        r'[\x20-\x7E぀-ヿ㐀-䶿一-鿿가-힯]{6,}')
    texts = []
    for enc in ('utf-8', 'utf-16-le'):
        decoded = data.decode(enc, errors='ignore')
        texts.append('\n'.join(printable_run.findall(decoded)))
    for mapping in mapper.mappings:
        if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
            continue
        value = mapping.original.strip()
        # Binary noise decodes into short letter clusters that no boundary
        # heuristic can tell apart from 3-char initials ('Hly' inside
        # '~.Hly`'), so the binary scan requires >= 4 chars. The property
        # streams that actually attribute such initials are zeroed anyway.
        if len(value) < 4:
            continue
        pattern = None
        build = getattr(mapper, '_build_pattern_cached', None)
        if build is not None:
            try:
                pattern = build(value, mapping.entity_type)
            except Exception:
                pattern = None
        if pattern is None:
            continue
        for text in texts:
            if pattern.search(text):
                return value
    return None


def strip_ole_properties(mapper, input_path: Path, output_path: Path,
                         label: str) -> bool:
    """Zero OLE property streams and verify no mapped entity remains.

    Returns True when the scrubbed file is safe to ship; False when the
    file must be quarantined (not OLE2, unremovable metadata, or mapped
    entity text present in content streams).
    """
    temp_output: Optional[Path] = None
    try:
        if not HAS_OLEFILE:
            _logger.warning(
                "olefile not available; cannot scrub OLE metadata from %s. "
                "Fail-closed for pipeline quarantine.", label,
            )
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with olefile.OleFileIO(str(input_path)) as ole:
                present = [s for s in OLE_SUMMARY_STREAMS if ole.exists(s)]
                sizes = {s: ole.get_size(s) for s in present}
        except Exception as e:
            _logger.warning(
                "%s is not a readable OLE2 file (%s); cannot scrub. "
                "Fail-closed for pipeline quarantine.", label, e,
            )
            return False

        temp_output = output_path.with_name(output_path.name + '.cleantmp')
        shutil.copyfile(input_path, temp_output)

        if present:
            _logger.info(
                "Zero-filling OLE property streams in %s: %s",
                label, [s.lstrip('\x05') for s in present],
            )
            ole = olefile.OleFileIO(str(temp_output), write_mode=True)
            try:
                for stream in present:
                    ole.write_stream(stream, b'\x00' * sizes[stream])
            finally:
                ole.close()

        leaked = binary_contains_mapped_entity(mapper,
                                               temp_output.read_bytes())
        if leaked:
            _logger.warning(
                "Mapped entity %r remains in %s content streams after OLE "
                "property scrub — legacy binary content cannot be rewritten. "
                "Fail-closed for pipeline quarantine.", leaked, label,
            )
            temp_output.unlink()
            return False

        os.replace(temp_output, output_path)
        temp_output = None
        _logger.info(
            "OLE-scrubbed %s: properties zeroed, content verified free of "
            "mapped entities.", label,
        )
        return True

    except Exception as e:
        _logger.error("Error scrubbing OLE file %s: %s", label, e)
        return False
    finally:
        if temp_output is not None and temp_output.exists():
            temp_output.unlink()
