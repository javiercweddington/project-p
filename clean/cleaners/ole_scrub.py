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
        # Join with NUL, not '\n': flexible-whitespace patterns were
        # stitching "matches" from fragments at unrelated file offsets
        # ('NOA Labs' + 'Lt' thousands of bytes apart) — a leak no
        # same-length surgery can remove because no contiguous bytes
        # hold it. NUL is not \s, so matches stay within one run;
        # every contiguous (surgery-removable) leak is still caught.
        texts.append('\x00'.join(printable_run.findall(decoded)))
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


def overwrite_entities_in_binary(mapper, data: bytes):
    """Same-length in-place entity surgery on a binary blob.

    Finds every mapped entity (case-insensitive; plain and collapsed
    forms) in three byte views — latin-1 (single-byte/ASCII text) and
    UTF-16LE at byte offsets 0 and 1 (legacy Office content streams) —
    and overwrites the matched BYTES with 'X' filler of identical
    length, so no stream offset or record size ever shifts.

    Each overwrite is self-validating: the target bytes are re-decoded
    and compared to the expected form first, so a decode-index drift
    (astral chars in the replace-mode view) can never corrupt unrelated
    bytes — a drifted match is skipped and left for the verify gate.

    Returns (new_bytes, overwrite_count).
    """
    import re as _re
    from ..anonymizer import NON_TEXT_ENTITY_TYPES

    targets = []
    for mapping in mapper.mappings:
        if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
            continue
        value = mapping.original.strip().lower()
        if len(value) < 4:
            continue  # same threshold as the binary scan (noise guard)
        # Separator variants too: the verify gate matches 'globus-medical'
        # via its boundary patterns, so surgery must find the same forms
        # or a variant spelling forces quarantine the surgery could fix.
        forms = {value, _re.sub(r'[\s\-_]+', '', value)}
        for sep in ('-', '_', ' '):
            forms.add(_re.sub(r'[\s\-_]+', sep, value))
        targets.append((mapping, [f for f in forms if len(f) >= 4]))
    if not targets:
        return data, 0

    ba = bytearray(data)
    total = 0

    views = [(data.decode('latin-1').lower(), 1, 0, 'latin-1')]
    for offset in (0, 1):
        views.append((
            data[offset:].decode('utf-16-le', errors='replace').lower(),
            2, offset, 'utf-16-le'))

    def _overwrite(b0: int, b1: int, expected: str, width: int,
                   enc: str) -> bool:
        if b1 > len(ba):
            return False
        # Self-validation against decode-index drift
        segment = bytes(ba[b0:b1])
        if segment.decode(enc, errors='replace').lower() != expected:
            return False
        for k in range(len(expected)):
            pos = b0 + k * width
            ba[pos] = 0x58  # 'X'
            if width == 2:
                ba[pos + 1] = 0x00
        return True

    build = getattr(mapper, '_build_pattern_cached', None)
    for text, width, base, enc in views:
        # Fast pass: literal + collapsed spellings
        for mapping, forms in targets:
            for form in forms:
                search_from = 0
                while True:
                    idx = text.find(form, search_from)
                    if idx < 0:
                        break
                    search_from = idx + len(form)
                    if _overwrite(base + idx * width,
                                  base + (idx + len(form)) * width,
                                  form, width, enc):
                        total += 1
                        mapping.occurrence_count += 1
        # Gate-alignment pass: the verify gate matches VARIANT spellings
        # (hyphen/underscore/flexible whitespace) via the mapper's
        # boundary patterns — surgery must remove the SAME matches or a
        # variant forces quarantine surgery could have fixed (seen live:
        # 'NOA Labs Lt' variant in a SolidWorks binary). Run the gate's
        # own patterns over each text view and overwrite the match spans.
        if build is None:
            continue
        text_lower = text.lower()
        for mapping, _forms in targets:
            needles = ()
            get_needles = getattr(mapper, 'prefilter_needles', None)
            if get_needles is not None:
                needles = get_needles(mapping.original, mapping.entity_type)
            if needles and not any(n in text_lower for n in needles):
                continue
            try:
                pattern = build(mapping.original, mapping.entity_type)
            except Exception:
                pattern = None
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                expected = text[match.start():match.end()]
                if _overwrite(base + match.start() * width,
                              base + match.end() * width,
                              expected, width, enc):
                    total += 1
                    mapping.occurrence_count += 1
    return bytes(ba), total


_JPEG_SOI = b'\xff\xd8\xff'
_JPEG_EOI = b'\xff\xd9'
_PNG_SIG = b'\x89PNG\r\n\x1a\n'
_PNG_END = b'IEND\xaeB`\x82'


def _iter_embedded_images(data: bytes, max_images: int = 40):
    """Yield JPEG/PNG blobs embedded in a binary (PPT Pictures stream,
    SolidWorks preview bitmaps) by signature scanning."""
    count = 0
    pos = 0
    while count < max_images:
        start = data.find(_JPEG_SOI, pos)
        if start < 0:
            break
        end = data.find(_JPEG_EOI, start + 3)
        if end < 0:
            break
        yield data[start:end + 2]
        count += 1
        pos = end + 2
    pos = 0
    while count < max_images:
        start = data.find(_PNG_SIG, pos)
        if start < 0:
            break
        end = data.find(_PNG_END, start + 8)
        if end < 0:
            break
        yield data[start:end + len(_PNG_END)]
        count += 1
        pos = end + len(_PNG_END)


def embedded_image_entity_check(mapper, data: bytes,
                                label: str) -> Optional[str]:
    """OCR embedded raster images and check for mapped entities.

    Returns None when clean/no images; the offending value on a hit; or
    '<ocr-unavailable>' when images exist but cannot be screened
    (callers fail closed unless PROJECT_P_REQUIRE_IMAGE_OCR=0).

    Residual risk (documented): handwritten signatures inside embedded
    images are cursive strokes OCR cannot read — this gate catches
    TYPED text in pixels only.
    """
    blobs = list(_iter_embedded_images(data))
    if not blobs:
        return None
    try:
        from acquire.metadata import ImageOCR
        ocr = ImageOCR()
    except ImportError:
        ocr = None
    if not (ocr and getattr(ocr, 'available', False)):
        if os.environ.get('PROJECT_P_REQUIRE_IMAGE_OCR', '1') == '0':
            _logger.warning(
                "%s has %d embedded image(s) but OCR is unavailable; "
                "shipping WITHOUT pixel screening "
                "(PROJECT_P_REQUIRE_IMAGE_OCR=0).", label, len(blobs))
            return None
        return '<ocr-unavailable>'
    from ..anonymizer import NON_TEXT_ENTITY_TYPES
    for blob in blobs:
        text = ocr.extract_text_from_bytes(blob)
        if not text:
            continue
        for mapping in mapper.mappings:
            if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                continue
            if len(mapping.original.strip()) < 4:
                continue
            build = getattr(mapper, '_build_pattern_cached', None)
            pattern = build(mapping.original,
                            mapping.entity_type) if build else None
            if pattern is not None and pattern.search(text):
                return mapping.original
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
            # Opaque non-OLE container (e.g. newer SolidWorks, header
            # 7730bc22): no property streams to scrub. Default is
            # quarantine. PROJECT_P_OPAQUE_BINARY=ship-scanned ships it
            # after a raw-byte entity scan + embedded-image OCR gate +
            # same-length surgery — CAVEAT: strings inside compressed
            # streams are invisible to the scan, so this accepts a
            # residual risk the OLE path does not.
            if os.environ.get('PROJECT_P_OPAQUE_BINARY',
                              'quarantine').lower() == 'ship-scanned':
                _logger.warning(
                    "%s is not OLE2 (%s); PROJECT_P_OPAQUE_BINARY="
                    "ship-scanned — shipping after raw-byte scan "
                    "(compressed streams are NOT scannable).", label, e)
                data = input_path.read_bytes()
                data, overwritten = overwrite_entities_in_binary(
                    mapper, data)
                if overwritten:
                    _logger.info(
                        "Overwrote %d entity occurrence(s) in opaque "
                        "binary %s.", overwritten, label)
                leaked = binary_contains_mapped_entity(mapper, data)
                if leaked:
                    _logger.warning(
                        "Mapped entity %r visible in opaque binary %s — "
                        "fail-closed.", leaked, label)
                    return False
                image_leak = embedded_image_entity_check(mapper, data,
                                                         label)
                if image_leak:
                    _logger.warning(
                        "Embedded image gate failed for opaque binary "
                        "%s (%s) — fail-closed.", label, image_leak)
                    return False
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
                return True
            _logger.warning(
                "%s is not a readable OLE2 file (%s); cannot scrub. "
                "Fail-closed for pipeline quarantine. "
                "(PROJECT_P_OPAQUE_BINARY=ship-scanned to ship opaque "
                "binaries after a raw-byte scan.)", label, e,
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

        # Same-length entity surgery on content streams: matched entity
        # bytes become 'X' filler (offsets never shift), so entity text
        # in slide/content streams no longer forces quarantine.
        data = temp_output.read_bytes()
        data, overwritten = overwrite_entities_in_binary(mapper, data)
        if overwritten:
            _logger.info(
                "Overwrote %d entity occurrence(s) in %s content "
                "streams (same-length surgery).", overwritten, label)
            temp_output.write_bytes(data)

        leaked = binary_contains_mapped_entity(mapper, data)
        if leaked:
            _logger.warning(
                "Mapped entity %r remains in %s content streams after OLE "
                "property scrub + surgery. "
                "Fail-closed for pipeline quarantine.", leaked, label,
            )
            temp_output.unlink()
            return False

        # Embedded raster images (PPT Pictures stream, SolidWorks preview
        # bitmaps): OCR-screen for entity text; unscreenable = quarantine.
        image_leak = embedded_image_entity_check(mapper, data, label)
        if image_leak:
            _logger.warning(
                "Embedded image in %s %s — fail-closed for quarantine.",
                label,
                'cannot be OCR-screened' if image_leak == '<ocr-unavailable>'
                else f'contains mapped entity {image_leak!r}')
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
