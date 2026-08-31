"""
Comb & destroy: verification-driven residual elimination.

The deterministic checks (Entity Leakage, Metadata) report each residual
with its ENTITY and EXACT LOCATION ('FILE_105.docx::word/media/
image7.jpeg::utf-8', 'FILE_106.xls::OLE-utf16'). This module consumes
those hits and eliminates them with targeted repairs:

- Embedded media members of Office zips (word/media/*, ppt/media/*):
  the entity lives in the image's OWN metadata (EXIF/XMP), which the
  standalone image cleaner never sees. Repair: re-encode the member
  from pixels — all metadata gone by construction.
- OLE / opaque binaries: targeted same-length surgery with the REPORTED
  values only. No minimum-length noise guard applies: these are
  confirmed hits from the verifier, not speculative scan needles (a
  3-char acronym like 'TTI' slipped the >=4 surgery threshold live).
- Anything else (or a repair that does not verify clean): the file is
  QUARANTINED. Ship-clean-or-quarantine — a comb pass never leaves a
  known residual in the deliverable.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_OFFICE_ZIP_EXTS = {'.docx', '.xlsx', '.pptx', '.xlsm', '.docm', '.pptm',
                    '.odt', '.ods', '.odp'}
_OLE_EXTS = {'.doc', '.xls', '.ppt', '.sldprt', '.sldasm'}
_MEDIA_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp',
               '.emf', '.wmf')
# Formats PIL cannot decode (HD Photo, video, audio): a flagged member
# in one of these gets a placeholder, never a copy-through.
_OPAQUE_MEDIA_EXTS = ('.wdp', '.jxr', '.mov', '.mp4', '.avi', '.wmv',
                      '.m4v', '.mp3', '.wav', '.bin')


def _placeholder_member_bytes() -> bytes:
    """A tiny neutral PNG used to overwrite undecodable media members."""
    import io
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (2, 2), (229, 229, 229)).save(buf, 'PNG')
        return buf.getvalue()
    except ImportError:
        return b''


def _parse_hit_location(file_path: str) -> Tuple[str, Optional[str]]:
    """Split a verifier hit path into (relative file, member-or-None).

    Formats seen: 'rel', 'rel[utf-8]', 'rel::member::utf-8',
    'rel::member', 'rel::OLE', 'rel::OLE-utf16', 'rel::EXIF',
    'rel::STEP_HEADER'.
    """
    rel = file_path.split('::', 1)[0]
    if rel.endswith(']') and '[' in rel:
        rel = rel[:rel.rindex('[')]
    member = None
    parts = file_path.split('::')
    if len(parts) >= 2 and parts[1] not in (
            'OLE', 'OLE-utf16', 'EXIF', 'STEP_HEADER'):
        member = parts[1]
    return rel, member


def _reencode_image_bytes(data: bytes) -> Optional[bytes]:
    """Rebuild an image from pixels only (drops EXIF/XMP/ICC/comments)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        fmt = (img.format or 'PNG').upper()
        for key in ('exif', 'icc_profile', 'xmp', 'comment', 'photoshop'):
            img.info.pop(key, None)
        buf = io.BytesIO()
        if fmt in ('JPEG', 'JPG', 'MPO'):
            img.convert('RGB').save(buf, 'JPEG', quality=90)
        elif fmt == 'PNG':
            img.save(buf, 'PNG')
        else:
            img.save(buf, fmt)
        return buf.getvalue()
    except Exception as e:
        _logger.debug("Image re-encode failed: %s", e)
        return None


def _comb_office_zip(path: Path, mapper, members: List[str]) -> bool:
    """Repair specific members of an Office zip in place.

    Media members are re-encoded from pixels; XML members get one more
    plain + cross-run replacement attempt. Returns True when every
    targeted member was rewritten.
    """
    from .cleaners.xml_pass import _replace_across_text_runs
    targeted = set(members)
    try:
        with zipfile.ZipFile(path, 'r') as zin:
            infos = zin.infolist()
            blobs = {i.filename: zin.read(i.filename) for i in infos}
    except zipfile.BadZipFile:
        return False

    ok = True
    for member in targeted:
        data = blobs.get(member)
        if data is None:
            ok = False
            continue
        lower = member.lower()
        if lower.endswith(_MEDIA_EXTS + _OPAQUE_MEDIA_EXTS):
            new = _reencode_image_bytes(data)
            if new is None:
                # Undecodable media (.wdp/.emf/.MOV — PIL can't rebuild
                # from pixels): REPLACE with a neutral placeholder
                # instead of quarantining the whole deck. The member's
                # content (and whatever it leaked) is destroyed by
                # construction; the slide shows a grey box.
                new = _placeholder_member_bytes()
                _logger.info(
                    "Comb: media member %s::%s not re-encodable — "
                    "replaced with placeholder (content destroyed)",
                    path.name, member)
            else:
                _logger.info("Comb: re-encoded media member %s::%s "
                             "(metadata destroyed)", path.name, member)
            blobs[member] = new
        else:
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                ok = False
                continue
            label = f'{path.name}::{member}'
            new_text = mapper.replace_in_text(text, source=label)
            new_text = _replace_across_text_runs(new_text, mapper, label)
            blobs[member] = new_text.encode('utf-8')

    tmp = path.with_name(path.name + '.combtmp')
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
            for info in infos:
                zout.writestr(info.filename, blobs[info.filename])
        os.replace(tmp, path)
    except Exception as e:
        _logger.warning("Comb: zip rewrite failed for %s: %s", path.name, e)
        if tmp.exists():
            tmp.unlink()
        return False
    return ok


def _comb_binary(path: Path, mapper,
                 values: List[Tuple[str, str]]) -> bool:
    """Targeted same-length surgery for CONFIRMED values in a binary."""
    from .cleaners.ole_scrub import overwrite_entities_in_binary

    class _Target:
        def __init__(self, original, entity_type):
            self.original = original
            self.entity_type = entity_type
            self.occurrence_count = 0

    class _TargetMapper:
        """Adapter exposing only the confirmed hit values (min length 2 —
        they are verifier-confirmed, not speculative), backed by the real
        mapper's pattern builder."""
        def __init__(self, real, targets):
            self.mappings = targets
            self._build_pattern_cached = getattr(
                real, '_build_pattern_cached', None)
            self.prefilter_needles = getattr(
                real, 'prefilter_needles', lambda *_a: ())

    targets = [_Target(v, t) for v, t in dict.fromkeys(values)]
    try:
        data = path.read_bytes()
    except OSError:
        return False
    # Bypass the >=4 length guard: temporarily pad? No — the guard lives
    # inside overwrite_entities_in_binary; run it with a shim mapper and
    # ALSO do a direct short-value pass for confirmed 2-3 char values.
    new_data, count = overwrite_entities_in_binary(
        _TargetMapper(mapper, targets), data)
    ba = bytearray(new_data)
    short_total = 0
    for value, _etype in dict.fromkeys(values):
        v = value.strip()
        if not (2 <= len(v) < 4):
            continue
        for enc, width in (('latin-1', 1), ('utf-16-le', 2)):
            needle = v.lower()
            for offset in ((0,) if width == 1 else (0, 1)):
                text = bytes(ba[offset:]).decode(enc, errors='replace').lower()
                start = 0
                while True:
                    idx = text.find(needle, start)
                    if idx < 0:
                        break
                    start = idx + len(needle)
                    b0 = offset + idx * width
                    b1 = b0 + len(needle) * width
                    if b1 > len(ba):
                        continue
                    if bytes(ba[b0:b1]).decode(
                            enc, errors='replace').lower() != needle:
                        continue
                    for k in range(len(needle)):
                        pos = b0 + k * width
                        ba[pos] = 0x58
                        if width == 2:
                            ba[pos + 1] = 0x00
                    short_total += 1
    if count or short_total:
        try:
            path.write_bytes(bytes(ba))
        except OSError:
            return False
        _logger.info("Comb: overwrote %d occurrence(s) of confirmed "
                     "values in %s", count + short_total, path.name)
    return True


def comb_residuals(mapper, staging_dir: Path, hits,
                   quarantine: Callable[[Path], None]) -> Dict[str, int]:
    """Eliminate reported residuals; quarantine what cannot be repaired.

    hits: LeakageHit list from the deterministic checks. Returns a
    summary dict {'repaired': n, 'quarantined': n, 'skipped': n}.
    """
    from .verifier import LeakageChecker

    by_file: Dict[str, List] = defaultdict(list)
    for hit in hits:
        if hit.entity_type == 'unverifiable':
            continue
        rel, member = _parse_hit_location(hit.file_path)
        by_file[rel].append((member, hit))

    checker = LeakageChecker(mapper)
    summary = {'repaired': 0, 'quarantined': 0, 'skipped': 0}

    for rel, file_hits in by_file.items():
        path = staging_dir / rel
        if not path.is_file():
            summary['skipped'] += 1
            continue
        suffix = path.suffix.lower()
        values = [(h.original, h.entity_type) for _m, h in file_hits]

        attempted = False
        if suffix in _OFFICE_ZIP_EXTS:
            members = sorted({m for m, _h in file_hits if m})
            if members:
                attempted = _comb_office_zip(path, mapper, members)
        elif suffix in _OLE_EXTS or True:
            # Binary/OLE and anything else text-like: targeted surgery
            # is same-length and self-validating, safe on any format.
            attempted = _comb_binary(path, mapper, values)

        # Re-verify THIS file with the same checks that flagged it.
        residual = []
        if attempted:
            residual = checker.check_file(path, path, rel)
            if suffix in _OFFICE_ZIP_EXTS:
                residual += checker._check_office_metadata(path, rel)
            elif suffix in _OLE_EXTS:
                residual += checker._check_ole_metadata(path, rel)
        if attempted and not residual:
            summary['repaired'] += 1
            _logger.info("Comb: %s verified clean after repair", rel)
        else:
            summary['quarantined'] += 1
            _logger.warning(
                "Comb: %s could not be fully repaired (%d residual) — "
                "quarantining.", rel, len(residual) if attempted else -1)
            quarantine(path)
    return summary
