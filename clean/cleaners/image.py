"""
ImageCleaner - remove EXIF/metadata from images and perform OCR.

Addresses JPEG-specific risks (ranked by failure likelihood):

1. Maker notes (CRITICAL) - Canon OwnerName, Nikon Artist, and other
   manufacturer-specific tags that survive naive EXIF strippers. These
   are embedded in the MakerNote tag (EXIF tag 271) and require
   explicit removal.

2. xmpMM:DocumentID/DerivedFrom (HIGH) - Links your cleaned file back
   to its ancestor. Even after "stripping" EXIF, the XMP packet may
   contain DocumentID and DerivedFrom URIs that fingerprint the file
   to its original.

3. EXIF thumbnail (HIGH) - A separate embedded JPEG (Tag 28322 / 0x7E0A)
   that shows the unredacted original. If you redact only the main image,
   the thumbnail still contains the sensitive content.

4. GPS coordinates (HIGH) - A whiteboard photo geolocates your facility.
   GPS data is in EXIF tag 34853 (0x8825).

5. Camera and lens serials (MEDIUM) - Unique identifiers that fingerprint
   the equipment used.

6. Content: connection diagrams with hostnames/IPs, datasheets with
   part numbers/vendors. OCR is mandatory here, not optional.

Strategy: Use PIL/Pillow for basic metadata stripping, then use direct
binary manipulation to remove maker notes, XMP packets, and EXIF thumbnails
that PIL cannot access through its API.

Option A: imports ImageOCR from acquire/metadata.py for OCR,
registers discovered entities in the EntityMapper.

Dependencies (optional):
- PIL/Pillow: Image manipulation and metadata stripping
- acquire/metadata.py ImageOCR: OCR for text detection in images
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import struct
from pathlib import Path
from typing import List, Optional, Tuple

from ..anonymizer import EntityMapper
from .text import TextCleaner

# Option A: Import ImageOCR from acquire module
# Using absolute import since acquire is a sibling package at project root
try:
    from acquire.metadata import ImageOCR
    HAS_IMAGE_OCR = True
except ImportError:
    HAS_IMAGE_OCR = False

_logger = logging.getLogger(__name__)

# Try optional dependencies
try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageCleaner:
    """Clean image files by removing EXIF and other metadata.

    Uses PIL/Pillow for basic metadata stripping and direct binary
    manipulation for maker notes, XMP packets, and EXIF thumbnails
    that PIL cannot access through its API.

    Uses ImageOCR from acquire/metadata.py to detect sensitive text
    content in images and register discovered entities in the EntityMapper.
    """

    SENSITIVE_EXIF_TAGS = {
        'Artist', 'Copyright', 'Copyrighted', 'ImageUniqueID',
        'GPSInfo', 'GPSAltitude', 'GPSLatitude', 'GPSLongitude',
        'DateTimeOriginal', 'DateTimeDigitized',
        'Make', 'Model', 'Software',
        'XPTitle', 'XPComment', 'XPAuthor', 'XPKeywords',
    }

    # JPEG marker constants
    JPEG_APP1_MARKER = 0xFFE1  # EXIF/XMP uses APP1
    JPEG_APP13_MARKER = 0xFFED  # Photoshop IRB
    JPEG_APP2_MARKER = 0xFFE2  # ICC profiles
    JPEG_COM_MARKER = 0xFFFE  # Comments

    def __init__(self, mapper: EntityMapper, quarantine_dir: Optional[Path] = None):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)
        self.image_ocr = ImageOCR() if HAS_IMAGE_OCR else None
        self._quarantine_dir = quarantine_dir

        if not HAS_IMAGE_OCR:
            _logger.warning(
                "ImageOCR not available from acquire/metadata.py; "
                "image OCR disabled"
            )

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Remove metadata from an image file and OCR for entity detection.

        Args:
            input_path: Source image path
            output_path: Destination image path

        Returns:
            True if cleaning was successful
        """
        if not HAS_PIL:
            _logger.warning("PIL not available; image fail-closed")
            return False

        ocr_functional = bool(self.image_ocr) and getattr(
            self.image_ocr, 'available', False)
        require_ocr = os.environ.get(
            'PROJECT_P_REQUIRE_IMAGE_OCR', '1') != '0'

        if not ocr_functional and require_ocr:
            # Unverifiable pixels must not ship. (Set
            # PROJECT_P_REQUIRE_IMAGE_OCR=0 to allow metadata-only
            # cleaning when OCR is unavailable.)
            _logger.warning(
                "OCR unavailable; image %s fail-closed for quarantine "
                "(pixel text cannot be screened).", input_path.name,
            )
            return False

        redacted_tmp: Optional[Path] = None
        try:
            ext = input_path.suffix.lower()
            work_input = input_path

            # Step 1: OCR-driven PIXEL REDACTION. Entity text found in the
            # image is blacked out at word level, then the redacted image
            # is re-OCRed; if any entity is still readable, fail closed.
            if ocr_functional:
                redaction = self._ocr_redact_pixels(input_path)
                if redaction is None:
                    _logger.warning(
                        "Could not verify pixel redaction for %s - "
                        "fail-closed for quarantine.", input_path.name,
                    )
                    return False
                redacted_img, had_redactions, ocr_word_count = redaction

                # DOCUMENT-SCREENSHOT GUARD: OCR is too noisy to certify a
                # text-dense image clean via entity matching alone (a
                # re-encode can garble exactly the words that leak — 'NOA
                # LABS' reads fine to a human but matches no pattern).
                # Policy (PROJECT_P_IMAGE_TEXT_POLICY):
                #   allow (default)  — ship with ENTITY-ONLY blackout
                #                      (people/banks/companies blocked,
                #                      amounts and generic text readable;
                #                      accepts OCR-noise residual risk)
                #   redact           — black out EVERY text region
                #   quarantine       — never ship document-style images
                word_limit = int(os.environ.get(
                    'PROJECT_P_IMAGE_TEXT_WORD_LIMIT', '15'))
                policy = os.environ.get(
                    'PROJECT_P_IMAGE_TEXT_POLICY', 'allow').lower()
                if ocr_word_count >= word_limit and policy != 'allow':
                    if policy == 'quarantine':
                        _logger.warning(
                            "%s is a document-style image (%d OCR words): "
                            "policy=quarantine — fail-closed.",
                            input_path.name, ocr_word_count,
                        )
                        return False
                    blacked = self._redact_all_text(input_path)
                    if blacked is None:
                        _logger.warning(
                            "Could not black out all text in document-style "
                            "image %s — fail-closed for quarantine.",
                            input_path.name,
                        )
                        return False
                    redacted_img = blacked
                    had_redactions = True
                    _logger.info(
                        "Blacked out ALL text regions in document-style "
                        "image %s (%d OCR words).",
                        input_path.name, ocr_word_count,
                    )

                if had_redactions:
                    redacted_tmp = input_path.with_name(
                        input_path.name + '.redact.tmp.png')
                    redacted_img.save(redacted_tmp, 'PNG')
                    work_input = redacted_tmp
                    _logger.info(
                        "Redacted entity text pixels in %s", input_path.name,
                    )

            # Step 2: Remove metadata (from the redacted copy when present)
            if ext in ('.jpg', '.jpeg'):
                return self._clean_jpeg(work_input, output_path)
            elif ext == '.tiff' or ext == '.tif':
                return self._clean_tiff(work_input, output_path)
            else:
                return self._clean_other_image(work_input, output_path,
                                               original_ext=ext)

        except Exception as e:
            _logger.error("Error cleaning image %s: %s", input_path, e)
            return False
        finally:
            if redacted_tmp is not None and redacted_tmp.exists():
                redacted_tmp.unlink()

    def _redact_all_text(self, input_path: Path):
        """Black out EVERY OCR-detected text region in an image file.

        Returns the blacked-out PIL image, or None on failure/unverified.
        """
        try:
            img = Image.open(input_path)
            img.load()
            work = img.convert('RGB')
        except Exception as e:
            _logger.warning(
                "Full-text blackout failed for %s: %s", input_path, e,
            )
            return None
        return self.redact_all_text_pil(work, source_name=input_path.name)

    def redact_all_text_pil(self, work, source_name: str = '<image>'):
        """Black out EVERY OCR-detected text region in a PIL RGB image.

        For document-style images this is the only redaction that is
        verifiable by construction: after blacking all word boxes, a
        re-OCR must find (almost) no text — a handful of stray noise
        words is tolerated, real residual text fails.

        Returns the blacked-out PIL image, or None on failure/unverified.
        """
        try:
            from PIL import ImageDraw
        except ImportError:
            return None
        if not (self.image_ocr and self.image_ocr.available):
            return None

        try:
            def _word_boxes(image):
                lines = self.image_ocr.ocr_lines(image)
                if lines is None:
                    raise RuntimeError('no OCR backend for word boxes')
                return [(x, y, w, h)
                        for words in lines.values()
                        for (_word, x, y, w, h) in words]

            draw = ImageDraw.Draw(work)
            for x, y, w, h in _word_boxes(work):
                pad = max(2, h // 6)
                draw.rectangle(
                    [x - pad, y - pad, x + w + pad, y + h + pad],
                    fill=(0, 0, 0))

            # Verify by construction: almost nothing may OCR afterwards
            residual_words = len(_word_boxes(work))
            if residual_words > 5:
                _logger.warning(
                    "%d words still OCR-readable after full-text blackout "
                    "of %s", residual_words, source_name,
                )
                return None
            return work

        except Exception as e:
            _logger.warning(
                "Full-text blackout failed for %s: %s", source_name, e,
            )
            return None

    def _ocr_redact_pixels(self, input_path: Path):
        """OCR an image file, black out entity text, verify.

        Returns (redacted_PIL_image, had_redactions, ocr_word_count) on
        success, or None when redaction could not be verified (caller
        must fail closed).
        """
        try:
            img = Image.open(input_path)
            img.load()
            work = img.convert('RGB')
        except Exception as e:
            _logger.warning(
                "OCR pixel redaction failed for %s: %s", input_path, e,
            )
            return None
        return self.redact_pil(work, source_name=str(input_path))

    def redact_pil(self, work, source_name: str = '<image>'):
        """OCR a PIL RGB image, black out entity text at word level, verify.

        Shared by image files and rasterized PDF pages. Returns
        (redacted_PIL_image, had_redactions, ocr_word_count) on success,
        or None when redaction could not be verified (caller must fail
        closed).
        """
        from ..anonymizer import NON_TEXT_ENTITY_TYPES
        try:
            from PIL import ImageDraw
        except ImportError:
            return None
        if not (self.image_ocr and self.image_ocr.available):
            return None

        try:
            def _ocr_lines(image):
                lines = self.image_ocr.ocr_lines(image)
                if lines is None:
                    raise RuntimeError('no OCR backend for word boxes')
                return lines

            def _entity_patterns():
                for mapping in self.mapper.mappings:
                    if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                        continue
                    pattern = self.mapper._build_pattern_cached(
                        mapping.original, mapping.entity_type)
                    if pattern is not None:
                        yield pattern

            lines = _ocr_lines(work)

            # Register deterministic identifiers from the OCR text so
            # accounts/codes/emails seen only in pixels also become
            # mapped entities (and are then redacted below).
            full_text = '\n'.join(
                ' '.join(w[0] for w in words) for words in lines.values())
            self.text_cleaner._register_emails(
                full_text, source=source_name)

            # GENERAL image-redaction rules (mapper-independent). OCR noise
            # garbles exact values ('JCW20200803F' reads as '3cuze2e0s03*'),
            # but token SHAPE survives, so shape rules are the reliable net:
            # 1. identifier-shaped tokens: >=4 digits in a >=6-char token
            #    (doc codes, account numbers, transaction refs, dates) —
            #    money amounts like 6,682.50 are explicitly excluded;
            # 2. ALL-CAPS runs of >=2 words (company/bank names:
            #    'NOA LABS LIMITED', 'WELLS FARGO BANK N A SAN FRANCTSCO').
            money_re = re.compile(r'^[\$€£¥]?[A-Za-z]{0,3}\**[\d,]*\.\d{2}\*?$')

            def _is_identifier_token(token: str) -> bool:
                if money_re.match(token):
                    return False
                alnum = re.sub(r'[^A-Za-z0-9]', '', token)
                digits = sum(ch.isdigit() for ch in alnum)
                letters = len(alnum) - digits
                # Digit-heavy identifiers (codes, accounts, refs, dates)
                if digits >= 4 and len(alnum) >= 6:
                    return True
                # Mixed letter-digit tokens: OCR often garbles digits into
                # letters ('RBT005150' -> 'ABTeES1S0'), so the digit count
                # drops — but words never mix digits at all, making any
                # long mixed token identifier-shaped.
                if digits >= 2 and letters >= 2 and len(alnum) >= 8:
                    return True
                return False

            def _is_caps_word(token: str) -> bool:
                return (token == token.upper()
                        and re.search(r'[A-Z]', token) is not None)

            # NER stage (the CV -> words+positions -> GLiNER -> cover-by-
            # position architecture): when GLiNER is importable (server),
            # every OCR line is NER-scanned and detected person/company
            # values become redaction spans. Locally (no GLiNER) the mapper
            # patterns + shape rules below carry the load.
            gliner_spans = {}
            try:
                from acquire import catalog as _catalog
                if getattr(_catalog, 'HAS_GLINER', False):
                    from acquire.catalog import (
                        _extract_entities_with_gliner_batch)
                    # ONE batched NER pass over all lines of the image
                    # (a model call per line left the GPU idle between
                    # tiny inputs).
                    line_items = list(lines.items())
                    line_texts = [' '.join(w[0] for w in words)
                                  for _key, words in line_items]
                    per_line = _extract_entities_with_gliner_batch(
                        line_texts, source_name)
                    for (key, _words), line_text, ents in zip(
                            line_items, line_texts, per_line):
                        spans = []
                        for ent in ents:
                            for m2 in re.finditer(
                                    re.escape(ent.value), line_text, re.I):
                                spans.append((m2.start(), m2.end()))
                        if spans:
                            gliner_spans[key] = spans
            except Exception as e:
                _logger.debug("GLiNER image NER unavailable: %s", e)

            draw = ImageDraw.Draw(work)
            had_redactions = False
            patterns = list(_entity_patterns())

            def _redact_box(x, y, w, h):
                nonlocal had_redactions
                pad = max(2, h // 8)
                draw.rectangle(
                    [x - pad, y - pad, x + w + pad, y + h + pad],
                    fill=(0, 0, 0))
                had_redactions = True

            # Rule pass: identifier tokens + all-caps runs. A lone ALL-CAPS
            # word of >=6 letters is also redacted — line wraps strand the
            # tail of a name run ('...N A SAN' / newline / 'FRANCTSCO').
            for words in lines.values():
                caps_run = []
                for word, x, y, w, h in words:
                    if _is_identifier_token(word):
                        _redact_box(x, y, w, h)
                    if (_is_caps_word(word)
                            and len(re.sub(r'[^A-Za-z]', '', word)) >= 6):
                        _redact_box(x, y, w, h)
                    if _is_caps_word(word) and not money_re.match(word):
                        caps_run.append((word, x, y, w, h))
                    else:
                        if sum(1 for t in caps_run if len(t[0]) >= 3) >= 2:
                            for _, cx, cy, cw, ch in caps_run:
                                _redact_box(cx, cy, cw, ch)
                        caps_run = []
                if sum(1 for t in caps_run if len(t[0]) >= 3) >= 2:
                    for _, cx, cy, cw, ch in caps_run:
                        _redact_box(cx, cy, cw, ch)

            for key, words in lines.items():
                # Build the line text with char offsets per word
                line_text = ''
                offsets = []
                for word, x, y, w, h in words:
                    if line_text:
                        line_text += ' '
                    offsets.append((len(line_text), len(line_text) + len(word),
                                    x, y, w, h))
                    line_text += word
                # Cover mapped-entity matches AND GLiNER-detected spans
                spans = [(m.start(), m.end())
                         for pattern in patterns
                         for m in pattern.finditer(line_text)]
                spans.extend(gliner_spans.get(key, []))
                for start, end in spans:
                    for (ws, we, x, y, w, h) in offsets:
                        if ws < end and we > start:
                            _redact_box(x, y, w, h)

            total_words = sum(len(words) for words in lines.values())

            if had_redactions:
                # Verify: re-OCR the redacted image; every entity pattern
                # must now be unreadable.
                verify_lines = _ocr_lines(work)
                verify_text = '\n'.join(
                    ' '.join(w[0] for w in words)
                    for words in verify_lines.values())
                for pattern in patterns:
                    if pattern.search(verify_text):
                        return None

            return work, had_redactions, total_words

        except Exception as e:
            _logger.warning(
                "OCR pixel redaction failed for %s: %s", source_name, e,
            )
            return None

    def _quarantine(self, input_path: Path, quarantine_dir: Path) -> None:
        """Copy an image to a quarantine directory for manual review."""
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = quarantine_dir / input_path.name
        # Handle name collisions
        counter = 1
        while dest.exists():
            stem = input_path.stem
            suffix = input_path.suffix
            dest = quarantine_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(input_path, dest)
        _logger.info("Quarantined %s -> %s", input_path, dest)

    def _ocr_and_check_entities(self, input_path: Path) -> bool:
        """OCR an image and check if any known entities appear in the text.

        Args:
            input_path: Source image path

        Returns:
            True if entities were found in the OCR text.
        """
        if not self.image_ocr:
            return False

        try:
            text = self.image_ocr.extract_text(input_path)
            if not text:
                return False

            _logger.debug(
                "OCR extracted %d chars from %s",
                len(text), input_path.name,
            )

            # Register any email addresses seen in the OCR text so they are
            # treated as entities (pixels can't be redacted, so a hit below
            # quarantines the image).
            self.text_cleaner._register_emails(
                text, source=str(input_path))

            # Non-mutating check: search with the boundary-aware patterns
            # WITHOUT running a replacement (replace_in_text would inflate
            # occurrence counts and fire the tracker for a discarded string).
            from ..anonymizer import NON_TEXT_ENTITY_TYPES
            for mapping in self.mapper.mappings:
                if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                    continue
                pattern = self.mapper._build_pattern_cached(
                    mapping.original, mapping.entity_type)
                if pattern is not None and pattern.search(text):
                    _logger.info(
                        "Entity %r found in OCR text from %s",
                        mapping.original, input_path.name,
                    )
                    return True

            return False

        except Exception as e:
            _logger.debug(
                "OCR/entity detection failed for %s: %s",
                input_path, e,
            )
            return False

    def _clean_jpeg(self, input_path: Path, output_path: Path) -> bool:
        """Clean JPEG by removing ALL metadata segments.

        This addresses the three critical JPEG risks:
        1. Maker notes (Canon OwnerName, Nikon Artist)
        2. xmpMM:DocumentID/DerivedFrom
        3. EXIF thumbnail (separate embedded JPEG)

        Strategy: Apply exif_transpose to handle orientation, then rebuild
        the JPEG from the pixel data only, discarding all APPn segments
        and COM segments. Do NOT re-encode at a fixed quality (the old
        quality=95 was a blind re-encode that degraded already-clean images).
        """
        try:
            img = Image.open(input_path)

            # Determine whether pixels must be transposed. If the Orientation
            # tag is 1/absent we can keep the original JPEG data quality via
            # quality='keep' (no re-encode inflation, no generation loss);
            # if a real rotation is needed we must re-encode.
            try:
                orientation = img.getexif().get(0x0112, 1)
            except Exception:
                orientation = 1
            needs_transpose = orientation not in (None, 0, 1)
            can_keep_quality = (img.format == 'JPEG') and not needs_transpose

            if needs_transpose:
                # CRITICAL: honor the Orientation tag BEFORE stripping EXIF,
                # otherwise photos ship rotated/mirrored.
                img = ImageOps.exif_transpose(img)

            # Convert to RGB if necessary (strip alpha for JPEG)
            if img.mode in ('RGBA', 'P', 'LA'):
                # Handle transparency by compositing on white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Drop metadata PIL would otherwise carry over from img.info
            # (COM comment segments, XMP, leftover EXIF bytes).
            for info_key in ('comment', 'exif', 'icc_profile', 'xmp'):
                img.info.pop(info_key, None)

            # Save with NO metadata preserved.
            # exif=b'' removes EXIF (including maker notes, GPS, thumbnail)
            # icc_profile=b'' removes ICC profiles
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = dict(
                exif=b'',         # No EXIF data
                icc_profile=b'',  # No ICC profile
                dpi=(72, 72),     # Generic DPI
                # Preserve progressive encoding — forcing baseline on a
                # progressive source inflates the file noticeably.
                progressive=bool(img.info.get('progressive')
                                 or img.info.get('progression')),
            )
            if can_keep_quality:
                # Reuse the source quantization tables: no size inflation,
                # no extra generation loss.
                save_kwargs['quality'] = 'keep'
            else:
                save_kwargs['quality'] = 95
            try:
                img.save(output_path, 'JPEG', **save_kwargs)
            except ValueError:
                # quality='keep' can fail for non-JPEG-backed images
                save_kwargs['quality'] = 95
                img.save(output_path, 'JPEG', **save_kwargs)

            # Verify no metadata survived
            self._verify_jpeg_clean(output_path)

            return True

        except Exception as e:
            _logger.error("JPEG cleaning failed: %s", e)
            return False

    def _clean_tiff(self, input_path: Path, output_path: Path) -> bool:
        """Clean TIFF by stripping ALL metadata including XMP and IPTC.

        TIFF files commonly carry XMP and IPTC metadata that survives
        naive EXIF stripping. We rebuild the TIFF from pixel data only.
        """
        try:
            img = Image.open(input_path)

            # Apply exif_transpose for orientation
            img = ImageOps.exif_transpose(img)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Drop metadata PIL would carry over via img.info
            # (icc_profile / xmp / exif survive a naive TIFF re-save).
            for info_key in ('comment', 'exif', 'icc_profile', 'xmp',
                             'photoshop'):
                img.info.pop(info_key, None)

            # Save TIFF from pixel data only (strips XMP, IPTC, EXIF, etc.)
            # save_all preserves multi-page TIFFs (plain save keeps page 1 only)
            img.save(
                output_path,
                'TIFF',
                compression='tiff_lzw',  # Lossless compression
                save_all=getattr(img, 'n_frames', 1) > 1,
            )

            return True

        except Exception as e:
            _logger.error("TIFF cleaning failed: %s", e)
            return False

    def _clean_other_image(self, input_path: Path, output_path: Path,
                           original_ext: Optional[str] = None) -> bool:
        """Clean non-JPEG/TIFF images by removing metadata.

        original_ext preserves the intended output format when input_path
        is a temporary redacted PNG standing in for the real file.
        """
        try:
            img = Image.open(input_path)

            # Apply exif_transpose for orientation (e.g., WebP, PNG)
            img = ImageOps.exif_transpose(img)

            ext = (original_ext or input_path.suffix).lower()

            # Drop metadata PIL would carry over via img.info (PNG iCCP/eXIf
            # chunks, WebP EXIF/ICC, GIF comments all round-trip otherwise).
            for info_key in ('comment', 'exif', 'icc_profile', 'xmp'):
                img.info.pop(info_key, None)

            # For PNG, WebP, etc., save without info dict
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Animated GIF/WebP must keep every frame — a plain save()
            # silently flattens to frame 1 (data loss under SUCCESS).
            multi_frame = getattr(img, 'n_frames', 1) > 1

            if ext == '.png':
                img.save(output_path, 'PNG')
            elif ext == '.webp':
                img.save(output_path, 'WEBP', save_all=multi_frame)
            elif ext == '.bmp':
                img.save(output_path, 'BMP')
            elif ext == '.gif':
                img.save(output_path, 'GIF', save_all=multi_frame)
            else:
                img.save(output_path)

            return True

        except Exception as e:
            _logger.error("Image cleaning failed for %s: %s", input_path, e)
            return False

    def _verify_jpeg_clean(self, output_path: Path) -> None:
        """Verify that a JPEG file has no metadata segments.

        Checks for:
        - APP1 segments (EXIF/XMP)
        - APP13 segments (Photoshop IRB)
        - COM segments (comments)
        """
        try:
            with open(output_path, 'rb') as f:
                data = f.read()

            # Look for JPEG markers
            pos = 2  # Skip SOI marker
            while pos < len(data) - 2:
                if data[pos] != 0xFF:
                    pos += 1
                    continue

                marker = data[pos + 1]
                if marker == 0xD9:  # EOI
                    break
                elif marker in (0xE0, 0xE1, 0xE2, 0xED, 0xEE, 0xEF):
                    # APPn marker found - check size
                    length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    if marker == 0xE1:  # APP1 (EXIF/XMP)
                        segment = data[pos:pos+length]
                        if b'XMP' in segment or b'xmpMM' in segment:
                            _logger.warning(
                                "XMP data survived cleaning in %s",
                                output_path.name,
                            )
                        if b'MakerNote' in segment:
                            _logger.warning(
                                "MakerNote survived cleaning in %s",
                                output_path.name,
                            )
                    pos += length + 2
                elif marker == 0xFE:  # COM (comment)
                    length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    pos += length + 2
                else:
                    pos += 1

        except Exception as e:
            _logger.debug("JPEG verification failed for %s: %s", output_path, e)

    def detect_entities_in_image(self, input_path: Path) -> List[str]:
        """OCR an image and detect sensitive entities in the text.

        Args:
            input_path: Source image path

        Returns:
            List of detected entity values
        """
        if not self.image_ocr:
            return []

        try:
            text = self.image_ocr.extract_text(input_path)
            if not text:
                return []

            detected = []
            for mapping in self.mapper.mappings:
                if mapping.original.lower() in text.lower():
                    detected.append(mapping.original)

            return detected
        except Exception as e:
            _logger.debug("Entity detection failed for %s: %s", input_path, e)
            return []

    def get_exif_summary(self, input_path: Path) -> dict:
        """Extract EXIF metadata summary for audit purposes.

        Args:
            input_path: Source image path

        Returns:
            Dictionary of EXIF tags and values
        """
        if not HAS_PIL:
            return {}

        try:
            img = Image.open(input_path)
            exif_data = img._getexif() or {}

            summary = {}
            for tag_id, value in exif_data.items():
                # Try to get tag name
                try:
                    from PIL.ExifTags import TAGS
                    tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
                except ImportError:
                    tag_name = f"Tag_{tag_id}"

                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."

                summary[tag_name] = value_str

            return summary
        except Exception as e:
            _logger.debug("Failed to extract EXIF from %s: %s", input_path, e)
            return {}

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in an image file.

        Args:
            input_path: Source image path

        Returns:
            List of detected risk descriptions
        """
        risks = []

        if not HAS_PIL:
            return risks or ["Cannot inspect image: PIL not available"]

        try:
            img = Image.open(input_path)
            exif_data = img._getexif() or {}

            # Check for GPS data
            if 'GPSInfo' in exif_data:
                risks.append("GPS coordinates present - geolocates facility")

            # Check for maker notes
            if 271 in exif_data:  # MakerNote tag
                risks.append("Maker notes present - may contain OwnerName/Artist")

            # Check for XMP
            if 700 in exif_data:  # Related to XMP
                risks.append("XMP metadata present - may contain DocumentID/DerivedFrom")

            # Check for embedded thumbnail
            if 28322 in exif_data or 0x7E0A in exif_data:
                risks.append("EXIF thumbnail present - separate embedded JPEG")

            # Check for camera serial
            for tag in exif_data:
                tag_name = str(tag)
                if 'serial' in tag_name.lower() or 'lens' in tag_name.lower():
                    risks.append(f"Camera/lens identifier: {tag_name}")
                    break

        except Exception as e:
            risks.append(f"Risk detection failed: {e}")

        return risks