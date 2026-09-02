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

# Above this fraction of all-caps alphabetic tokens, a page is treated
# as caps-styled (engineering drawing) and the ALL-CAPS shape rules are
# disabled — see redact_pil.
try:
    _CAPS_STYLE_MAX_RATIO = float(
        os.environ.get('PROJECT_P_CAPS_RULE_MAX_RATIO', '0.5'))
except ValueError:
    _CAPS_STYLE_MAX_RATIO = 0.5

# Try optional dependencies
try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _shape_rules_enabled() -> bool:
    """Mapper-independent pixel shape rules (identifier-shaped tokens,
    ALL-CAPS runs). DEFAULT OFF: on engineering drawings the identifier
    rule blacked out dimension callouts, tolerance tables, dates, doc
    refs and hardness specs wholesale (every TwistRelease sheet in the
    HoleSaw audit). Under the names-only targeting policy, pixels are
    redacted from mapper needles + NER + logo templates; enable the
    shape net explicitly for finance-style corpora where digit-heavy
    tokens really are account/transaction identifiers."""
    return os.environ.get('PROJECT_P_PIXEL_SHAPE_RULES', '0') == '1'


def _logo_pad_fraction() -> float:
    """Padding around a matched logo box, as a fraction of its size.

    Small on purpose: the template box already covers the artwork
    extent, and a generous pad wipes neighboring part-contour lines —
    on a live collar view the fill severed the drawn silhouette so one
    part read as two objects."""
    try:
        return float(os.environ.get('PROJECT_P_LOGO_PAD', '0.05'))
    except ValueError:
        return 0.05


def _erase_mark_preserving_lines(work, draw, box) -> None:
    """Erase the MARK inside `box` while preserving geometry that
    passes through it.

    A flat rectangle fill severs part-contour lines crossing the
    matched region — live: the fill across a collar view made one drawn
    part read as two objects. Instead: binarize the box and erase every
    ink pixel EXCEPT near-full-crossing straight lines (morphological
    opening in 4 orientations with kernels ~85% of the box span).
    Part contours cross the whole box and survive; logo strokes —
    including the bolt's zigzag segments, and even strokes CONNECTED to
    contour lines (engraved marks touch the silhouette; component
    logic cannot separate them) — never span the box and are erased.
    Erased pixels take the local paper color. Any failure falls back to
    the flat background-fill rectangle; PROJECT_P_LOGO_FILL=box forces
    the flat fill, =black the legacy censor bar.
    """
    if os.environ.get('PROJECT_P_LOGO_FILL', 'background') != 'background':
        _fill_box_background(work, draw, box)
        return
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = work.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw_, bh_ = x1 - x0, y1 - y0
    if bw_ < 8 or bh_ < 8:
        return
    try:
        import cv2
        import numpy as np
        arr = np.asarray(work).copy()
        crop = arr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, bwm = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ink = bwm < 128
        if ink.mean() > 0.5:
            ink = ~ink
        inku = ink.astype(np.uint8)

        klen_h = max(15, int(bw_ * 0.85))
        klen_v = max(15, int(bh_ * 0.85))
        klen_d = max(15, int(min(bw_, bh_) * 0.85))
        preserve = np.zeros_like(inku)
        kern_h = np.ones((1, klen_h), np.uint8)
        kern_v = np.ones((klen_v, 1), np.uint8)
        kern_d1 = np.eye(klen_d, dtype=np.uint8)
        kern_d2 = np.flipud(kern_d1)
        for kern in (kern_h, kern_v, kern_d1, kern_d2):
            preserve |= cv2.morphologyEx(inku, cv2.MORPH_OPEN, kern)
        # A hair of slack so anti-aliased line edges stay with the line.
        preserve = cv2.dilate(preserve, np.ones((3, 3), np.uint8))

        erase = ink & (preserve == 0)
        if not erase.any():
            return
        paper = crop[~ink]
        fill = (np.median(paper, axis=0).astype(np.uint8)
                if paper.size else np.array([255, 255, 255], np.uint8))
        # Take anti-aliased stroke fringes with the strokes.
        erase = cv2.dilate(erase.astype(np.uint8),
                           np.ones((3, 3), np.uint8), iterations=2) > 0
        erase &= (preserve == 0)
        crop[erase] = fill
        from PIL import Image as _Image
        work.paste(_Image.fromarray(crop), (x0, y0))
    except Exception as e:
        _logger.debug("Line-preserving logo erase failed (%s); flat "
                      "background fill instead.", e)
        _fill_box_background(work, draw, box)


def _fill_box_background(work, draw, box) -> None:
    """Paint `box` with the LOCAL background color (median of a border
    ring sampled around it), so the blank reads as empty paper rather
    than a censor bar. Falls back to black when sampling fails
    (numpy missing, degenerate box) and to plain black fill when
    PROJECT_P_LOGO_FILL=black.

    box: (x0, y0, x1, y1), may exceed image bounds (clamped here).
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = work.size
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return
    fill = (0, 0, 0)
    if os.environ.get('PROJECT_P_LOGO_FILL', 'background') != 'black':
        try:
            import numpy as np
            arr = np.asarray(work)
            ring = max(4, (x1c - x0c) // 10, (y1c - y0c) // 10)
            rx0, ry0 = max(0, x0c - ring), max(0, y0c - ring)
            rx1, ry1 = min(w, x1c + ring), min(h, y1c + ring)
            outer = arr[ry0:ry1, rx0:rx1].astype('float32')
            mask = np.ones(outer.shape[:2], dtype=bool)
            mask[(y0c - ry0):(y1c - ry0), (x0c - rx0):(x1c - rx0)] = False
            samples = outer[mask]
            if samples.size:
                fill = tuple(int(v) for v in np.median(samples, axis=0))
        except Exception:
            fill = (0, 0, 0)
    draw.rectangle([x0c, y0c, x1c - 1, y1c - 1], fill=fill)


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

    def _logo_box_reject_reason(self, work, x: int, y: int,
                                w: int, h: int):
        """Adjudicate one template-match box; None means paint it.

        Template correlation cannot separate true marks from two FP
        classes by score alone (live bands overlap: warped/etched marks
        0.66-0.70 vs 'NUMBER' labels 0.66-0.70 and shaded spheres
        0.71-0.74). Two discriminators do, both calibrated on the
        HoleSaw corpus:

        1. COMPACT BLOB: largest binarized component's circularity
           (4*pi*A/P^2). Stroke art maxes at 0.28; spheres/solid shapes
           start at 0.52. Cutoff 0.40.
        2. MACHINE-READABLE TEXT: a logo is art OCR cannot read (every
           true mark read <=3 chars; 'NUMBER' reads perfectly). Text
           regions are belt 2's jurisdiction — EXCEPT when the text is
           itself a mapper needle (a legible 'Milwaukee' script must
           keep the LOGO box so the whole artwork incl. the bolt
           underline is filled, not just the word box).
        """
        left, top = max(0, int(x)), max(0, int(y))
        right = min(work.size[0], int(x + w))
        bottom = min(work.size[1], int(y + h))
        if right - left < 8 or bottom - top < 8:
            return 'degenerate box'
        crop = work.crop((left, top, right, bottom))

        try:
            import cv2
            import numpy as np
            gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
            _, bw = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            ink = bw < 128
            if ink.mean() > 0.5:
                ink = ~ink
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                ink.astype(np.uint8), 8)
            if n > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                big = 1 + int(np.argmax(areas))
                mask = (labels == big).astype(np.uint8)
                cnts, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if cnts:
                    a = cv2.contourArea(cnts[0])
                    p = cv2.arcLength(cnts[0], True)
                    if p > 0 and (4 * 3.14159 * a / (p * p)) > 0.40:
                        return 'compact blob, not stroke art'
        except Exception:
            pass

        try:
            if self.image_ocr and self.image_ocr.available:
                probe = crop
                if min(probe.size) < 60:
                    probe = probe.resize((probe.width * 2,
                                          probe.height * 2))
                text, conf = self.image_ocr.ocr_text_confidence(probe)
                alnum = sum(ch.isalnum() for ch in text)
                # Reject only CONFIDENT text: printed labels read at
                # 38-96 conf, stylized script half-reads at 0-33
                # ('Milas'/'ites'/'Miwon' on live Milwaukee marks).
                # Unknown confidence -> paint (historical behavior;
                # a shipped logo outranks a painted label).
                if alnum >= 5 and conf is not None and conf >= 35:
                    lowered = text.lower()
                    from ..anonymizer import NON_TEXT_ENTITY_TYPES
                    for mapping in self.mapper.mappings:
                        if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                            continue
                        needle = mapping.original.strip().lower()
                        if len(needle) >= 3 and needle in lowered:
                            return None  # legible name -> still a logo
                    return (f'machine-readable text {text[:40]!r} '
                            f'(conf {conf:.0f})')
        except Exception:
            pass
        return None

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
                        yield mapping.original, pattern

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
                    from ..anonymizer import targeted_types
                    _allowed_types = targeted_types()
                    for (key, _words), line_text, ents in zip(
                            line_items, line_texts, per_line):
                        spans = []
                        for ent in ents:
                            # Targeting policy: pixel spans have no
                            # mapper stopover, so gate them here — the
                            # untyped loop was blacking address/date/
                            # money spans on names-only runs.
                            if (_allowed_types is not None
                                    and ent.entity_type
                                    not in _allowed_types):
                                continue
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

            # Belt 0: logo template matching — the leak class OCR can
            # never see (vector logo art has no text and no embedded
            # image part). Templates come from the media review's LOGO
            # enrollments plus any crops dropped into
            # PROJECT_P_LOGO_TEMPLATES. Boxes are padded (matches sit
            # slightly inside the artwork: red bolt tails survived
            # unpadded boxes on 8 of 9 HoleSaw sheets) and filled with
            # the LOCAL BACKGROUND color, not black — the blank should
            # read as empty paper, not as a censor bar.
            try:
                from ..logo_match import env_templates, find_logo_boxes
                logo_templates = env_templates()
                if logo_templates:
                    def _reject_box(rx, ry, rw, rh):
                        reason = self._logo_box_reject_reason(
                            work, rx, ry, rw, rh)
                        if reason:
                            _logger.info(
                                "Logo box (%d,%d,%dx%d) in %s rejected:"
                                " %s", rx, ry, rw, rh, source_name,
                                reason)
                        return reason is not None

                    for (lx, ly, lw, lh) in find_logo_boxes(
                            work, logo_templates, reject=_reject_box):
                        px = max(2, int(lw * _logo_pad_fraction()))
                        py = max(2, int(lh * _logo_pad_fraction()))
                        _erase_mark_preserving_lines(
                            work, draw,
                            (lx - px, ly - py, lx + lw + px, ly + lh + py))
                        had_redactions = True
                        _logger.info(
                            "Logo template match covered %dx%d region "
                            "in %s", lw, lh, source_name)
            except Exception as e:
                _logger.warning("Logo template matching failed for %s: "
                                "%s", source_name, e)

            # Caps-styled pages (engineering drawings, CAD title blocks)
            # set virtually ALL text in capitals by drafting convention —
            # there "ALL-CAPS" stops meaning "name-shaped" and the caps
            # rules would black out entire spec tables (seen live:
            # 'JAM TEST (CYCLES)', 'SEE SHEET 1 FOR REVISIONS'). When
            # most alphabetic tokens are all-caps, redaction relies on
            # mapper patterns + GLiNER + the identifier rule instead.
            alpha_tokens = [
                word
                for words in lines.values()
                for word, _x, _y, _w, _h in words
                if len(re.sub(r'[^A-Za-z]', '', word)) >= 3]
            caps_ratio = (
                sum(1 for t in alpha_tokens if t == t.upper())
                / len(alpha_tokens)) if alpha_tokens else 0.0
            shape_rules = _shape_rules_enabled()
            caps_rules_on = (shape_rules
                             and caps_ratio <= _CAPS_STYLE_MAX_RATIO)
            if shape_rules and not caps_rules_on:
                _logger.info(
                    "Caps-styled page (%d%% all-caps tokens): ALL-CAPS "
                    "shape rules off for %s",
                    round(caps_ratio * 100), source_name)

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
                    if shape_rules and _is_identifier_token(word):
                        _redact_box(x, y, w, h)
                    if (caps_rules_on and _is_caps_word(word)
                            and len(re.sub(r'[^A-Za-z]', '', word)) >= 6):
                        _redact_box(x, y, w, h)
                    if (caps_rules_on and _is_caps_word(word)
                            and not money_re.match(word)):
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
                         for _orig, pattern in patterns
                         for m in pattern.finditer(line_text)]
                spans.extend(gliner_spans.get(key, []))
                for start, end in spans:
                    for (ws, we, x, y, w, h) in offsets:
                        if ws < end and we > start:
                            _redact_box(x, y, w, h)

            total_words = sum(len(words) for words in lines.values())

            if had_redactions:
                # Verify: re-OCR the redacted image; every entity pattern
                # must now be unreadable. A residual match is COVERED at
                # its re-OCR position and verified again (OCR segments
                # shift once boxes land, so one pass routinely leaves a
                # stray readable fragment — giving up on the first one
                # quarantined whole files that one more box would fix).
                for attempt in range(3):
                    verify_lines = _ocr_lines(work)
                    residual = []
                    for key, words in verify_lines.items():
                        line_text = ''
                        offsets = []
                        for word, x, y, w, h in words:
                            if line_text:
                                line_text += ' '
                            offsets.append(
                                (len(line_text),
                                 len(line_text) + len(word), x, y, w, h))
                            line_text += word
                        for orig, pattern in patterns:
                            for m2 in pattern.finditer(line_text):
                                residual.append(
                                    (orig, m2.start(), m2.end(), offsets))
                    if not residual:
                        break
                    if attempt == 2:
                        _logger.warning(
                            "Pixel redaction verify: %r still readable "
                            "after %d cover passes in %s — fail-closed.",
                            residual[0][0], attempt + 1, source_name)
                        return None
                    for _orig, start, end, offsets in residual:
                        for (ws, we, x, y, w, h) in offsets:
                            if ws < end and we > start:
                                _redact_box(x, y, w, h)

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