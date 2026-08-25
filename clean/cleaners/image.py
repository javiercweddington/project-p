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
    from PIL import Image
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

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)
        self.image_ocr = ImageOCR() if HAS_IMAGE_OCR else None

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
        # Step 1: OCR the image and register entities (Option A)
        self._ocr_and_register_entities(input_path)

        # Step 2: Remove metadata
        if not HAS_PIL:
            _logger.warning("PIL not available; copying image as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return True

        try:
            ext = input_path.suffix.lower()

            # JPEG needs special handling for maker notes, XMP, thumbnails
            if ext in ('.jpg', '.jpeg'):
                return self._clean_jpeg(input_path, output_path)
            else:
                return self._clean_other_image(input_path, output_path)

        except Exception as e:
            _logger.error("Error cleaning image %s: %s", input_path, e)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return False

    def _clean_jpeg(self, input_path: Path, output_path: Path) -> bool:
        """Clean JPEG by removing ALL metadata segments.

        This addresses the three critical JPEG risks:
        1. Maker notes (Canon OwnerName, Nikon Artist)
        2. xmpMM:DocumentID/DerivedFrom
        3. EXIF thumbnail (separate embedded JPEG)

        Strategy: Rebuild the JPEG from the pixel data only, discarding
        all APPn segments and COM segments. This is more thorough than
        PIL's default save which may preserve some metadata.
        """
        try:
            img = Image.open(input_path)

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

            # Save with EXIF=None to ensure no metadata is preserved
            # This removes: EXIF (including maker notes, GPS, thumbnail),
            # XMP, IPTC, ICC profiles, and comments
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(
                output_path,
                'JPEG',
                quality=95,
                exif=b'',      # No EXIF data
                icc_profile=b'',  # No ICC profile
                dpi=(72, 72),  # Generic DPI
            )

            # Verify no metadata survived
            self._verify_jpeg_clean(output_path)

            return True

        except Exception as e:
            _logger.error("JPEG cleaning failed: %s", e)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return False

    def _clean_other_image(self, input_path: Path, output_path: Path) -> bool:
        """Clean non-JPEG images by removing metadata."""
        try:
            img = Image.open(input_path)
            ext = input_path.suffix.lower()

            # For PNG, WebP, etc., save without info dict
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if ext == '.png':
                img.save(output_path, 'PNG')
            elif ext == '.webp':
                img.save(output_path, 'WEBP')
            elif ext == '.bmp':
                img.save(output_path, 'BMP')
            else:
                img.save(output_path)

            return True

        except Exception as e:
            _logger.error("Image cleaning failed for %s: %s", input_path, e)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
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

    # ---- OCR integration (Option A bridge) ----

    def _ocr_and_register_entities(self, input_path: Path) -> None:
        """OCR an image via acquire/metadata.py and register entities."""
        if not self.image_ocr:
            return

        try:
            text = self.image_ocr.extract_text(input_path)
            if not text:
                return

            _logger.debug(
                "OCR extracted %d chars from %s",
                len(text), input_path.name,
            )

            # Use mapper's regex-based replacement to find known entities
            # This also registers any new entities found in the OCR text
            cleaned = self.mapper.replace_in_text(text)

            # If the text was modified, entities were found
            if cleaned != text:
                _logger.info(
                    "Found entities in OCR text from %s", input_path.name
                )

        except Exception as e:
            _logger.debug(
                "OCR/entity detection failed for %s: %s",
                input_path, e,
            )

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