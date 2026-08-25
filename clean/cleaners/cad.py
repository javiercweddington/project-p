"""
CADCleaner - clean CAD files across all common formats.

Addresses CAD-specific risks based on format analysis:

.sldprt / .sldasm (SolidWorks) - OLE compound file:
  - Summary stream (Author, Last Saved By, Company)
  - Custom properties, file-level and per-configuration
  - External references, embedded design tables
  - Strategy: COPY-AS-IS with warning. Extract metadata via CADMetadataExtractor.

.step / .stp (STEP/ISO 10303) - ASCII:
  - HEADER: FILE_NAME, FILE_DESCRIPTION, PRODUCT names
  - Strategy: Extract metadata, register entities, then text clean.

.stl (Stereolithography) - ASCII or binary:
  - Binary header may contain 80 bytes of text
  - Strategy: Text clean for ASCII, strip header for binary.

.iges / .igs (IGES) - ASCII: Header entities
.obj (Wavefront OBJ) - ASCII: Comment lines
.dxf (AutoCAD DXF) - ASCII: HEADER section variables
.err (Creo error logs) - Text: Full paths, usernames

.3mf (3D Manufacturing Format) - ZIP-based: recurse and clean

Option A: imports CADMetadataExtractor from acquire/metadata.py
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from ..anonymizer import EntityMapper
from .text import TextCleaner

# Option A: Import extractor from acquire module
# Using absolute import since acquire is a sibling package at project root
try:
    from acquire.metadata import CADMetadataExtractor
    HAS_CAD_EXTRACTOR = True
except ImportError:
    HAS_CAD_EXTRACTOR = False

_logger = logging.getLogger(__name__)

try:
    import pyole2
    HAS_PYOLE2 = True
except ImportError:
    HAS_PYOLE2 = False


CAD_ASCII_TEXT = {
    '.step', '.stp', '.iges', '.igs', '.obj',
    '.dxf', '.stl', '.err',
}

CAD_BINARY_COPY = {
    '.sldprt', '.sldasm', '.prt', '.asm',
    '.x_t', '.x_b', '.ipt', '.iam',
    '.dwg', '.fbx', '.blend',
}

CAD_ZIP_BASED = {'.3mf'}


_METADATA_ENTITY_TYPE_MAP = {
    'author': 'person',
    'last_saved_by': 'person',
    'company': 'company',
    'designer': 'person',
    'organization': 'company',
    'manager': 'person',
    'product': 'product',
    'description': 'sensitive_doc',
    'title': 'sensitive_doc',
    'subject': 'sensitive_doc',
    'comments': 'sensitive_doc',
    'creator_tool': 'product',
    'create_date': 'date',
    'modify_date': 'date',
    'time': 'date',
    'directory': 'address',
    'revision': 'sensitive_doc',
    'name': 'product',
    'value': 'product',
}


class CADCleaner:
    """Clean CAD files using CADMetadataExtractor from acquire/metadata.py."""

    STEP_FILE_NAME_RE = re.compile(
        r"FILE_NAME\s*\([^;]*\);",
        re.MULTILINE,
    )

    STEP_FILE_DESCRIPTION_RE = re.compile(
        r"FILE_DESCRIPTION\s*\([^;]*\);",
        re.MULTILINE,
    )

    STEP_PRODUCT_RE = re.compile(
        r"PRODUCT\s*\(\s*'([^']*?)'\s*,\s*'([^']*?)'\s*,\s*'([^']*?)'\s*,\s*'([^']*?)'\s*\)",
        re.MULTILINE,
    )

    STL_SOLID_RE = re.compile(
        r'^solid\s+\S+',
        re.MULTILINE,
    )

    OBJ_MTLIB_RE = re.compile(
        r'^mtllib\s+\S+',
        re.MULTILINE,
    )

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)
        self.cad_extractor = CADMetadataExtractor() if HAS_CAD_EXTRACTOR else None

        if not HAS_CAD_EXTRACTOR:
            _logger.warning(
                "CADMetadataExtractor not available; CAD metadata extraction disabled"
            )

    # ---- public API ----

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean a CAD file: extract metadata, register entities, clean."""
        ext = input_path.suffix.lower()

        # Step 1: Extract metadata and register entities (Option A)
        self._extract_and_register_metadata(input_path)

        # Step 2: Format-specific cleaning
        if ext in CAD_ASCII_TEXT:
            return self._clean_ascii_cad(input_path, output_path, ext)
        elif ext in CAD_BINARY_COPY:
            return self._copy_binary_cad_with_warning(input_path, output_path, ext)
        elif ext in CAD_ZIP_BASED:
            return self._handle_zip_based_cad(input_path, output_path, ext)
        else:
            return self._copy_binary_cad_with_warning(input_path, output_path, ext)

    # ---- metadata extraction (Option A bridge) ----

    def _extract_and_register_metadata(self, input_path: Path) -> None:
        """Extract CAD metadata via acquire/metadata.py and register in mapper."""
        if not self.cad_extractor:
            return

        try:
            metadata = self.cad_extractor.extract_metadata(input_path)
            if not metadata:
                return

            for key, value in metadata.items():
                if not value or len(value.strip()) < 2:
                    continue

                entity_type = _METADATA_ENTITY_TYPE_MAP.get(
                    key.lower(), 'sensitive_doc'
                )

                placeholder = self.mapper.get_or_create(
                    entity_type=entity_type,
                    value=value.strip(),
                    source=str(input_path),
                )

                _logger.debug(
                    "Registered CAD entity: %r -> %s (from %s, type=%s)",
                    value.strip(), placeholder, key, entity_type,
                )

        except Exception as e:
            _logger.debug(
                "Failed to extract/register CAD metadata from %s: %s",
                input_path, e,
            )

    # ---- format-specific cleaners ----

    def _clean_ascii_cad(self, input_path: Path, output_path: Path, ext: str) -> bool:
        """Clean ASCII-based CAD files using text cleaning."""
        # For STL, check if binary
        if ext == '.stl':
            return self._clean_stl(input_path, output_path)

        # For other ASCII CAD files, use text cleaner
        return self.text_cleaner.clean_file(input_path, output_path)

    def _clean_stl(self, input_path: Path, output_path: Path) -> bool:
        """Clean STL files (ASCII or binary)."""
        try:
            with open(input_path, 'rb') as f:
                header = f.read(80)

            # Binary STL: first 80 bytes are header, followed by 4-byte int (triangle count)
            # ASCII STL: starts with "solid "
            if header[:6] == b'solid ':
                # ASCII STL: text clean
                return self.text_cleaner.clean_file(input_path, output_path)
            else:
                # Binary STL: strip header, copy rest
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(input_path, 'rb') as fin:
                    # Skip original 80-byte header
                    fin.seek(80)
                    rest = fin.read()

                with open(output_path, 'wb') as fout:
                    # Write clean header
                    fout.write(b'Cleaned STL - Project P' + b'\x00' * 36)
                    fout.write(rest)

                return True

        except Exception as e:
            _logger.error("Error cleaning STL %s: %s", input_path, e)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return False

    def _copy_binary_cad_with_warning(self, input_path: Path,
                                       output_path: Path, ext: str) -> bool:
        """Copy binary CAD file with warning about potential metadata."""
        _logger.warning(
            "Binary CAD file (%s) copied as-is. "
            "May contain embedded metadata (author, paths, properties). "
            "Consider converting to STEP format for thorough cleaning.",
            ext,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return True

    def _handle_zip_based_cad(self, input_path: Path,
                               output_path: Path, ext: str) -> bool:
        """Handle ZIP-based CAD formats like .3mf."""
        _logger.info("ZIP-based CAD format %s handled by ZipCleaner", ext)
        # Defer to ZipCleaner for ZIP-based formats
        from .zip_cleaner import ZipCleaner
        zip_cleaner = ZipCleaner(self.mapper)
        return zip_cleaner.clean_file(input_path, output_path)

    # ---- risk detection ----

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in a CAD file."""
        risks = []
        ext = input_path.suffix.lower()

        if ext in CAD_BINARY_COPY:
            risks.append(
                f"Binary CAD format ({ext}) may contain embedded metadata "
                f"(author, paths, properties) that cannot be cleaned"
            )

        # Use extractor to check for sensitive metadata
        if self.cad_extractor:
            try:
                has_sensitive, sensitive_meta = (
                    self.cad_extractor.has_sensitive_metadata(input_path)
                )
                if has_sensitive:
                    risks.append(
                        f"Sensitive metadata found: "
                        f"{', '.join(sensitive_meta.keys())}"
                    )
            except Exception:
                pass

        return risks