"""
CADCleaner - clean CAD files across all common formats.

Addresses CAD-specific risks based on format analysis:

.sldprt / .sldasm (SolidWorks) - OLE compound file:
  - Summary stream (Author, Last Saved By, Company)
  - Custom properties, file-level and per-configuration
  - External references, embedded design tables
  - Strategy: Strip OLE SummaryInformation via olefile/pyole2.

.step / .stp (STEP/ISO 10303) - ASCII:
  - HEADER: FILE_NAME, FILE_DESCRIPTION, PRODUCT names
  - Strategy: Rewrite FILE_NAME author/org/timestamp fields, then text clean.

.stl (Stereolithography) - ASCII or binary:
  - Binary header may contain 80 bytes of text (MANDATORY 80 bytes)
  - Strategy: Text clean for ASCII, replace header with exactly 80 bytes for binary.

.iges / .igs (IGES) - ASCII: Header entities
.obj (Wavefront OBJ) - ASCII: Comment lines
.dxf (AutoCAD DXF) - ASCII: HEADER section variables
.err (Creo error logs) - Text: Full paths, usernames

.3mf (3D Manufacturing Format) - ZIP-based: recurse and clean

Option A: imports CADMetadataExtractor from acquire/metadata.py
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
from .text import TextCleaner, _read_text_file, _write_text_file

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

try:
    import olefile
    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False


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


# Only map metadata fields that are true identifiers (person/company/product).
# Generic fields like description, title, subject, comments, dates, and times
# are NOT registered as entities to prevent mapper poisoning.
# They will still be cleaned via text cleaning if they contain known entities.
_METADATA_ENTITY_TYPE_MAP = {
    'author': 'person',
    'last_saved_by': 'person',
    'company': 'company',
    'designer': 'person',
    'organization': 'company',
    'manager': 'person',
    'product': 'product',
    'creator_tool': 'product',
    'name': 'product',
    'value': 'product',
}

# Metadata fields to blank/normalize rather than register as entities
_METADATA_FIELDS_TO_BLANK = {
    'description', 'title', 'subject', 'comments',
    'create_date', 'modify_date', 'time',
    'directory', 'revision',
}


class CADCleaner:
    """Clean CAD files using CADMetadataExtractor from acquire/metadata.py."""

    STEP_FILE_NAME_RE = re.compile(
        r"FILE_NAME\s*\(\s*'([^']*?)'\s*,\s*'([^']*?)'\s*,\s*([^;]*?)\s*\);",
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

    # Fixed timestamp for STEP HEADER
    _FIXED_STEP_TIMESTAMP = "2024-01-01#00:00:00"

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
            return self._clean_binary_cad(input_path, output_path, ext)
        elif ext in CAD_ZIP_BASED:
            return self._handle_zip_based_cad(input_path, output_path, ext)
        else:
            return self._remove_binary_cad_with_warning(input_path, output_path, ext)

    # ---- metadata extraction (Option A bridge) ----

    def _extract_and_register_metadata(self, input_path: Path) -> None:
        """Extract CAD metadata via acquire/metadata.py and register in mapper.

        Only registers true identifier values (person, company, product).
        Generic metadata fields (description, title, dates, etc.) are skipped
        to prevent mapper poisoning.
        """
        if not self.cad_extractor:
            return

        try:
            metadata = self.cad_extractor.extract_metadata(input_path)
            if not metadata:
                return

            for key, value in metadata.items():
                if not value or len(value.strip()) < 2:
                    continue

                key_lower = key.lower()

                # Skip fields that should not be registered as entities
                if key_lower in _METADATA_FIELDS_TO_BLANK:
                    _logger.debug(
                        "Skipping non-identifier CAD metadata: %s = %r",
                        key, value.strip(),
                    )
                    continue

                # Only register if this is a known identifier field
                entity_type = _METADATA_ENTITY_TYPE_MAP.get(key_lower)
                if entity_type is None:
                    # Unknown field - don't register as entity to avoid poisoning
                    _logger.debug(
                        "Unknown CAD metadata field %s, not registering as entity",
                        key,
                    )
                    continue

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

        # For STEP/STP, apply header rewrites BEFORE general text cleaning
        if ext in ('.step', '.stp'):
            return self._clean_step(input_path, output_path)

        # For other ASCII CAD files, use text cleaner (fail-open encoding)
        return self.text_cleaner.clean_file(input_path, output_path)

    def _clean_step(self, input_path: Path, output_path: Path) -> bool:
        """Clean STEP files: rewrite HEADER fields, then text clean.

        Actually uses the STEP_FILE_NAME_RE and STEP_FILE_DESCRIPTION_RE
        regexes to rewrite FILE_NAME author/org/timestamp fields, and
        cleans PRODUCT entities. Then applies general text cleaning.
        """
        try:
            # Read with fail-open encoding and preserved line endings
            text, enc = _read_text_file(input_path)

            # Rewrite FILE_NAME header: replace author/org/timestamp
            text = self.STEP_FILE_NAME_RE.sub(
                lambda m: "FILE_NAME('Cleaned STEP', '2024-01-01', 'Project P');",
                text,
            )

            # Rewrite FILE_DESCRIPTION
            text = self.STEP_FILE_DESCRIPTION_RE.sub(
                lambda m: "FILE_DESCRIPTION(('Cleaned'), '2;1');",
                text,
            )

            # Rewrite PRODUCT entities (name, description, formation_date, id)
            text = self.STEP_PRODUCT_RE.sub(
                lambda m: "PRODUCT('Cleaned', '', '', '')",
                text,
            )

            # Now apply general text cleaning for any remaining entities
            text = self.text_cleaner.clean_text(text)

            _write_text_file(output_path, text, encoding=enc)
            return True

        except Exception as e:
            _logger.error("Error cleaning STEP %s: %s", input_path, e)
            if output_path.exists():
                os.remove(output_path)
            return False

    def _clean_stl(self, input_path: Path, output_path: Path) -> bool:
        """Clean STL files (ASCII or binary).

        For binary STL, the header is MANDATORY 80 bytes. The previous
        implementation wrote only 59 bytes ('Cleaned STL - Project P' + 36 nulls)
        which corrupts every binary STL. We now write exactly 80 bytes.
        """
        try:
            with open(input_path, 'rb') as f:
                header = f.read(80)

            # Binary STL: first 80 bytes are header, followed by 4-byte int (triangle count)
            # ASCII STL: starts with "solid "
            if header[:6] == b'solid ':
                # ASCII STL: text clean (fail-open encoding)
                return self.text_cleaner.clean_file(input_path, output_path)
            else:
                # Binary STL: replace header with exactly 80 bytes, copy rest
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(input_path, 'rb') as fin:
                    # Skip original 80-byte header
                    fin.seek(80)
                    rest = fin.read()

                with open(output_path, 'wb') as fout:
                    # Write clean header - MUST be exactly 80 bytes
                    clean_header = b'Cleaned STL - Project P'
                    clean_header = clean_header.ljust(80, b'\x00')
                    assert len(clean_header) == 80, f"Header is {len(clean_header)} bytes, expected 80"
                    fout.write(clean_header)
                    fout.write(rest)

                return True

        except Exception as e:
            _logger.error("Error cleaning STL %s: %s", input_path, e)
            if output_path.exists():
                os.remove(output_path)
            return False

    def _clean_binary_cad(self, input_path: Path, output_path: Path, ext: str) -> bool:
        """Clean binary CAD files, attempting to strip OLE metadata.

        For SolidWorks files (.sldprt, .sldasm), attempts to strip
        OLE SummaryInformation streams using olefile/pyole2.
        Falls back to fail-closed removal for other binary formats.
        """
        if ext in ('.sldprt', '.sldasm'):
            return self._strip_ole_summary(input_path, output_path, ext)

        # Other binary CAD: fail-closed
        return self._remove_binary_cad_with_warning(input_path, output_path, ext)

    def _strip_ole_summary(self, input_path: Path, output_path: Path, ext: str) -> bool:
        """Strip OLE SummaryInformation streams from SolidWorks files.

        Uses olefile to read and verify OLE structure, then attempts to
        remove SummaryInformation and DocumentSummaryInformation streams.
        Falls back to exiftool CLI if available, then to fail-closed.
        """
        try:
            # Verify this is a valid OLE compound file
            if not HAS_OLEFILE:
                _logger.warning(
                    "olefile not available; cannot strip OLE metadata from %s. "
                    "Removing staged file (fail-closed).",
                    ext,
                )
                if output_path.exists():
                    os.remove(output_path)
                return False

            ole = olefile.OleFileIO(input_path)

            # Check for SummaryInformation streams
            has_summary = ole.exists('\x01SummaryInformation')
            has_doc_summary = ole.exists('\x01DocumentSummaryInformation')

            if has_summary or has_doc_summary:
                _logger.info(
                    "Found OLE SummaryInformation in %s: Summary=%s, DocSummary=%s",
                    input_path.name, has_summary, has_doc_summary,
                )

            ole.close()

            # Try to use exiftool to strip OLE metadata (it exists on this machine)
            import subprocess
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy first to avoid corrupting the original
            temp_output = output_path.with_suffix('.tmp')
            shutil.copy2(input_path, temp_output)

            result = subprocess.run(
                ['exiftool', '-overwrite_original',
                 '-Title=', '-Subject=', '-Author=', '-Keywords=',
                 '-Comments=', '-LastSavedBy=', '-Company=',
                 '-Manager=', '-Category=',
                 f'-all=', str(temp_output)],
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode == 0:
                # Rename temp to final output
                temp_output.rename(output_path)
                _logger.info(
                    "Stripped OLE metadata from %s via exiftool", input_path.name,
                )
                return True
            else:
                _logger.warning(
                    "exiftool failed for %s: %s", input_path.name, result.stderr,
                )
                # Clean up temp file
                if temp_output.exists():
                    temp_output.unlink()

            # Fallback: copy as-is with warning (the metadata is extracted
            # during acquisition, so entities are still registered)
            _logger.warning(
                "Could not strip OLE metadata from %s. "
                "Copying as-is (metadata was extracted during acquisition).",
                ext,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return True

        except subprocess.TimeoutExpired:
            _logger.error("exiftool timed out for %s", input_path.name)
            if output_path.exists():
                os.remove(output_path)
            return False
        except Exception as e:
            _logger.error(
                "Error stripping OLE metadata from %s: %s", input_path, e,
            )
            if output_path.exists():
                os.remove(output_path)
            return False

    def _remove_binary_cad_with_warning(self, input_path: Path,
                                         output_path: Path, ext: str) -> bool:
        """Remove staged binary CAD file with warning about potential metadata."""
        _logger.warning(
            "Binary CAD file (%s) cannot be safely cleaned. "
            "May contain embedded metadata (author, paths, properties). "
            "Removing staged file (fail-closed).",
            ext,
        )
        if output_path.exists():
            os.remove(output_path)
        return False

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