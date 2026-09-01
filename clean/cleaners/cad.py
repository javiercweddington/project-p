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
    '.slddrw',             # SolidWorks drawings — same container family;
                           # unrouted they hit the router's unknown-type
                           # fail-closed (18 quarantines on a live corpus)
    '.x_t', '.x_b', '.ipt', '.iam',
    '.dwg', '.fbx', '.blend',
    '.mpp',                # MS Project: OLE compound file; not CAD, but
                           # the OLE property scrub + binary surgery +
                           # scan gates are exactly the right treatment
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
        r"FILE_NAME\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,([^;]*?)\)\s*;",
        re.DOTALL,
    )

    # FILE_DESCRIPTION((description...), 'implementation_level');
    STEP_FILE_DESCRIPTION_RE = re.compile(
        r"FILE_DESCRIPTION\s*\(\s*\((.*?)\)\s*,\s*'([^']*)'\s*\)\s*;",
        re.DOTALL,
    )

    # Real STEP syntax: PRODUCT('id', 'name', 'description', (#ref, ...))
    # — three quoted fields followed by a reference set, NOT four quoted fields.
    STEP_PRODUCT_RE = re.compile(
        r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,",
    )

    # Generic STEP labels that must never be registered as entities
    # (they appear thousands of times in ordinary files).
    _STEP_GENERIC_NAMES = {
        'none', 'part', 'assembly', 'unknown', 'default', 'solid', 'body',
        'component', 'product', 'shape', 'open cascade step translator',
        'lens', 'cover', 'plate', 'housing', 'bracket', 'frame', 'panel',
        'screw', 'washer', 'gasket', 'base', 'top', 'bottom', 'left',
        'right', 'front', 'back', 'inner', 'outer', 'main',
    }

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
            source = str(input_path)

            # Rewrite FILE_NAME with the MANDATORY 7-field form:
            # (name, time_stamp, (author), (organization),
            #  preprocessor_version, originating_system, authorization)
            # Name is text-cleaned (may carry product identifiers);
            # timestamp is normalized; author/org/tool fields are blanked.
            def _file_name_sub(m: re.Match) -> str:
                cleaned_name = self.text_cleaner.clean_text(m.group(1), source=source)
                return (
                    f"FILE_NAME('{cleaned_name}', '2024-01-01T00:00:00', "
                    f"(''), (''), '', '', '');"
                )
            text = self.STEP_FILE_NAME_RE.sub(_file_name_sub, text, count=1)

            # Rewrite FILE_DESCRIPTION: clean the description strings,
            # preserve the implementation level (schema-relevant).
            def _file_desc_sub(m: re.Match) -> str:
                cleaned_desc = self.text_cleaner.clean_text(m.group(1), source=source)
                return f"FILE_DESCRIPTION(({cleaned_desc}), '{m.group(2)}');"
            text = self.STEP_FILE_DESCRIPTION_RE.sub(_file_desc_sub, text, count=1)

            # Pseudonymize PRODUCT id/name fields through the mapper so the
            # same part gets the same placeholder across all files. Generic
            # labels ('NONE', 'PART', ...) are never registered.
            def _product_sub(m: re.Match) -> str:
                # STEP escapes apostrophes as '' — a statement containing
                # them would mis-span the quoted-field regex and corrupt
                # the geometry file; leave such statements untouched.
                if "''" in m.group(0):
                    return m.group(0)

                def anon(value: str) -> str:
                    stripped = value.strip()
                    if (len(stripped) >= 3
                            and stripped.lower() not in self._STEP_GENERIC_NAMES):
                        return self.mapper.get_or_create(
                            'product', stripped, source=source)
                    return value
                cleaned_desc = self.text_cleaner.clean_text(m.group(3), source=source)
                return (f"PRODUCT('{anon(m.group(1))}', '{anon(m.group(2))}', "
                        f"'{cleaned_desc}',")
            text = self.STEP_PRODUCT_RE.sub(_product_sub, text)

            # Now apply general text cleaning for any remaining entities
            text = self.text_cleaner.clean_text(text, source=source)

            _write_text_file(output_path, text, encoding=enc)
            return True

        except Exception as e:
            _logger.error("Error cleaning STEP %s: %s", input_path, e)
            return False

    def _clean_stl(self, input_path: Path, output_path: Path) -> bool:
        """Clean STL files (ASCII or binary).

        For binary STL, the header is MANDATORY 80 bytes. The previous
        implementation wrote only 59 bytes ('Cleaned STL - Project P' + 36 nulls)
        which corrupts every binary STL. We now write exactly 80 bytes.
        """
        try:
            with open(input_path, 'rb') as f:
                sample = f.read(512)

            # ASCII STL starts with "solid " AND the following bytes are text.
            # A binary STL whose 80-byte header happens to start with "solid "
            # would corrupt if treated as text, so also require the sample to
            # be NUL-free and decodable.
            is_ascii_stl = False
            # Accept 'solid' followed by space OR newline (nameless ASCII
            # STLs are legal: 'solid\n...') — requiring the trailing space
            # misrouted them to the binary path, corrupting geometry.
            if (sample[:5] == b'solid'
                    and (len(sample) == 5 or sample[5:6] in b' \t\r\n')
                    and b'\x00' not in sample):
                try:
                    sample.decode('utf-8')
                    is_ascii_stl = True
                except UnicodeDecodeError:
                    is_ascii_stl = False

            if is_ascii_stl:
                # ASCII STL: clean the solid/endsolid names, then entity-clean
                text, enc = _read_text_file(input_path)
                source = str(input_path)
                text = re.sub(r'^(solid)[ \t]+[^\r\n]*', r'\1 cleaned',
                              text, flags=re.MULTILINE)
                text = re.sub(r'^(endsolid)[ \t]+[^\r\n]*', r'\1 cleaned',
                              text, flags=re.MULTILINE)
                text = self.text_cleaner.clean_text(text, source=source)
                _write_text_file(output_path, text, encoding=enc)
                return True
            else:
                # Binary STL: replace header with exactly 80 bytes, copy rest
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(input_path, 'rb') as fin:
                    # Skip original 80-byte header
                    fin.seek(80)
                    rest = fin.read()

                with open(output_path, 'wb') as fout:
                    # Clean header - the format mandates exactly 80 bytes
                    fout.write(b'Cleaned STL - Project P'.ljust(80, b'\x00')[:80])
                    fout.write(rest)

                return True

        except Exception as e:
            _logger.error("Error cleaning STL %s: %s", input_path, e)
            return False

    def _clean_binary_cad(self, input_path: Path, output_path: Path, ext: str) -> bool:
        """Clean binary CAD files, attempting to strip OLE metadata.

        For SolidWorks files (.sldprt, .sldasm), attempts to strip
        OLE SummaryInformation streams using olefile/pyole2.
        Falls back to fail-closed removal for other binary formats.
        """
        if ext in ('.sldprt', '.sldasm', '.slddrw', '.mpp'):
            # SolidWorks parts/assemblies/drawings and MS Project: OLE
            # compound (or newer opaque container) — property scrub +
            # binary surgery + scan gates, honoring
            # PROJECT_P_OPAQUE_BINARY for the non-OLE variants.
            return self._strip_ole_summary(input_path, output_path, ext)

        # Other binary CAD: fail-closed
        return self._remove_binary_cad_with_warning(input_path, output_path, ext)

    # OLE property-set streams live under the \x05 prefix (not \x01).
    _OLE_SUMMARY_STREAMS = ('\x05SummaryInformation',
                            '\x05DocumentSummaryInformation')

    def _strip_ole_summary(self, input_path: Path, output_path: Path, ext: str) -> bool:
        """Strip OLE SummaryInformation streams from SolidWorks files.

        Delegates to the shared OLE scrubber (ole_scrub), which zeroes
        the property streams, performs same-length entity surgery on any
        stray text in content streams, OCR-screens embedded preview
        bitmaps, and verifies with boundary-aware patterns (this class's
        old raw substring check false-quarantined clean parts on 3-letter
        initials inside binary noise).

        Note: exiftool is deliberately NOT used here — it cannot write
        OLE-based CAD formats, so an exiftool round-trip can never succeed.
        """
        from .ole_scrub import strip_ole_properties
        return strip_ole_properties(
            self.mapper, input_path, output_path, input_path.name)

    def _binary_contains_mapped_entity(self, data: bytes) -> bool:
        """Check raw bytes for any mapped entity in UTF-8 or UTF-16LE form.

        Case-insensitive for ASCII characters (bytes.lower() lowercases
        ASCII only, which matches how the entity needles are encoded).
        """
        lowered = data.lower()
        for mapping in self.mapper.mappings:
            value = mapping.original.strip()
            if len(value) < 3:
                continue
            for enc in ('utf-8', 'utf-16-le'):
                try:
                    needle = value.lower().encode(enc)
                except Exception:
                    continue
                if needle and needle in lowered:
                    return True
        return False

    def _remove_binary_cad_with_warning(self, input_path: Path,
                                         output_path: Path, ext: str) -> bool:
        """Remove staged binary CAD file with warning about potential metadata."""
        _logger.warning(
            "Binary CAD file (%s) cannot be safely cleaned. "
            "May contain embedded metadata (author, paths, properties). "
            "Fail-closed: returning False for pipeline quarantine.",
            ext,
        )
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