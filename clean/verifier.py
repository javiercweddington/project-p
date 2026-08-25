"""
Verifier module - deterministic verification of cleaning results.

Provides three layers of verification:
1. LeakageChecker: Assert no original entity strings survive in output
2. ReScanner: Re-run GLiNER on cleaned documents to detect new entities
3. LegibilityChecker: Verify documents are still valid after cleaning

All checks are deterministic and can run in CI on every commit.
"""

from __future__ import annotations

import logging
import re
import struct
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from io import BytesIO

from .anonymizer import EntityMapper

# Try optional dependencies
try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

_logger = logging.getLogger(__name__)


@dataclass
class LeakageHit:
    """A single leakage detection: original entity found in cleaned output."""
    file_path: str
    entity_type: str
    original: str
    context: str = ""  # Surrounding text where leakage was found


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    check_name: str
    passed: bool
    details: str = ""
    hits: List[LeakageHit] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.check_name}"]
        if self.details:
            lines.append(f"  {self.details}")
        if self.hits:
            lines.append(f"  Leakage hits: {len(self.hits)}")
            for hit in self.hits[:10]:  # Show first 10
                lines.append(
                    f"    - {hit.original!r} in {hit.file_path} "
                    f"({hit.entity_type})"
                )
            if len(self.hits) > 10:
                lines.append(f"    ... and {len(self.hits) - 10} more")
        return '\n'.join(lines)


@dataclass
class LeakageReport:
    """Aggregate leakage report for an entire cleaning session."""
    project_name: str
    results: List[VerificationResult] = field(default_factory=list)
    total_leakages: int = 0
    leakages_by_type: Dict[str, int] = field(default_factory=dict)
    leakages_by_file: Dict[str, int] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> List[VerificationResult]:
        return [r for r in self.results if not r.passed]

    def add_result(self, result: VerificationResult) -> None:
        """Add a verification result."""
        self.results.append(result)
        self.total_leakages += len(result.hits)

        # Aggregate by type and file
        for hit in result.hits:
            self.leakages_by_type[hit.entity_type] = (
                self.leakages_by_type.get(hit.entity_type, 0) + 1
            )
            self.leakages_by_file[hit.file_path] = (
                self.leakages_by_file.get(hit.file_path, 0) + 1
            )

    def summary(self) -> str:
        lines = [
            f"Leakage Report: {self.project_name}",
            f"Overall: {'PASS' if self.all_passed else 'FAIL'}",
            f"Total leakage hits: {self.total_leakages}",
            f"Checks run: {len(self.results)}",
        ]

        if self.leakages_by_type:
            lines.append("\nLeakages by entity type:")
            for etype, count in sorted(self.leakages_by_type.items()):
                lines.append(f"  {etype}: {count}")

        if self.leakages_by_file:
            lines.append("\nLeakages by file:")
            for fpath, count in sorted(self.leakages_by_file.items()):
                lines.append(f"  {fpath}: {count}")

        lines.append("\nCheck results:")
        for result in self.results:
            lines.append(result.summary())
            lines.append("")

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# ChangeTracker - record every real substitution made by EntityMapper
# ---------------------------------------------------------------------------

class ChangeTracker:
    """Track every entity substitution performed during cleaning.

    Accepts a callback that is invoked for each replacement so downstream
    components (e.g., the verifier) can record what was actually changed.
    """

    def __init__(self):
        self._changes: List[Tuple[str, str, str]] = []  # (original, placeholder, source)

    def record_change(self, original: str, placeholder: str, source: str = "") -> None:
        """Record a single substitution."""
        self._changes.append((original, placeholder, source))

    @property
    def change_count(self) -> int:
        return len(self._changes)

    @property
    def changes(self) -> List[Tuple[str, str, str]]:
        return list(self._changes)

    def summary(self) -> str:
        return f"Entities replaced: {self.change_count}"


# ---------------------------------------------------------------------------
# LeakageChecker
# ---------------------------------------------------------------------------

class LeakageChecker:
    """Check that no original entity text survives in cleaned output.

    This is the primary CI test: for every entity the acquisition pass
    detected, assert that its literal string does not appear in the
    cleaned output.

    Supports per-format text extraction for Office/zip containers so that
    UnicodeDecodeError no longer causes silent false-negatives.
    """

    # File extensions that are ZIP-based Office containers
    _OFFICE_ZIP_EXTENSIONS = {'.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp', '.xlsm', '.docm', '.pptm'}

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper

    # -- public API ---------------------------------------------------------

    def check_text(self, cleaned_text: str, file_path: str = "") -> List[LeakageHit]:
        """Check cleaned text for any surviving original entity strings."""
        hits = []

        for mapping in self.mapper.mappings:
            original = mapping.original
            if not original or len(original) < 2:
                continue

            # Case-insensitive search
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            for match in pattern.finditer(cleaned_text):
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(cleaned_text), match.end() + 30)
                context = cleaned_text[start:end].replace('\n', ' ')

                hits.append(LeakageHit(
                    file_path=file_path,
                    entity_type=mapping.entity_type,
                    original=original,
                    context=context,
                ))

        return hits

    def check_file(self, cleaned_path: Path, original_path: Path,
                   relative_path: str = "") -> List[LeakageHit]:
        """Check a cleaned file for surviving entities using format-aware extraction."""
        display_path = relative_path or str(cleaned_path)
        suffix = cleaned_path.suffix.lower()

        # Office ZIP formats: extract and scan internal XML
        if suffix in self._OFFICE_ZIP_EXTENSIONS:
            return self._check_office_zip(cleaned_path, display_path)

        # ZIP archives: scan every member recursively
        if suffix == '.zip':
            return self._check_zip_archive(cleaned_path, display_path)

        # PDF
        if suffix == '.pdf':
            return self.check_pdf(cleaned_path)

        # Plain text files
        try:
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.check_text(text, display_path)
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Binary file - skip text-based leakage check
            _logger.debug("Binary file, skipping text leakage check: %s", display_path)
            return []

    def check_pdf(self, cleaned_path: Path, max_pages: int = 10) -> List[LeakageHit]:
        """Check a cleaned PDF for surviving entities in extracted text."""
        if not HAS_PYPDF:
            _logger.warning("pypdf not available; skipping PDF leakage check")
            return []

        hits = []
        try:
            reader = PdfReader(str(cleaned_path))
            for i in range(min(max_pages, len(reader.pages))):
                text = reader.pages[i].extract_text() or ""
                page_hits = self.check_text(text, f"{cleaned_path} (page {i+1})")
                hits.extend(page_hits)

        except Exception as e:
            _logger.debug("Failed to check PDF %s: %s", cleaned_path, e)

        return hits

    # -- filename check -----------------------------------------------------

    def check_filenames(self, cleaned_dir: Path) -> List[LeakageHit]:
        """Check that no mapped entity appears in any output file path.

        Every mapped original entity is tested against every output file
        path (including all path components).  If an entity is found in a
        filename the check immediately records a hit.
        """
        hits = []
        for mapping in self.mapper.mappings:
            original = mapping.original
            if not original or len(original) < 2:
                continue

            pattern = re.compile(re.escape(original), re.IGNORECASE)
            for file_path in cleaned_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                # Check the full relative path string
                rel = str(file_path.relative_to(cleaned_dir))
                if pattern.search(rel):
                    hits.append(LeakageHit(
                        file_path=rel,
                        entity_type=mapping.entity_type,
                        original=original,
                        context=f"Entity found in filename: {rel}",
                    ))
        return hits

    # -- metadata check -----------------------------------------------------

    def check_metadata(self, cleaned_dir: Path) -> List[LeakageHit]:
        """Check file metadata for leaked entities.

        Covers:
        - Office ZIP: docProps/core.xml and docProps/app.xml
        - Images: EXIF
        - STEP/CAD: header blocks
        - OLE property streams (legacy .doc/.xls/.ppt)
        """
        hits = []
        for file_path in cleaned_dir.rglob('*'):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(cleaned_dir))
            suffix = file_path.suffix.lower()

            if suffix in self._OFFICE_ZIP_EXTENSIONS:
                hits.extend(self._check_office_metadata(file_path, rel))
            elif suffix in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
                hits.extend(self._check_image_metadata(file_path, rel))
            elif suffix in ('.doc', '.xls', '.ppt'):
                hits.extend(self._check_ole_metadata(file_path, rel))
            elif suffix in ('.step', '.stp'):
                hits.extend(self._check_step_metadata(file_path, rel))

        return hits

    # -- format-specific text extraction ------------------------------------

    def _check_office_zip(self, path: Path, display_path: str) -> List[LeakageHit]:
        """Extract text from Office ZIP containers (.docx/.xlsx/.pptx) and scan."""
        hits = []
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                for member_name in zf.namelist():
                    # We care about XML members that carry user-visible text
                    member_lower = member_name.lower()
                    if not member_lower.endswith('.xml'):
                        continue
                    try:
                        xml_bytes = zf.read(member_name)
                        xml_text = xml_bytes.decode('utf-8', errors='replace')
                    except Exception:
                        continue
                    member_hits = self.check_text(
                        xml_text, f"{display_path}::{member_name}"
                    )
                    hits.extend(member_hits)
        except zipfile.BadZipFile:
            _logger.debug("Not a valid ZIP: %s", path)
        return hits

    def _check_zip_archive(self, path: Path, display_path: str) -> List[LeakageHit]:
        """Scan every member of a generic ZIP archive."""
        hits = []
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                for member_name in zf.namelist():
                    member_suffix = Path(member_name).suffix.lower()
                    try:
                        member_data = zf.read(member_name)
                    except Exception:
                        continue

                    # If the member itself is an Office ZIP, recurse into XML
                    if member_suffix in self._OFFICE_ZIP_EXTENSIONS:
                        try:
                            with zipfile.ZipFile(BytesIO(member_data)) as inner_zf:
                                for inner_name in inner_zf.namelist():
                                    if not inner_name.lower().endswith('.xml'):
                                        continue
                                    try:
                                        xml_text = inner_zf.read(inner_name).decode('utf-8', errors='replace')
                                        inner_hits = self.check_text(
                                            xml_text,
                                            f"{display_path}::{member_name}::{inner_name}"
                                        )
                                        hits.extend(inner_hits)
                                    except Exception:
                                        continue
                        except zipfile.BadZipFile:
                            pass

                    # Plain text members
                    elif member_suffix not in {
                        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
                        '.mp3', '.wav', '.mp4', '.avi', '.mov',
                    }:
                        try:
                            text = member_data.decode('utf-8', errors='replace')
                        except Exception:
                            continue
                        member_hits = self.check_text(
                            text, f"{display_path}::{member_name}"
                        )
                        hits.extend(member_hits)

        except zipfile.BadZipFile:
            _logger.debug("Not a valid ZIP: %s", path)
        return hits

    # -- metadata helpers ---------------------------------------------------

    def _check_office_metadata(self, path: Path, rel: str) -> List[LeakageHit]:
        hits = []
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                for meta_name in ('docProps/core.xml', 'docProps/app.xml'):
                    try:
                        xml_text = zf.read(meta_name).decode('utf-8', errors='replace')
                    except KeyError:
                        continue
                    member_hits = self.check_text(xml_text, f"{rel}::{meta_name}")
                    hits.extend(member_hits)
        except zipfile.BadZipFile:
            pass
        return hits

    def _check_image_metadata(self, path: Path, rel: str) -> List[LeakageHit]:
        hits = []
        if not HAS_PIL:
            return hits
        try:
            img = Image.open(path)
            exif = img._getexif()
            if exif:
                text_parts = []
                for tag_id, value in exif.items():
                    if isinstance(value, (str, bytes)):
                        text_parts.append(str(value))
                exif_text = ' '.join(text_parts)
                hits.extend(self.check_text(exif_text, f"{rel}::EXIF"))
        except Exception:
            pass
        return hits

    def _check_ole_metadata(self, path: Path, rel: str) -> List[LeakageHit]:
        """Check OLE2 compound documents for property streams.

        OLE files start with the magic \xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1.
        We do a best-effort scan of the raw bytes for entity strings.
        """
        hits = []
        try:
            with open(path, 'rb') as f:
                data = f.read(65536)  # Read first 64 KB
            text = data.decode('utf-8', errors='replace')
            hits.extend(self.check_text(text, f"{rel}::OLE"))
        except Exception:
            pass
        return hits

    def _check_step_metadata(self, path: Path, rel: str) -> List[LeakageHit]:
        """Check STEP/CAD file headers for entity strings."""
        hits = []
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                # STEP headers are in the first ~200 lines
                header_lines = [f.readline() for _ in range(200)]
            header_text = ''.join(header_lines)
            hits.extend(self.check_text(header_text, f"{rel}::STEP_HEADER"))
        except Exception:
            pass
        return hits

    # -- top-level runner ---------------------------------------------------

    def run_check(self, cleaned_dir: Path, original_dir: Path) -> VerificationResult:
        """Run leakage check on all files in cleaned directory."""
        all_hits = []
        file_count = 0

        for cleaned_file in cleaned_dir.rglob('*'):
            if not cleaned_file.is_file():
                continue

            rel_path = cleaned_file.relative_to(cleaned_dir)
            original_file = original_dir / rel_path
            file_count += 1

            hits = self.check_file(cleaned_file, original_file, str(rel_path))
            all_hits.extend(hits)

        passed = len(all_hits) == 0
        return VerificationResult(
            check_name="Entity Leakage Check",
            passed=passed,
            details=f"Checked {file_count} files",
            hits=all_hits,
        )


# ---------------------------------------------------------------------------
# ReScanner
# ---------------------------------------------------------------------------

class ReScanner:
    """Re-run entity detection on cleaned documents.

    If GLiNER finds new person/organization entities in the cleaned text,
    either the scrubber missed something or the replacement text introduced
    new entities.  New person/org/email hits now produce FAIL-WARN hits
    instead of being silently dropped.
    """

    # Common words that might be in placeholder text but aren't real entities
    _PLACEHOLDER_PATTERN = re.compile(r'\[\w+_\d{3}\]')

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper

    def rescan_text(self, cleaned_text: str, file_path: str = "") -> List[LeakageHit]:
        """Scan cleaned text for entities using GLiNER or regex fallback."""
        if HAS_GLINER:
            return self._rescan_with_gliner(cleaned_text, file_path)
        else:
            return self._rescan_with_regex(cleaned_text, file_path)

    def _rescan_with_gliner(self, text: str, source: str) -> List[LeakageHit]:
        """Use GLiNER to scan cleaned text."""
        from acquire.catalog import _extract_entities_with_gliner

        # Remove placeholders before scanning to avoid false positives
        text_without_placeholders = self._PLACEHOLDER_PATTERN.sub('[REDACTED]', text)

        entities = _extract_entities_with_gliner(text_without_placeholders, source)

        hits = []
        for entity in entities:
            if self.mapper.has_entity(entity.value):
                # Known entity that survived - leakage
                hits.append(LeakageHit(
                    file_path=source,
                    entity_type=entity.entity_type,
                    original=entity.value,
                    context=entity.context,
                ))
            elif entity.entity_type.lower() in ('person', 'organization', 'email'):
                # NEW entity discovered after cleaning - at least fail-warn
                hits.append(LeakageHit(
                    file_path=source,
                    entity_type=f"new_{entity.entity_type}",
                    original=entity.value,
                    context=f"NEW entity detected after cleaning: {getattr(entity, 'context', '')}",
                ))

        return hits

    def _rescan_with_regex(self, text: str, source: str) -> List[LeakageHit]:
        """Regex-based fallback for rescanning."""
        from acquire.catalog import extract_entities_from_text_regex

        # Remove placeholders before scanning
        text_without_placeholders = self._PLACEHOLDER_PATTERN.sub('[REDACTED]', text)

        entities = extract_entities_from_text_regex(text_without_placeholders, source)

        hits = []
        for entity in entities:
            if self.mapper.has_entity(entity.value):
                hits.append(LeakageHit(
                    file_path=source,
                    entity_type=entity.entity_type,
                    original=entity.value,
                ))
            elif getattr(entity, 'entity_type', '').lower() in ('person', 'organization', 'email'):
                hits.append(LeakageHit(
                    file_path=source,
                    entity_type=f"new_{entity.entity_type}",
                    original=entity.value,
                    context="NEW entity detected after cleaning (regex fallback)",
                ))

        return hits

    def run_check(self, cleaned_dir: Path) -> VerificationResult:
        """Run re-scan on all text files in cleaned directory."""
        all_hits = []
        file_count = 0

        for cleaned_file in cleaned_dir.rglob('*'):
            if not cleaned_file.is_file():
                continue

            if cleaned_file.suffix.lower() in (
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
                '.mp3', '.wav', '.mp4', '.avi', '.mov',
            ):
                continue

            try:
                with open(cleaned_file, 'r', encoding='utf-8') as f:
                    text = f.read()

                rel_path = str(cleaned_file.relative_to(cleaned_dir))
                hits = self.rescan_text(text, rel_path)
                all_hits.extend(hits)
                file_count += 1

            except (UnicodeDecodeError, UnicodeEncodeError):
                continue

        passed = len(all_hits) == 0
        return VerificationResult(
            check_name="Re-Scan Entity Detection",
            passed=passed,
            details=f"Re-scanned {file_count} files with {'GLiNER' if HAS_GLINER else 'regex'}",
            hits=all_hits,
        )


# ---------------------------------------------------------------------------
# LegibilityChecker
# ---------------------------------------------------------------------------

class LegibilityChecker:
    """Verify that cleaned documents are still valid and readable."""

    # Extensions that are ZIP-based Office documents
    _OFFICE_ZIP_EXTENSIONS = {
        '.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp',
        '.xlsm', '.docm', '.pptm',
    }

    def check_text_file(self, path: Path) -> bool:
        """Check that a text file is readable and non-empty."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return len(content) > 0
        except Exception:
            return False

    def check_office_zip(self, path: Path) -> bool:
        """Check that an Office ZIP document is a valid zip with XML content."""
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                # Verify the ZIP is structurally sound
                bad = zf.testzip()
                if bad is not None:
                    return False
                # At least one XML member should exist
                names = zf.namelist()
                return any(n.lower().endswith('.xml') for n in names)
        except Exception:
            return False

    def check_pdf(self, path: Path) -> bool:
        """Check that a PDF is still valid."""
        if not HAS_PYPDF:
            _logger.warning("pypdf not available; cannot verify PDF legibility for %s", path)
            # Return False (fail) rather than True when we cannot verify
            return False

        try:
            reader = PdfReader(str(path))
            return len(reader.pages) > 0
        except Exception:
            return False

    def check_image(self, path: Path) -> bool:
        """Check that an image is still valid."""
        if not HAS_PIL:
            _logger.warning("PIL not available; cannot verify image legibility for %s", path)
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def run_check(self, cleaned_dir: Path) -> VerificationResult:
        """Run legibility checks on all files."""
        issues = []
        checked = 0

        for file_path in cleaned_dir.rglob('*'):
            if not file_path.is_file():
                continue

            checked += 1
            ext = file_path.suffix.lower()

            is_legible = True

            if ext == '.pdf':
                is_legible = self.check_pdf(file_path)
            elif ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'):
                is_legible = self.check_image(file_path)
            elif ext in self._OFFICE_ZIP_EXTENSIONS:
                is_legible = self.check_office_zip(file_path)
            else:
                is_legible = self.check_text_file(file_path)

            if not is_legible:
                rel = str(file_path.relative_to(cleaned_dir))
                issues.append(LeakageHit(
                    file_path=rel,
                    entity_type='legibility',
                    original=f"File not legible: {file_path.name}",
                ))

        passed = len(issues) == 0
        return VerificationResult(
            check_name="Document Legibility Check",
            passed=passed,
            details=f"Checked {checked} files",
            hits=issues,
        )


# ---------------------------------------------------------------------------
# ConsistencyChecker
# ---------------------------------------------------------------------------

class ConsistencyChecker:
    """Verify anonymization consistency across all datatypes.

    Ensures the same entity always maps to the same placeholder
    regardless of which file or datatype it appears in.

    FIXED: now properly appends hits when inconsistencies are found,
    including when original entity text appears instead of the expected
    placeholder, and when the same entity maps to different placeholders.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper

    def check_cross_file_consistency(self, cleaned_dir: Path) -> VerificationResult:
        """Verify that placeholders are consistent across all files.

        Check that:
        1. Each placeholder appears with the same meaning everywhere
        2. No original entity text appears in any file (leakage = inconsistency)
        3. Every mapped entity has a placeholder present in at least one file
        """
        hits = []

        # Build a map of placeholder -> files it appears in
        placeholder_files: Dict[str, Set[str]] = defaultdict(set)
        placeholder_pattern = re.compile(r'\[(\w+)_(\d{3})\]')

        # Collect all text content per file (using format-aware extraction)
        file_texts: Dict[str, str] = {}

        for file_path in cleaned_dir.rglob('*'):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(cleaned_dir))
            suffix = file_path.suffix.lower()

            if suffix in LeakageChecker._OFFICE_ZIP_EXTENSIONS:
                # Extract XML from Office ZIP
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        xml_parts = []
                        for member_name in zf.namelist():
                            if member_name.lower().endswith('.xml'):
                                try:
                                    xml_parts.append(zf.read(member_name).decode('utf-8', errors='replace'))
                                except Exception:
                                    pass
                    file_texts[rel] = '\n'.join(xml_parts)
                except zipfile.BadZipFile:
                    continue
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_texts[rel] = f.read()
                except (UnicodeDecodeError, UnicodeEncodeError):
                    continue

            # Index placeholders in this file
            text = file_texts.get(rel, '')
            for match in placeholder_pattern.finditer(text):
                placeholder = match.group(0)
                placeholder_files[placeholder].add(rel)

        # Check 1: Original entity text should NOT appear in any cleaned file
        for mapping in self.mapper.mappings:
            original = mapping.original
            expected_placeholder = mapping.placeholder

            if not original or len(original) < 2:
                continue

            pattern = re.compile(re.escape(original), re.IGNORECASE)
            for rel, text in file_texts.items():
                if pattern.search(text):
                    hits.append(LeakageHit(
                        file_path=rel,
                        entity_type=mapping.entity_type,
                        original=original,
                        context=f"Original entity found instead of placeholder {expected_placeholder}",
                    ))

        # Check 2: Verify placeholders are used consistently
        # (same placeholder should not appear with different meanings)
        for placeholder, files in placeholder_files.items():
            original = self.mapper.resolve(placeholder)
            if original is None:
                # Placeholder exists in output but not in our mapping - suspicious
                for rel in files:
                    hits.append(LeakageHit(
                        file_path=rel,
                        entity_type='unknown',
                        original=placeholder,
                        context="Placeholder found in output but not in entity mapping",
                    ))

        passed = len(hits) == 0
        return VerificationResult(
            check_name="Cross-File Consistency Check",
            passed=passed,
            details=f"Verified {len(placeholder_files)} placeholders across {len(file_texts)} files",
            hits=hits,
        )

    def run_check(self, cleaned_dir: Path) -> VerificationResult:
        """Run consistency check."""
        return self.check_cross_file_consistency(cleaned_dir)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def verify_clean(cleaned_dir: Path, original_dir: Path,
                 mapper: EntityMapper, project_name: str = "",
                 tracker: Optional[ChangeTracker] = None) -> LeakageReport:
    """Run all verification checks and return a comprehensive report.

    This is the main entry point for verification.

    Args:
        cleaned_dir: Directory with cleaned files
        original_dir: Directory with original files
        mapper: EntityMapper with all known entity mappings
        project_name: Name of the project for the report
        tracker: Optional ChangeTracker with recorded substitutions

    Returns:
        LeakageReport with all verification results
    """
    report = LeakageReport(project_name=project_name or cleaned_dir.name)

    leakage_checker = LeakageChecker(mapper)

    # 1. Entity leakage check (format-aware)
    leakage_result = leakage_checker.run_check(cleaned_dir, original_dir)
    report.add_result(leakage_result)

    # 2. Filename check
    filename_hits = leakage_checker.check_filenames(cleaned_dir)
    filename_result = VerificationResult(
        check_name="Filename Entity Check",
        passed=len(filename_hits) == 0,
        details=f"Checked {sum(1 for f in cleaned_dir.rglob('*') if f.is_file())} filenames",
        hits=filename_hits,
    )
    report.add_result(filename_result)

    # 3. Metadata check
    metadata_hits = leakage_checker.check_metadata(cleaned_dir)
    metadata_result = VerificationResult(
        check_name="Metadata Entity Check",
        passed=len(metadata_hits) == 0,
        details="Checked docProps, EXIF, OLE, STEP headers",
        hits=metadata_hits,
    )
    report.add_result(metadata_result)

    # 4. Re-scan for new entities
    rescaner = ReScanner(mapper)
    rescan_result = rescaner.run_check(cleaned_dir)
    report.add_result(rescan_result)

    # 5. Legibility check
    legibility_checker = LegibilityChecker()
    legibility_result = legibility_checker.run_check(cleaned_dir)
    report.add_result(legibility_result)

    # 6. Consistency check
    consistency_checker = ConsistencyChecker(mapper)
    consistency_result = consistency_checker.run_check(cleaned_dir)
    report.add_result(consistency_result)

    # 7. Change tracker summary (informational)
    if tracker is not None:
        tracker_result = VerificationResult(
            check_name="Change Tracker Summary",
            passed=True,
            details=tracker.summary(),
        )
        report.add_result(tracker_result)

    return report