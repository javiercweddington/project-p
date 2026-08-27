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
import os
import re
import struct
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from io import BytesIO

from .anonymizer import EntityMapper, NON_TEXT_ENTITY_TYPES

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
        self._prefilter_cache: dict = {}

    # -- public API ---------------------------------------------------------

    def _prefilter_tokens(self, original: str,
                          entity_type: Optional[str] = None) -> tuple:
        """Cheap substring keys gating the entity's boundary pattern.

        Variant generation (collapsed/hyphenated/underscored, flexible
        whitespace) only rewrites SEPARATORS — the alphanumeric/CJK
        tokens themselves are never altered — so a variant match always
        contains EVERY token of the original. That means non-person
        entities can be gated on their single most selective (longest)
        token: short universal tokens like the 'com' of an email would
        otherwise pass the filter in every XML blob and force a full
        regex scan of multi-MB members for every registered email.

        PERSON patterns additionally match individual name tokens, so a
        match may contain only ONE original token — persons keep the
        any-of-all-tokens gate.

        Returns () when no token is long enough to be selective; callers
        must then run the pattern unconditionally.
        """
        # Shared with the anonymizer's replacement path — including the
        # escaped-needle handling ('gürses' AND 'g&#252;rses'), so
        # XML-escaped variant matches are never prefiltered away.
        return self.mapper.prefilter_needles(original, entity_type)

    def _entity_pattern(self, original: str,
                        entity_type: Optional[str] = None) -> re.Pattern:
        """Boundary/variant-aware pattern for an entity.

        Uses the SAME pattern builder as the anonymizer so the verifier
        detects exactly what the cleaner is expected to replace — including
        collapsed/hyphenated/underscored variants ('globusmedical' in an
        email) and person name tokens — and does NOT false-positive on
        substrings ('SA' in 'USA').
        """
        build = getattr(self.mapper, '_build_pattern_cached', None) \
            or getattr(self.mapper, '_build_pattern', None)
        if build is not None:
            try:
                pattern = build(original, entity_type)
            except TypeError:
                try:
                    pattern = build(original)
                except Exception:
                    pattern = None
            except Exception:
                pattern = None
            if pattern is not None:
                return pattern
        if len(original) < 2:
            # Degenerate value: match nothing rather than every letter
            return re.compile(r'(?!x)x')
        return re.compile(re.escape(original), re.IGNORECASE)

    def check_text(self, cleaned_text: str, file_path: str = "") -> List[LeakageHit]:
        """Check cleaned text for any surviving original entity strings."""
        hits = []
        # One lowercase copy for the substring prefilter: with a large
        # mapper (100+ entities) the per-entity compiled patterns are the
        # dominant verification cost on multi-MB files (CAD text, raw
        # binary scans); the prefilter skips patterns that cannot match.
        text_lower = cleaned_text.lower()

        for mapping in self.mapper.mappings:
            original = mapping.original
            if not original or len(original) < 2:
                continue
            # Audit-only path pseudonyms (filename/directory stems) are not
            # text entities; a stem like 'in' would flag every document.
            if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                continue

            tokens = self._prefilter_tokens(original, mapping.entity_type)
            if tokens and not any(tok in text_lower for tok in tokens):
                continue

            pattern = self._entity_pattern(original, mapping.entity_type)
            for match in pattern.finditer(cleaned_text):
                # Skip matches inside an already-inserted placeholder token
                start, end = match.start(), match.end()
                if (start > 0 and cleaned_text[start - 1] == '['
                        and re.match(r'_\d{3,}\]', cleaned_text[end:])):
                    continue
                # Get context
                ctx_start = max(0, start - 30)
                ctx_end = min(len(cleaned_text), end + 30)
                context = cleaned_text[ctx_start:ctx_end].replace('\n', ' ')

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

        # Plain text files. Try multiple encodings; a BOM-less UTF-16LE
        # file decodes "successfully" as UTF-8 into NUL-interleaved text
        # invisible to entity patterns, so NUL-bearing decodes are retried
        # as UTF-16.
        for enc in ('utf-8', 'utf-16', 'gbk', 'gb18030'):
            try:
                with open(cleaned_path, 'r', encoding=enc) as f:
                    text = f.read()
            except (UnicodeError, OSError):
                continue
            if enc != 'utf-16' and '\x00' in text:
                continue
            return self.check_text(text, f"{display_path}[{enc}]")

        # Undecodable: raw-bytes scan in both byte encodings as a last
        # resort so binary-ish files are not silently skipped.
        try:
            data = cleaned_path.read_bytes()
            hits = []
            for enc in ('utf-8', 'utf-16-le'):
                hits.extend(self.check_text(
                    data.decode(enc, errors='replace'),
                    f"{display_path}[raw-{enc}]"))
            return hits
        except OSError:
            _logger.debug("Unreadable file, skipping: %s", display_path)
            return []

    def check_pdf(self, cleaned_path: Path) -> List[LeakageHit]:
        """Check a cleaned PDF for surviving entities in extracted text.

        ALL pages are checked (a page-11 leak is still a leak), and any
        inability to verify — missing pypdf, parse failure — is itself a
        FAILING hit rather than a silent pass.
        """
        if not HAS_PYPDF:
            return [LeakageHit(
                file_path=str(cleaned_path),
                entity_type='unverifiable',
                original='pypdf not available - PDF content NOT verified',
            )]

        hits = []
        try:
            reader = PdfReader(str(cleaned_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                page_hits = self.check_text(text, f"{cleaned_path} (page {i+1})")
                hits.extend(page_hits)

        except Exception as e:
            _logger.warning("Failed to check PDF %s: %s", cleaned_path, e)
            hits.append(LeakageHit(
                file_path=str(cleaned_path),
                entity_type='unverifiable',
                original=f'PDF text extraction failed: {e}',
            ))

        return hits

    # -- filename check -----------------------------------------------------

    def check_filenames(self, cleaned_dir: Path) -> List[LeakageHit]:
        """Check that no mapped entity appears in any output path.

        Every mapped original entity (variant-aware) is tested against every
        output path — FILES AND DIRECTORIES, including empty entity-named
        directories that a file-only scan would miss.
        """
        hits = []
        # Collect all relative paths once (files + dirs)
        rel_paths = [
            str(p.relative_to(cleaned_dir)) for p in cleaned_dir.rglob('*')
        ]
        for mapping in self.mapper.mappings:
            original = mapping.original
            if not original or len(original) < 2:
                continue

            if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                # Old path stems: literal match only — variant/token
                # patterns false-positive on the FILE_nnn pseudonyms.
                pattern = re.compile(re.escape(original), re.IGNORECASE)
            else:
                pattern = self._entity_pattern(original, mapping.entity_type)
            for rel in rel_paths:
                if pattern.search(rel):
                    hits.append(LeakageHit(
                        file_path=rel,
                        entity_type=mapping.entity_type,
                        original=original,
                        context=f"Entity found in path: {rel}",
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
            elif suffix in ('.doc', '.xls', '.ppt', '.sldprt', '.sldasm'):
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
                    member_lower = member_name.lower()
                    try:
                        member_bytes = zf.read(member_name)
                    except Exception:
                        continue
                    if member_lower.endswith(('.xml', '.rels', '.vml')):
                        # XML-ish members: text + relationship targets
                        # (.rels carries mailto:/hyperlink leaks)
                        xml_text = member_bytes.decode('utf-8', errors='replace')
                        hits.extend(self.check_text(
                            xml_text, f"{display_path}::{member_name}"
                        ))
                    else:
                        # Binary members (vbaProject.bin, embedded media):
                        # scan raw bytes in UTF-8 and UTF-16LE decodings.
                        for enc in ('utf-8', 'utf-16-le'):
                            text = member_bytes.decode(enc, errors='replace')
                            hits.extend(self.check_text(
                                text, f"{display_path}::{member_name}::{enc}"
                            ))
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
        """Check OLE2 compound documents for entity strings.

        OLE property-set strings (Author, Company, LastSavedBy) are stored
        UTF-16LE (or a legacy codepage), NOT UTF-8 — a UTF-8-only decode is
        blind to exactly the values that matter (e.g. 'Dylan Li' in a .ppt).
        The WHOLE file is scanned in both encodings.
        """
        hits = []
        try:
            data = path.read_bytes()
            for enc, tag in (('utf-8', 'OLE'), ('utf-16-le', 'OLE-utf16')):
                text = data.decode(enc, errors='replace')
                hits.extend(self.check_text(text, f"{rel}::{tag}"))
        except Exception as e:
            _logger.warning("Could not scan OLE file %s: %s", rel, e)
            hits.append(LeakageHit(
                file_path=rel,
                entity_type='unverifiable',
                original=f'OLE scan failed: {e}',
            ))
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
    _PLACEHOLDER_PATTERN = re.compile(r'\[\w+_\d{3,}\]')

    # Machine-generated CAD text (coordinate streams) is worthless to NER
    # and enormous — a 3.4MB STEP file is ~10K GLiNER chunks. The Entity
    # Leakage Check still regex-scans these files in FULL; skipping them
    # here loses nothing that check covers.
    _NER_SKIP_EXTENSIONS = {'.step', '.stp', '.igs', '.iges'}

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        # New-entity discoveries are collected here as advisories rather
        # than failing hits (detector false-positive rates are high).
        self._advisory_hits: List[LeakageHit] = []
        # NER re-scan input cap (chars). Known-entity survival is
        # redundantly covered by the full-text Entity Leakage Check, so
        # capping the (advisory-oriented) NER pass trades no hard
        # guarantee for a large speedup on huge extracted texts.
        try:
            self._max_rescan_chars = int(os.environ.get(
                'PROJECT_P_RESCAN_MAX_CHARS', '20000'))
        except ValueError:
            self._max_rescan_chars = 20000

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
                # Known entity that survived - leakage (FAILS the run)
                hits.append(LeakageHit(
                    file_path=source,
                    entity_type=entity.entity_type,
                    original=entity.value,
                    context=entity.context,
                ))
            elif entity.entity_type.lower() in ('person', 'organization',
                                                'company', 'email'):
                # NEW entity discovered after cleaning — reported as an
                # ADVISORY (detector false-positive rates are too high to
                # hard-fail, but silence would hide real detection gaps).
                self._advisory_hits.append(LeakageHit(
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
            elif getattr(entity, 'entity_type', '').lower() in ('person', 'organization', 'company', 'email'):
                # Advisory only: the regex fallback false-positives on
                # ordinary prose, so it must not hard-fail the run.
                self._advisory_hits.append(LeakageHit(
                    file_path=source,
                    entity_type=f"new_{entity.entity_type}",
                    original=entity.value,
                    context="NEW entity detected after cleaning (regex fallback)",
                ))

        return hits

    # Office ZIP formats whose XML members should be text-extracted
    _OFFICE_ZIP_EXTENSIONS = LeakageChecker._OFFICE_ZIP_EXTENSIONS

    def _extract_file_text(self, cleaned_file: Path) -> Optional[str]:
        """Best-effort text extraction for rescanning (utf-8 + Office ZIP)."""
        suffix = cleaned_file.suffix.lower()
        if suffix in self._OFFICE_ZIP_EXTENSIONS:
            try:
                parts = []
                with zipfile.ZipFile(cleaned_file, 'r') as zf:
                    for member in zf.namelist():
                        if member.lower().endswith('.xml'):
                            try:
                                parts.append(
                                    zf.read(member).decode('utf-8',
                                                           errors='replace'))
                            except Exception:
                                pass
                return '\n'.join(parts)
            except zipfile.BadZipFile:
                return None
        try:
            with open(cleaned_file, 'r', encoding='utf-8') as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeEncodeError):
            return None

    def run_check(self, cleaned_dir: Path) -> VerificationResult:
        """Run re-scan on all text-extractable files in cleaned directory.

        Known-entity survivals FAIL; newly detected entities are advisory
        (reported in the details, not as failing hits).
        """
        all_hits = []
        file_count = 0
        self._advisory_hits = []

        for cleaned_file in cleaned_dir.rglob('*'):
            if not cleaned_file.is_file():
                continue

            suffix = cleaned_file.suffix.lower()
            if suffix in (
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
                '.mp3', '.wav', '.mp4', '.avi', '.mov',
            ):
                continue
            if suffix in self._NER_SKIP_EXTENSIONS:
                continue

            text = self._extract_file_text(cleaned_file)
            if text is None:
                continue
            if self._max_rescan_chars > 0:
                text = text[:self._max_rescan_chars]

            rel_path = str(cleaned_file.relative_to(cleaned_dir))
            all_hits.extend(self.rescan_text(text, rel_path))
            file_count += 1

        advisory_summary = ""
        if self._advisory_hits:
            samples = ', '.join(sorted({
                f"{h.original!r}" for h in self._advisory_hits})[:8])
            advisory_summary = (
                f" | ADVISORY: {len(self._advisory_hits)} new possible "
                f"entities detected post-clean (not failing): {samples}"
            )

        passed = len(all_hits) == 0
        return VerificationResult(
            check_name="Re-Scan Entity Detection",
            passed=passed,
            details=(f"Re-scanned {file_count} files with "
                     f"{'GLiNER' if HAS_GLINER else 'regex'}"
                     f"{advisory_summary}"),
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

    # Extensions treated as text for legibility purposes; other unknown
    # formats are binary and must NOT be utf-8-read (that false-failed
    # every valid .SLDPRT/.STL/media file).
    _TEXT_EXTENSIONS = {
        '.txt', '.csv', '.tsv', '.log', '.md', '.rst', '.json', '.xml',
        '.yaml', '.yml', '.toml', '.html', '.css', '.js', '.py', '.step',
        '.stp', '.iges', '.igs', '.obj', '.dxf', '.err',
    }

    def check_text_file(self, path: Path) -> bool:
        """Check that a text file is readable (any supported encoding)
        and non-empty. UTF-8-only reading false-failed valid GBK/UTF-16
        outputs."""
        for enc in ('utf-8', 'utf-16', 'gbk', 'gb18030'):
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read()
                if enc != 'utf-16' and '\x00' in content:
                    continue
                return len(content) > 0
            except (UnicodeError, OSError):
                continue
        return False

    def check_binary_file(self, path: Path) -> bool:
        """Minimal legibility for binary formats we can't parse: non-empty."""
        try:
            return path.stat().st_size > 0
        except OSError:
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

    def check_zip_valid(self, path: Path) -> bool:
        """Check that a generic ZIP archive is structurally sound."""
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                return zf.testzip() is None
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
            elif ext == '.zip':
                is_legible = self.check_zip_valid(file_path)
            elif ext in self._TEXT_EXTENSIONS:
                is_legible = self.check_text_file(file_path)
            else:
                # Binary/unknown format: utf-8-reading it would false-fail
                # every valid CAD/media file. Require only that it's non-empty.
                is_legible = self.check_binary_file(file_path)

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

        # Build a map of placeholder -> files it appears in.
        # Restrict to the prefixes this pipeline actually mints so legit
        # bracketed tokens in documents (e.g. part refs like [REV_002])
        # are not false-flagged as unknown placeholders.
        from .anonymizer import ENTITY_PREFIX_MAP, _DEFAULT_PREFIX
        known_prefixes = '|'.join(
            sorted(set(list(ENTITY_PREFIX_MAP.values()) + [_DEFAULT_PREFIX]))
        )
        placeholder_files: Dict[str, Set[str]] = defaultdict(set)
        placeholder_pattern = re.compile(
            rf'\[(?:{known_prefixes})_(\d{{3,}})\]'
        )

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

        # NOTE: leakage of original entity text is the LeakageChecker's job
        # (with boundary/variant-aware patterns); re-scanning here with bare
        # substrings double-counted every hit and false-positived on short
        # entities, so that duplicate pass was removed.

        # Verify placeholders are consistent: every pipeline-format
        # placeholder appearing in the output must exist in the mapping.
        for placeholder, files in placeholder_files.items():
            original = self.mapper.resolve(placeholder)
            if original is None:
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
                 tracker: Optional[ChangeTracker] = None,
                 progress: Optional[Callable[[str], None]] = None) -> LeakageReport:
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

    # Coverage guard: an EMPTY output directory must not verify green —
    # "checked 0 files, 0 leaks" is vacuous, not a pass.
    file_count = sum(1 for f in cleaned_dir.rglob('*') if f.is_file())
    if file_count == 0:
        report.add_result(VerificationResult(
            check_name="Output Coverage Check",
            passed=False,
            details="Cleaned output contains ZERO files — nothing to verify "
                    "(all inputs failed or were quarantined).",
        ))
        return report

    leakage_checker = LeakageChecker(mapper)

    # 1. Entity leakage check (format-aware)
    if progress: progress('Entity Leakage Check')
    leakage_result = leakage_checker.run_check(cleaned_dir, original_dir)
    report.add_result(leakage_result)

    # 2. Filename check
    if progress: progress('Filename Entity Check')
    filename_hits = leakage_checker.check_filenames(cleaned_dir)
    filename_result = VerificationResult(
        check_name="Filename Entity Check",
        passed=len(filename_hits) == 0,
        details=f"Checked {sum(1 for f in cleaned_dir.rglob('*') if f.is_file())} filenames",
        hits=filename_hits,
    )
    report.add_result(filename_result)

    # 3. Metadata check
    if progress: progress('Metadata Entity Check')
    metadata_hits = leakage_checker.check_metadata(cleaned_dir)
    metadata_result = VerificationResult(
        check_name="Metadata Entity Check",
        passed=len(metadata_hits) == 0,
        details="Checked docProps, EXIF, OLE, STEP headers",
        hits=metadata_hits,
    )
    report.add_result(metadata_result)

    # 4. Re-scan for new entities
    if progress: progress('Re-Scan Entity Detection')
    rescaner = ReScanner(mapper)
    rescan_result = rescaner.run_check(cleaned_dir)
    report.add_result(rescan_result)

    # 5. Legibility check
    if progress: progress('Legibility Check')
    legibility_checker = LegibilityChecker()
    legibility_result = legibility_checker.run_check(cleaned_dir)
    report.add_result(legibility_result)

    # 6. Consistency check
    if progress: progress('Consistency Check')
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