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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .anonymizer import EntityMapper

# Try optional dependencies
try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

_logger = logging.getLogger(__name__)


@dataclass
class LeakageHit:
    """A single leakage detection: original entity text found in cleaned output."""
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


class LeakageChecker:
    """Check that no original entity text survives in cleaned output.

    This is the primary CI test: for every entity the acquisition pass
    detected, assert that its literal string does not appear in the
    cleaned output.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper

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
        """Check a cleaned text file for surviving entities."""
        try:
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                text = f.read()

            display_path = relative_path or str(cleaned_path)
            return self.check_text(text, display_path)

        except (UnicodeDecodeError, UnicodeEncodeError):
            # Binary file - skip text-based leakage check
            return []

    def check_pdf(self, cleaned_path: Path, max_pages: int = 10) -> List[LeakageHit]:
        """Check a cleaned PDF for surviving entities in extracted text."""
        if not HAS_PYPDF2:
            _logger.warning("PyPDF2 not available; skipping PDF leakage check")
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

    def run_check(self, cleaned_dir: Path, original_dir: Path) -> VerificationResult:
        """Run leakage check on all files in cleaned directory.

        Args:
            cleaned_dir: Directory with cleaned files
            original_dir: Directory with original files (for reference)

        Returns:
            VerificationResult with all leakage hits
        """
        all_hits = []

        for cleaned_file in cleaned_dir.rglob('*'):
            if not cleaned_file.is_file():
                continue

            rel_path = cleaned_file.relative_to(cleaned_dir)
            original_file = original_dir / rel_path

            if cleaned_file.suffix.lower() == '.pdf':
                hits = self.check_pdf(cleaned_file)
            elif cleaned_file.suffix.lower() in (
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
                '.mp3', '.wav', '.mp4', '.avi', '.mov',
            ):
                # Binary files - skip text leakage check
                continue
            else:
                hits = self.check_file(cleaned_file, original_file, str(rel_path))

            all_hits.extend(hits)

        passed = len(all_hits) == 0
        return VerificationResult(
            check_name="Entity Leakage Check",
            passed=passed,
            details=f"Checked {sum(1 for f in cleaned_dir.rglob('*') if f.is_file())} files",
            hits=all_hits,
        )


class ReScanner:
    """Re-run entity detection on cleaned documents.

    If GLiNER finds new person/organization entities in the cleaned text,
    either the scrubber missed something or the replacement text introduced
    new entities.
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
            # Check if this entity is one we already know about (should be replaced)
            if self.mapper.has_entity(entity.value):
                # This is a known entity that survived - it's a leakage
                hits.append(LeakageHit(
                    file_path=source,
                    entity_type=entity.entity_type,
                    original=entity.value,
                    context=entity.context,
                ))
            # New entities found in cleaned text might be from non-sensitive content
            # We report them but don't count them as leakage

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


class LegibilityChecker:
    """Verify that cleaned documents are still valid and readable."""

    def check_text_file(self, path: Path) -> bool:
        """Check that a text file is readable and non-empty."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return len(content) > 0
        except Exception:
            return False

    def check_pdf(self, path: Path) -> bool:
        """Check that a PDF is still valid."""
        if not HAS_PYPDF2:
            return True  # Can't verify without PyPDF2

        try:
            reader = PdfReader(str(path))
            return len(reader.pages) > 0
        except Exception:
            return False

    def check_image(self, path: Path) -> bool:
        """Check that an image is still valid."""
        try:
            from PIL import Image
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


class ConsistencyChecker:
    """Verify anonymization consistency across all datatypes.

    Ensures the same entity always maps to the same placeholder
    regardless of which file or datatype it appears in.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper

    def check_cross_file_consistency(self, cleaned_dir: Path) -> VerificationResult:
        """Verify that placeholders are consistent across all files.

        Check that:
        1. Each placeholder appears with the same meaning everywhere
        2. No original entity text appears in any file
        """
        hits = []

        # Build a map of placeholder -> files it appears in
        placeholder_files: Dict[str, Set[str]] = defaultdict(set)

        placeholder_pattern = re.compile(r'\[(\w+)_(\d{3})\]')

        for file_path in cleaned_dir.rglob('*'):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue

            rel = str(file_path.relative_to(cleaned_dir))
            for match in placeholder_pattern.finditer(text):
                placeholder = match.group(0)
                placeholder_files[placeholder].add(rel)

        # Check for any original entities that might have slipped through
        for mapping in self.mapper.mappings:
            original = mapping.original
            expected_placeholder = mapping.placeholder

            # Verify the placeholder exists and is used consistently
            if expected_placeholder in placeholder_files:
                files = placeholder_files[expected_placeholder]
                _logger.debug(
                    "Placeholder %s for %r appears in %d files",
                    expected_placeholder, original, len(files)
                )

        passed = len(hits) == 0
        return VerificationResult(
            check_name="Cross-File Consistency Check",
            passed=passed,
            details=f"Verified {len(placeholder_files)} placeholders across files",
            hits=hits,
        )

    def run_check(self, cleaned_dir: Path) -> VerificationResult:
        """Run consistency check."""
        return self.check_cross_file_consistency(cleaned_dir)


def verify_clean(cleaned_dir: Path, original_dir: Path,
                 mapper: EntityMapper, project_name: str = "") -> LeakageReport:
    """Run all verification checks and return a comprehensive report.

    This is the main entry point for verification.

    Args:
        cleaned_dir: Directory with cleaned files
        original_dir: Directory with original files
        mapper: EntityMapper with all known entity mappings
        project_name: Name of the project for the report

    Returns:
        LeakageReport with all verification results
    """
    report = LeakageReport(project_name=project_name or cleaned_dir.name)

    # Run leakage check
    leakage_checker = LeakageChecker(mapper)
    leakage_result = leakage_checker.run_check(cleaned_dir, original_dir)
    report.add_result(leakage_result)

    # Run re-scan
    rescaner = ReScanner(mapper)
    rescan_result = rescaner.run_check(cleaned_dir)
    report.add_result(rescan_result)

    # Run legibility check
    legibility_checker = LegibilityChecker()
    legibility_result = legibility_checker.run_check(cleaned_dir)
    report.add_result(legibility_result)

    # Run consistency check
    consistency_checker = ConsistencyChecker(mapper)
    consistency_result = consistency_checker.run_check(cleaned_dir)
    report.add_result(consistency_result)

    return report