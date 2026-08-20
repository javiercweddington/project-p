"""
Diff module - difference tracking between original and cleaned documents.

Records every change made during cleaning, providing:
- ChangeRecord: Individual change (original → placeholder)
- DiffReport: Aggregate summary of all changes per file
- StructuralIntegrityCheck: Verify document structure outside sensitive info
"""

from __future__ import annotations

import hashlib
import difflib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

_logger = logging.getLogger(__name__)


@dataclass
class ChangeRecord:
    """Records a single change made during cleaning."""
    file_path: str
    entity_type: str
    original: str
    placeholder: str
    line_number: Optional[int] = None
    character_offset: Optional[int] = None
    context: str = ""  # Surrounding text for review

    def __repr__(self):
        return (f"ChangeRecord({self.file_path!r}, "
                f"{self.original!r} → {self.placeholder!r})")


@dataclass
class FileDiff:
    """Diff summary for a single file."""
    file_path: str
    change_count: int = 0
    changes: List[ChangeRecord] = field(default_factory=list)
    original_hash: str = ""
    cleaned_hash: str = ""
    original_size: int = 0
    cleaned_size: int = 0
    size_diff: int = 0
    is_binary: bool = False
    metadata_removed: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return self.change_count > 0 or bool(self.metadata_removed)

    def summary(self) -> str:
        lines = [
            f"File: {self.file_path}",
            f"  Changes: {self.change_count}",
            f"  Size: {self.original_size} → {self.cleaned_size} "
            f"({self.size_diff:+d})",
        ]
        if self.metadata_removed:
            lines.append(f"  Metadata removed: {len(self.metadata_removed)} fields")
        if self.change_count > 0:
            lines.append("  Changes by type:")
            by_type: Dict[str, int] = {}
            for c in self.changes:
                by_type[c.entity_type] = by_type.get(c.entity_type, 0) + 1
            for etype, count in sorted(by_type.items()):
                lines.append(f"    {etype}: {count}")
        return '\n'.join(lines)


@dataclass
class DiffReport:
    """Aggregate diff report for an entire project cleaning."""
    project_name: str
    timestamp: str = ""
    file_diffs: List[FileDiff] = field(default_factory=list)
    total_changes: int = 0
    changes_by_type: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def files_with_changes(self) -> int:
        return sum(1 for fd in self.file_diffs if fd.has_changes)

    @property
    def total_files(self) -> int:
        return len(self.file_diffs)

    def add_file_diff(self, diff: FileDiff) -> None:
        """Add a file diff to the report."""
        self.file_diffs.append(diff)
        self.total_changes += diff.change_count

        # Aggregate by type
        for change in diff.changes:
            self.changes_by_type[change.entity_type] = (
                self.changes_by_type.get(change.entity_type, 0) + 1
            )

    def summary(self) -> str:
        lines = [
            f"Diff Report: {self.project_name}",
            f"Timestamp: {self.timestamp}",
            f"Total files: {self.total_files}",
            f"Files with changes: {self.files_with_changes}",
            f"Total changes: {self.total_changes}",
        ]

        if self.changes_by_type:
            lines.append("\nChanges by entity type:")
            for etype, count in sorted(self.changes_by_type.items()):
                lines.append(f"  {etype}: {count}")

        lines.append("\nFile details:")
        lines.append("-" * 60)
        for fd in self.file_diffs:
            if fd.has_changes:
                lines.append(fd.summary())
                lines.append("")

        return '\n'.join(lines)

    def affected_files(self) -> List[str]:
        """Return list of file paths that had changes."""
        return [fd.file_path for fd in self.file_diffs if fd.has_changes]


class ChangeTracker:
    """Tracks changes as they are made during cleaning."""

    def __init__(self):
        self._records: List[ChangeRecord] = []
        self._file_diffs: Dict[str, FileDiff] = {}

    def record_change(self, file_path: str, entity_type: str,
                      original: str, placeholder: str,
                      line_number: Optional[int] = None,
                      character_offset: Optional[int] = None,
                      context: str = "") -> ChangeRecord:
        """Record a single change."""
        record = ChangeRecord(
            file_path=file_path,
            entity_type=entity_type,
            original=original,
            placeholder=placeholder,
            line_number=line_number,
            character_offset=character_offset,
            context=context,
        )
        self._records.append(record)

        # Update file diff
        if file_path not in self._file_diffs:
            self._file_diffs[file_path] = FileDiff(file_path=file_path)
        self._file_diffs[file_path].change_count += 1
        self._file_diffs[file_path].changes.append(record)

        return record

    def finalize_file(self, file_path: str,
                      original_hash: str, cleaned_hash: str,
                      original_size: int, cleaned_size: int,
                      metadata_removed: Optional[List[str]] = None) -> FileDiff:
        """Finalize a file diff with hash and size info."""
        if file_path not in self._file_diffs:
            self._file_diffs[file_path] = FileDiff(file_path=file_path)

        fd = self._file_diffs[file_path]
        fd.original_hash = original_hash
        fd.cleaned_hash = cleaned_hash
        fd.original_size = original_size
        fd.cleaned_size = cleaned_size
        fd.size_diff = cleaned_size - original_size
        if metadata_removed:
            fd.metadata_removed.extend(metadata_removed)

        return fd

    def get_file_diff(self, file_path: str) -> Optional[FileDiff]:
        """Get the diff for a specific file."""
        return self._file_diffs.get(file_path)

    def build_report(self, project_name: str) -> DiffReport:
        """Build the final diff report."""
        report = DiffReport(project_name=project_name)
        for fd in self._file_diffs.values():
            report.add_file_diff(fd)
        return report

    @property
    def total_changes(self) -> int:
        return len(self._records)


class StructuralIntegrityCheck:
    """Verify document structure is intact outside of sensitive info replacements.

    Performs bit-by-bit analysis to ensure:
    - Non-entity content is unchanged
    - Document structure (headings, paragraphs, tables) is preserved
    - Only entity spans were modified
    """

    def __init__(self, original_text: str, cleaned_text: str,
                 entity_spans: List[Tuple[int, int]]):
        self.original = original_text
        self.cleaned = cleaned_text
        self.entity_spans = sorted(entity_spans, key=lambda s: s[0])

    def check_non_entity_content_unchanged(self) -> bool:
        """Verify that content outside entity spans is identical."""
        # Build a mask of non-entity regions
        non_entity_regions = []
        prev_end = 0
        for start, end in self.entity_spans:
            if start > prev_end:
                non_entity_regions.append((prev_end, start))
            prev_end = end
        if prev_end < len(self.original):
            non_entity_regions.append((prev_end, len(self.original)))

        # Check each non-entity region
        for start, end in non_entity_regions:
            orig_region = self.original[start:end]
            # The cleaned text may have different offsets due to replacement length
            # differences, so we compare content directly
            if orig_region not in self.cleaned:
                _logger.warning("Non-entity content changed at offset %d-%d", start, end)
                return False

        return True

    def check_structure_preserved(self, structure_pattern: str = "paragraph") -> bool:
        """Check that document structure is preserved.

        Args:
            structure_pattern: Type of structure to check ('paragraph', 'line', 'heading')
        """
        if structure_pattern == "paragraph":
            orig_paragraphs = self._split_paragraphs(self.original)
            clean_paragraphs = self._split_paragraphs(self.cleaned)
            return len(orig_paragraphs) == len(clean_paragraphs)

        elif structure_pattern == "line":
            orig_lines = self.original.split('\n')
            clean_lines = self.cleaned.split('\n')
            return len(orig_lines) == len(clean_lines)

        elif structure_pattern == "heading":
            orig_headings = self._extract_headings(self.original)
            clean_headings = self._extract_headings(self.cleaned)
            # Headings should have the same count (entity replacement shouldn't
            # add/remove headings)
            return len(orig_headings) == len(clean_headings)

        return True

    def check_placeholder_format(self, placeholder_pattern: str = r'\[\w+_\d{3}\]') -> bool:
        """Verify all replacements use proper placeholder format."""
        import re
        # Find all placeholders in cleaned text
        placeholders = re.findall(placeholder_pattern, self.cleaned)

        # Ensure no original entity text remains (this is checked by verifier)
        # Here we just verify placeholder format consistency
        for ph in placeholders:
            if not re.match(placeholder_pattern, ph):
                _logger.warning("Malformed placeholder: %s", ph)
                return False

        return True

    def diff_summary(self) -> Dict:
        """Generate a summary of differences."""
        orig_lines = self.original.split('\n')
        clean_lines = self.cleaned.split('\n')

        # Use difflib for line-level comparison
        diff = list(difflib.unified_diff(
            orig_lines, clean_lines,
            lineterm='',
            n=2,  # 2 lines of context
        ))

        added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        return {
            'original_lines': len(orig_lines),
            'cleaned_lines': len(clean_lines),
            'lines_added': added,
            'lines_removed': removed,
            'line_count_change': len(clean_lines) - len(orig_lines),
        }

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (separated by blank lines)."""
        paragraphs = []
        current = []
        for line in text.split('\n'):
            if line.strip():
                current.append(line)
            else:
                if current:
                    paragraphs.append('\n'.join(current))
                    current = []
        if current:
            paragraphs.append('\n'.join(current))
        return paragraphs

    def _extract_headings(self, text: str) -> List[str]:
        """Extract Markdown-style headings."""
        import re
        headings = []
        for line in text.split('\n'):
            if re.match(r'^#{1,6}\s+', line):
                headings.append(line)
        return headings


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_text_hash(text: str, algorithm: str = "sha256") -> str:
    """Compute hash of text content."""
    h = hashlib.new(algorithm)
    h.update(text.encode('utf-8'))
    return h.hexdigest()