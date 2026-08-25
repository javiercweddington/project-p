"""
Pipeline module - orchestration for the cleaning workflow.

Coordinates the full cleaning pipeline:
1. Copy project files to /tmp/clean/{project_name}/
2. Build entity mapper from acquisition sensitivity flags
3. Clean each file using the appropriate cleaner
4. Run verification suite
5. Generate diff report
6. Files remain in /tmp/ until compact step consumes them

Usage:
    pipeline = CleanPipeline(project_group)
    result = pipeline.run()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .anonymizer import EntityMapper, SpanBasedReplacer
from .cleaner import FileCleanerRouter, TEXT_EXTS, IMAGE_EXTS
from .diff import ChangeTracker, DiffReport, compute_file_hash, compute_text_hash
from .verifier import verify_clean, LeakageReport, ChangeTracker as VerifierChangeTracker

_logger = logging.getLogger(__name__)

# Default staging directory
DEFAULT_STAGING_DIR = Path("/tmp/clean")

# Fixed timestamp for cleaned output files (2024-01-01 00:00:00 UTC)
# Normalizes filesystem mtimes to prevent temporal leakage
CLEAN_MTIME = 1704067200  # 2024-01-01 00:00:00 UTC as Unix timestamp

# Size-delta threshold: warn if cleaned file shrinks by less than this percentage
# A cleaned PDF that shrank 2% did not lose its incremental updates
MIN_CLEAN_SHRINK_PERCENT = 5.0

# Size-delta threshold: warn if cleaned file grows (should not happen)
MAX_CLEAN_GROWTH_PERCENT = 10.0


class CleanResult:
    """Result of a cleaning pipeline execution."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.staging_dir: Optional[Path] = None
        self.mapper: Optional[EntityMapper] = None
        self.diff_report: Optional[DiffReport] = None
        self.leakage_report: Optional[LeakageReport] = None
        self.files_cleaned: int = 0
        self.files_failed: int = 0
        self.total_entities_replaced: int = 0
        self.success: bool = False
        self.errors: List[str] = []

    def summary(self) -> str:
        lines = [
            f"Clean Result: {self.project_name}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Files cleaned: {self.files_cleaned}",
            f"Files failed: {self.files_failed}",
            f"Entities replaced: {self.total_entities_replaced}",
        ]

        if self.staging_dir:
            lines.append(f"Staging dir: {self.staging_dir}")

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for error in self.errors[:5]:
                lines.append(f"  - {error}")

        return '\n'.join(lines)


class CleanPipeline:
    """Orchestrates the full cleaning pipeline for a project.

    Workflow:
    1. Copy source files to staging directory (/tmp/clean/{project}/)
    2. Build entity mapper from acquisition sensitivity flags
    3. Pre-scan: populate mapper with all entities before cleaning
    4. Clean each file using the appropriate cleaner
    5. Run verification suite
    6. Generate diff report
    7. Save mapper for audit trail
    """

    def __init__(self, project_name: str,
                 source_dir: Path,
                 staging_dir: Optional[Path] = None,
                 mapper: Optional[EntityMapper] = None):
        """Initialize the cleaning pipeline.

        Args:
            project_name: Name of the project being cleaned
            source_dir: Directory containing original project files
            staging_dir: Output directory (defaults to /tmp/clean/{project_name})
            mapper: Pre-built EntityMapper (optional; will be built from flags if not provided)
        """
        self.project_name = project_name
        self.source_dir = Path(source_dir)
        self.staging_dir = Path(staging_dir) if staging_dir else (
            DEFAULT_STAGING_DIR / project_name
        )
        # Create the diff.ChangeTracker that records every substitution
        self.tracker = ChangeTracker()

        # Create a verifier ChangeTracker that the mapper callback feeds into
        self._verifier_tracker = VerifierChangeTracker()

        # Wire up the mapper so every real substitution is recorded.
        # The callback signature is: (original, placeholder, source) -> None
        def _on_replace(original: str, placeholder: str, source: str) -> None:
            # We don't know entity_type at callback time, but we can look it up
            mapping = mapper.get_mapping(original) if mapper else None
            entity_type = mapping.entity_type if mapping else 'unknown'
            self.tracker.record_change(
                file_path=source or "",
                entity_type=entity_type,
                original=original,
                placeholder=placeholder,
            )
            self._verifier_tracker.record_change(original, placeholder, source or "")

        self.mapper = mapper or EntityMapper(tracker_callback=_on_replace)
        self._entity_spans: Dict[str, List[Tuple[int, int, str, str]]] = {}

    def load_entity_spans(self, spans: Dict[str, List[Tuple[int, int, str, str]]]) -> None:
        """Load entity spans from acquisition pass.

        Args:
            spans: Dictionary mapping relative file paths to lists of
                   (start, end, entity_type, source) tuples
        """
        self._entity_spans = spans

    # Only register true identifier values in the mapper.
    # Generic keywords like "invoice", "payment", "nda" from sensitive_doc detection
    # would corrupt content by replacing those common words everywhere.
    # Similarly, generic dates/descriptions from CAD metadata are not identifiers.
    _IDENTIFIER_ENTITY_TYPES = frozenset({
        'person', 'company', 'email', 'phone', 'product',
    })

    def load_sensitivity_flags(self, flags: List) -> None:
        """Pre-populate the entity mapper from acquisition sensitivity flags.

        Only registers true identifier values (person, company, email, phone, product).
        Generic keywords like "invoice", "payment" from sensitive_doc detection are
        skipped to prevent corrupting normal content.

        This ensures consistent placeholders across all files before cleaning begins.

        Args:
            flags: List of SensitivityFlag objects from acquire.catalog
        """
        for flag in flags:
            # Skip non-identifier entity types to prevent mapper poisoning
            if flag.flag_type not in self._IDENTIFIER_ENTITY_TYPES:
                _logger.debug(
                    "Skipping non-identifier flag: %r (%s from %s)",
                    flag.value, flag.flag_type, flag.source,
                )
                continue

            self.mapper.get_or_create(
                entity_type=flag.flag_type,
                value=flag.value,
                source=flag.source,
            )

    def run(self) -> CleanResult:
        """Execute the full cleaning pipeline.

        Returns:
            CleanResult with all metrics and reports
        """
        result = CleanResult(self.project_name)
        result.staging_dir = self.staging_dir
        result.mapper = self.mapper

        _logger.info("Starting clean pipeline for %s", self.project_name)

        # Step 1: Copy source to staging
        try:
            self._copy_to_staging()
        except Exception as e:
            result.errors.append(f"Failed to copy to staging: {e}")
            result.success = False
            return result

        # Step 2: Clean files
        cleaned, failed = self._clean_all_files()
        result.files_cleaned = cleaned
        result.files_failed = failed
        result.total_entities_replaced = self.tracker.total_changes

        if failed > 0 and cleaned == 0:
            result.errors.append(f"All {failed} files failed to clean")
            result.success = False
            return result

        # Step 3: Run verification
        try:
            leakage_report = verify_clean(
                cleaned_dir=self.staging_dir,
                original_dir=self.source_dir,
                mapper=self.mapper,
                project_name=self.project_name,
                tracker=self._verifier_tracker,
            )
            result.leakage_report = leakage_report

            if not leakage_report.all_passed:
                result.errors.append(
                    f"Verification failed: {len(leakage_report.failed_checks)} checks failed"
                )
                _logger.warning(
                    "Verification failed for %s: %d leakages",
                    self.project_name, leakage_report.total_leakages,
                )
        except Exception as e:
            result.errors.append(f"Verification error: {e}")
            _logger.error("Verification failed for %s: %s", self.project_name, e)

        # Step 4: Build diff report
        result.diff_report = self.tracker.build_report(self.project_name)

        # Step 5: Save mapper for audit trail
        self._save_mapper()

        # Determine overall success: any failure means the run is not successful
        result.success = (
            cleaned > 0
            and failed == 0
            and len(result.errors) == 0
            and (result.leakage_report is None or result.leakage_report.all_passed)
        )

        _logger.info(
            "Clean pipeline completed for %s: %d cleaned, %d failed, %d entities replaced",
            self.project_name, cleaned, failed, result.total_entities_replaced,
        )

        return result

    def _copy_to_staging(self) -> None:
        """Copy source files to staging directory."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.source_dir, self.staging_dir, dirs_exist_ok=True)

        _logger.info("Copied %s to %s", self.source_dir, self.staging_dir)

    def _clean_all_files(self) -> Tuple[int, int]:
        """Clean all files in staging directory.

        Returns:
            Tuple of (files_cleaned, files_failed)
        """
        router = FileCleanerRouter(self.mapper)
        cleaned = 0
        failed = 0

        for staging_file in self.staging_dir.rglob('*'):
            if not staging_file.is_file():
                continue

            # Skip hidden/audit files
            if staging_file.name.startswith('.'):
                continue

            rel_path = staging_file.relative_to(self.staging_dir)
            source_file = self.source_dir / rel_path

            # Get entity spans for this file
            entity_spans = self._entity_spans.get(str(rel_path), None)

            # Record original size before cleaning
            orig_size = source_file.stat().st_size

            # Clean in-place (staging file is both input and output)
            success = router.clean_file(
                input_path=source_file,
                output_path=staging_file,
                entity_spans=entity_spans,
            )

            if success:
                # Record hashes and sizes
                orig_hash = compute_file_hash(source_file)
                clean_hash = compute_file_hash(staging_file)
                clean_size = staging_file.stat().st_size

                # Size-delta check: verify cleaning had effect
                self._check_size_delta(rel_path, orig_size, clean_size)

                # Normalize filesystem mtime to prevent temporal leakage
                self._normalize_mtime(staging_file)

                self.tracker.finalize_file(
                    file_path=str(rel_path),
                    original_hash=orig_hash,
                    cleaned_hash=clean_hash,
                    original_size=orig_size,
                    cleaned_size=clean_size,
                )
                cleaned += 1
            else:
                failed += 1
                _logger.warning("Failed to clean %s", rel_path)
                # Quarantine: move the original file outside the deliverable
                self._quarantine_file(staging_file, rel_path)

        # Anonymize filenames and directory components
        self._anonymize_paths()

        return cleaned, failed

    def _anonymize_paths(self) -> None:
        """Rename files and directories in staging to replace sensitive entities.

        Walks the staging directory bottom-up so that child paths are renamed
        before their parents, avoiding broken intermediate paths. Also handles
        ZIP member names by reprocessing any .zip files through the cleaner.

        Addresses the risk that filenames like:
            "JCW20200615 INVOICE - Acme Corp - John Smith.xlsm"
        contain sensitive entities that would otherwise be delivered as-is.
        """
        # Collect all directories and files
        dirs_to_rename: List[Path] = []
        files_to_rename: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(self.staging_dir, topdown=False):
            current_dir = Path(dirpath)

            # Skip staging root itself
            if current_dir == self.staging_dir:
                continue

            # Check if directory name needs anonymization
            anonymized_dir = self.mapper.replace_in_text(current_dir.name)
            if anonymized_dir != current_dir.name:
                dirs_to_rename.append(current_dir)

            # Check files in this directory
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                anonymized_file = self.mapper.replace_in_text(filename)
                if anonymized_file != filename:
                    files_to_rename.append(current_dir / filename)

        # Rename files first
        for file_path in files_to_rename:
            if not file_path.exists():
                continue
            anonymized_name = self.mapper.replace_in_text(file_path.name)
            # Ensure extension is preserved properly
            new_path = file_path.parent / anonymized_name
            self._safe_rename(file_path, new_path, "file")

        # Rename directories bottom-up
        for dir_path in dirs_to_rename:
            if not dir_path.exists():
                continue
            anonymized_name = self.mapper.replace_in_text(dir_path.name)
            new_path = dir_path.parent / anonymized_name
            self._safe_rename(dir_path, new_path, "directory")

    def _safe_rename(self, old_path: Path, new_path: Path, item_type: str) -> None:
        """Safely rename a file or directory, handling conflicts.

        Args:
            old_path: Current path
            new_path: Desired new path
            item_type: "file" or "directory" for logging
        """
        if old_path == new_path:
            return

        # Handle name conflicts by adding a suffix
        if new_path.exists():
            stem = new_path.stem
            suffix = new_path.suffix
            counter = 1
            while new_path.exists():
                new_path = new_path.parent / f"{stem}_{counter:03d}{suffix}"
                counter += 1

        try:
            old_path.rename(new_path)
            _logger.info(
                "Anonymized %s: %s -> %s",
                item_type, old_path.name, new_path.name,
            )
        except OSError as e:
            _logger.warning(
                "Failed to rename %s %s: %s",
                item_type, old_path.name, e,
            )

    def _check_size_delta(self, rel_path: Path, orig_size: int, clean_size: int) -> None:
        """Check size delta between original and cleaned file.

        A cleaned PDF that shrank 2% did not lose its incremental updates.
        A cleaned file that grew is suspicious (should not happen).

        Args:
            rel_path: Relative path for logging
            orig_size: Original file size in bytes
            clean_size: Cleaned file size in bytes
        """
        if orig_size == 0:
            return

        delta = clean_size - orig_size
        delta_percent = (delta / orig_size) * 100

        # Check for growth (should not happen after cleaning)
        if delta_percent > MAX_CLEAN_GROWTH_PERCENT:
            _logger.warning(
                "Size-delta WARNING: %s grew %.1f%% (%d -> %d bytes). "
                "Cleaning may have failed or added content.",
                rel_path, delta_percent, orig_size, clean_size,
            )
        elif delta_percent > 0:
            _logger.info(
                "Size-delta INFO: %s grew %.1f%% (%d -> %d bytes).",
                rel_path, delta_percent, orig_size, clean_size,
            )

        # Check for insufficient shrinkage (ghost content may remain)
        if 0 > delta_percent > -MIN_CLEAN_SHRINK_PERCENT:
            _logger.warning(
                "Size-delta WARNING: %s shrank only %.1f%% (%d -> %d bytes). "
                "Ghost content (incremental updates, pivot cache, etc.) may remain.",
                rel_path, abs(delta_percent), orig_size, clean_size,
            )
        elif delta_percent <= -MIN_CLEAN_SHRINK_PERCENT:
            _logger.info(
                "Size-delta OK: %s shrank %.1f%% (%d -> %d bytes).",
                rel_path, abs(delta_percent), orig_size, clean_size,
            )

    def _quarantine_file(self, staging_file: Path, rel_path: Path) -> None:
        """Move a failed file to a quarantine directory outside the deliverable.

        Failed files are moved to {_staging_dir_parent}/_quarantine/{project_name}/
        so they are not included in the cleaned output.

        Args:
            staging_file: Path to the file in staging that failed to clean
            rel_path: Relative path within the staging directory
        """
        quarantine_dir = self.staging_dir.parent / '_quarantine' / self.project_name
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        quarantine_path = quarantine_dir / rel_path
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            staging_file.rename(quarantine_path)
            _logger.info(
                "Quarantined %s -> %s", rel_path, quarantine_path,
            )
        except OSError as e:
            _logger.warning(
                "Failed to quarantine %s: %s", rel_path, e,
            )

    def _normalize_mtime(self, file_path: Path) -> None:
        """Normalize filesystem mtime to prevent temporal leakage.

        Filesystem modification times map your work schedule. By setting
        all cleaned files to a fixed timestamp, we remove this correlation
        vector.

        Args:
            file_path: Path to the file whose mtime should be normalized
        """
        try:
            os.utime(file_path, (CLEAN_MTIME, CLEAN_MTIME))
        except OSError as e:
            _logger.debug(
                "Failed to normalize mtime for %s: %s",
                file_path, e,
            )

    def _save_mapper(self) -> None:
        """Save the entity mapper to an audit location outside the staging dir.

        The mapper contains the reverse mapping with every original sensitive
        value in plaintext. Writing it into the deliverable folder would leak
        those values, so it is placed in a separate audit directory at the
        same level as the staging parent.
        """
        audit_dir = self.staging_dir.parent / '_audit' / self.project_name
        audit_dir.mkdir(parents=True, exist_ok=True)
        mapper_path = audit_dir / ".entity_mapper.json"
        try:
            with open(mapper_path, 'w') as f:
                json.dump(self.mapper.to_dict(), f, indent=2)
            _logger.info("Saved entity mapper to %s (audit location)", mapper_path)
        except Exception as e:
            _logger.warning("Failed to save entity mapper: %s", e)

    @classmethod
    def from_project_group(cls, project_group,
                           staging_dir: Optional[Path] = None) -> 'CleanPipeline':
        """Create a CleanPipeline from a ProjectGroup.

        Args:
            project_group: ProjectGroup from acquire.read
            staging_dir: Optional staging directory override

        Returns:
            Configured CleanPipeline ready to run
        """
        from acquire.catalog import ProjectCatalog

        project_name = project_group.project_name
        source_file = project_group.latest_file

        if not source_file:
            raise ValueError(f"No files in project group {project_name}")

        # Determine source directory
        if source_file.is_zipped:
            # Extract to temp directory for processing
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix=f"project_p_{project_name}_"))
            with zipfile.ZipFile(source_file.filepath, 'r') as zf:
                zf.extractall(temp_dir)
            source_dir = temp_dir
        else:
            source_dir = source_file.filepath

        # Build pipeline
        pipeline = cls(
            project_name=project_name,
            source_dir=source_dir,
            staging_dir=staging_dir,
        )

        # Load sensitivity flags from catalog
        catalog = ProjectCatalog(project_group)
        flags = catalog.all_sensitivity_flags
        pipeline.load_sensitivity_flags(flags)

        return pipeline


def clean_project(project_group,
                  staging_dir: Optional[Path] = None) -> CleanResult:
    """Convenience function: clean a ProjectGroup in one call.

    Args:
        project_group: ProjectGroup from acquire.read
        staging_dir: Optional staging directory override

    Returns:
        CleanResult with all metrics and reports
    """
    pipeline = CleanPipeline.from_project_group(project_group, staging_dir)
    return pipeline.run()


def clean_directory(source_dir: Path, project_name: str,
                    staging_dir: Optional[Path] = None,
                    mapper: Optional[EntityMapper] = None) -> CleanResult:
    """Clean a directory without requiring a ProjectGroup.

    Args:
        source_dir: Directory with files to clean
        project_name: Name for the project (used in reports)
        staging_dir: Optional staging directory override
        mapper: Pre-built EntityMapper with known entities

    Returns:
        CleanResult with all metrics and reports
    """
    pipeline = CleanPipeline(
        project_name=project_name,
        source_dir=Path(source_dir),
        staging_dir=staging_dir,
        mapper=mapper,
    )
    return pipeline.run()