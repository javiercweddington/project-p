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
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .anonymizer import EntityMapper, SpanBasedReplacer
from .cleaner import FileCleanerRouter, TEXT_EXTS, IMAGE_EXTS, AUDIO_EXTS, VIDEO_EXTS
from .diff import ChangeTracker, DiffReport, compute_file_hash, compute_text_hash
from .verifier import verify_clean, LeakageReport

_logger = logging.getLogger(__name__)

# Default staging directory
DEFAULT_STAGING_DIR = Path("/tmp/clean")


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
        self.mapper = mapper or EntityMapper()
        self.tracker = ChangeTracker()
        self._entity_spans: Dict[str, List[Tuple[int, int, str, str]]] = {}

    def load_entity_spans(self, spans: Dict[str, List[Tuple[int, int, str, str]]]) -> None:
        """Load entity spans from acquisition pass.

        Args:
            spans: Dictionary mapping relative file paths to lists of
                   (start, end, entity_type, source) tuples
        """
        self._entity_spans = spans

    def load_sensitivity_flags(self, flags: List) -> None:
        """Pre-populate the entity mapper from acquisition sensitivity flags.

        This ensures consistent placeholders across all files before cleaning begins.

        Args:
            flags: List of SensitivityFlag objects from acquire.catalog
        """
        for flag in flags:
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

        # Determine overall success
        result.success = (
            cleaned > 0
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

            rel_path = staging_file.relative_to(self.staging_dir)
            source_file = self.source_dir / rel_path

            # Get entity spans for this file
            entity_spans = self._entity_spans.get(str(rel_path), None)

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
                orig_size = source_file.stat().st_size
                clean_size = staging_file.stat().st_size

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

        return cleaned, failed

    def _save_mapper(self) -> None:
        """Save the entity mapper for audit trail."""
        mapper_path = self.staging_dir / ".entity_mapper.json"
        try:
            with open(mapper_path, 'w') as f:
                json.dump(self.mapper.to_dict(), f, indent=2)
            _logger.info("Saved entity mapper to %s", mapper_path)
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