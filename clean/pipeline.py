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
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .anonymizer import EntityMapper, SpanBasedReplacer, PLACEHOLDER_TOKEN_RE
from .cleaner import FileCleanerRouter, TEXT_EXTS, IMAGE_EXTS
from .diff import ChangeTracker, DiffReport, compute_file_hash, compute_text_hash
from .verifier import verify_clean, LeakageReport, ChangeTracker as VerifierChangeTracker

_logger = logging.getLogger(__name__)


class _Progress:
    """Minimal, dependency-free progress reporting.

    On a TTY: a live single-line bar  [#####.....] 12/34 cleaning: file.pdf
    Otherwise (piped to a log/CI): one plain line per stage, one per N items.
    Control: PROJECT_P_PROGRESS=1 forces on, =0 forces off; default is
    "on when stderr is a TTY, plain-line mode otherwise".
    """

    def __init__(self):
        import sys
        env = os.environ.get('PROJECT_P_PROGRESS', '').strip()
        self._tty = sys.stderr.isatty()
        self.enabled = env != '0'
        self._live = self._tty and env != '0'
        self._stream = sys.stderr
        self._last_len = 0

    def stage(self, label: str) -> None:
        if not self.enabled:
            return
        self._clear_line()
        self._stream.write(f'--- {label}\n')
        self._stream.flush()

    def step(self, index: int, total: int, item: str = '') -> None:
        if not self.enabled or total <= 0:
            return
        if self._live:
            width = 24
            filled = int(width * index / total)
            bar = '#' * filled + '.' * (width - filled)
            line = f'[{bar}] {index}/{total} {item[:48]}'
            pad = max(0, self._last_len - len(line))
            self._stream.write('\r' + line + ' ' * pad)
            self._last_len = len(line)
            self._stream.flush()
        elif index == total or index % 5 == 0:
            self._stream.write(f'    {index}/{total} {item[:60]}\n')
            self._stream.flush()

    def finish(self) -> None:
        self._clear_line()

    def _clear_line(self) -> None:
        if self._live and self._last_len:
            self._stream.write('\r' + ' ' * self._last_len + '\r')
            self._stream.flush()
            self._last_len = 0


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
                 mapper: Optional[EntityMapper] = None,
                 one_pass: Optional[bool] = None):
        """Initialize the cleaning pipeline.

        Args:
            project_name: Name of the project being cleaned
            source_dir: Directory containing original project files
            staging_dir: Output directory (defaults to /tmp/clean/{project_name})
            mapper: Pre-built EntityMapper (optional; will be built from flags if not provided)
            one_pass: True = up-front discovery, then a SINGLE clean pass
                and a single verification (no retroactive passes, no
                iterative LLM loop; anything missed fails verification
                instead of triggering re-cleans). None = read
                PROJECT_P_ONE_PASS env (default off, preserving the
                iterative behavior for library callers).
        """
        if one_pass is None:
            one_pass = os.environ.get('PROJECT_P_ONE_PASS', '0') == '1'
        self.one_pass = one_pass
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
            # Late-bind through self.mapper: the constructor-arg `mapper` is
            # None when the pipeline builds its own mapper, which made every
            # recorded change type 'unknown'.
            mapping = self.mapper.get_mapping(original) if self.mapper else None
            entity_type = mapping.entity_type if mapping else 'unknown'
            self.tracker.record_change(
                file_path=source or "",
                entity_type=entity_type,
                original=original,
                placeholder=placeholder,
            )
            self._verifier_tracker.record_change(original, placeholder, source or "")

        self.mapper = mapper
        if self.mapper is None:
            self.mapper = EntityMapper(tracker_callback=_on_replace)
        else:
            # Wire up the tracker callback even for externally-provided mappers
            # so that replacements are counted in the diff report
            existing_callback = self.mapper._tracker_callback
            if existing_callback is not None:
                # Chain callbacks: call both the existing one and our tracker
                def _chained_callback(original: str, placeholder: str, source: str) -> None:
                    existing_callback(original, placeholder, source)
                    _on_replace(original, placeholder, source)
                self.mapper._tracker_callback = _chained_callback
            else:
                self.mapper._tracker_callback = _on_replace
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
        self._progress = _Progress()

        # Step 1: Copy source to staging
        try:
            self._progress.stage(f'copying {self.source_dir.name} to staging')
            self._copy_to_staging()
        except Exception as e:
            result.errors.append(f"Failed to copy to staging: {e}")
            result.success = False
            return result

        # Step 1b (one-pass mode): discover ALL entities up front — LLM
        # scan plus deterministic identifier registration on the staged
        # (still-original) content — so a single clean pass suffices and
        # no retroactive re-cleans are needed.
        if self.one_pass:
            self._upfront_discovery(result)

        # Step 2: Clean files
        cleaned, failed = self._clean_all_files(
            max_extra_passes=0 if self.one_pass else 2)
        result.files_cleaned = cleaned
        result.files_failed = failed
        result.total_entities_replaced = self.tracker.total_changes

        if failed > 0 and cleaned == 0:
            # Do NOT return early: verification, the diff report, and the
            # mapper audit trail must still be produced for failed runs.
            result.errors.append(f"All {failed} files failed to clean")

        if getattr(self, '_images_without_ocr', 0):
            result.errors.append(
                f"{self._images_without_ocr} image(s) processed WITHOUT OCR "
                f"screening — sensitive pixel text cannot be ruled out. "
                f"Install tesseract (and pytesseract) to clear this."
            )

        # Step 2b (iterative mode only): LLM-backed discovery of entities
        # the deterministic detectors can't express (prose names,
        # companies, addresses, Chinese text). BACKSTOP ONLY:
        # deterministic replacement has already run; this loop registers
        # what the local model still sees and re-cleans until it finds
        # nothing new. In one-pass mode discovery already ran up front
        # and the LLM Cleanliness Check remains the (single) gate.
        if not self.one_pass:
            self._llm_discovery_loop(result)

        # Step 2c: Anonymize filenames and directory components
        # Must run BEFORE verification so the verifier sees anonymized paths
        self._progress.stage('anonymizing file and directory names')
        self._anonymize_paths()

        # Step 2d: Final mtime sweep — renames and directories would
        # otherwise keep original modification times (temporal leak).
        self._normalize_all_mtimes()

        # Step 2e (sample mode): LLM audits one file per type, fixes what
        # it finds everywhere, re-checks. Result is appended to the
        # leakage report after verification runs.
        self._sample_audit_result = None
        self._llm_sample_audit(result)

        # Step 3: Run verification
        try:
            leakage_report = verify_clean(
                cleaned_dir=self.staging_dir,
                original_dir=self.source_dir,
                mapper=self.mapper,
                project_name=self.project_name,
                tracker=self._verifier_tracker,
                progress=lambda label: self._progress.stage(
                    f'verifying: {label}'),
            )

            # Final LLM gate: anything the local model can still identify
            # in the cleaned output is a failing hit (mode-dependent; see
            # PROJECT_P_LLM_VERIFY in clean/llm_detect.py).
            try:
                from .llm_detect import LLMCleanlinessJudge
                self._progress.stage('verifying: LLM Cleanliness Check')
                leakage_report.add_result(
                    LLMCleanlinessJudge(self.mapper).run_check(
                        self.staging_dir)
                )
            except Exception as e:
                _logger.warning("LLM cleanliness check errored: %s", e)

            if getattr(self, '_sample_audit_result', None) is not None:
                leakage_report.add_result(self._sample_audit_result)

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
        """Copy source files to staging directory.

        Hidden files (.env, .DS_Store, ._AppleDouble, .git, ...) are NEVER
        copied: the cleaning loop skips dotfiles, so copying them meant they
        shipped verbatim — including under a SUCCESS result.
        """
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

        # Clear stale quarantine/audit state from prior runs of this project
        for stale in (self._quarantine_root() / self.project_name,
                      self._audit_root() / self.project_name):
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.source_dir, self.staging_dir, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('.*', '__MACOSX'),
        )

        _logger.info("Copied %s to %s (hidden files excluded)",
                     self.source_dir, self.staging_dir)

    def _quarantine_root(self) -> Path:
        """Quarantine root OUTSIDE the deliverable tree.

        The staging parent (default /tmp/clean) is exactly what the compact
        step consumes, so quarantined raw originals must not live inside it.
        """
        parent = self.staging_dir.parent
        return parent / (parent.name + '_quarantine')

    def _audit_root(self) -> Path:
        """Audit root OUTSIDE the deliverable tree (see _quarantine_root)."""
        parent = self.staging_dir.parent
        return parent / (parent.name + '_audit')

    def _clean_all_files(self, max_extra_passes: int = 2) -> Tuple[int, int]:
        """Clean all files in staging directory.

        Args:
            max_extra_passes: retroactive re-clean passes allowed when the
                mapper grew mid-run (0 in one-pass mode: discovery already
                ran up front, and verification catches any straggler).

        Returns:
            Tuple of (files_cleaned, files_failed)
        """
        router = FileCleanerRouter(self.mapper)
        cleaned = 0
        failed = 0
        images_cleaned = 0
        cleaned_files: List[Tuple[Path, Path]] = []  # (staging_file, rel_path)
        mapping_count_before = self.mapper.mapping_count

        todo = [f for f in self.staging_dir.rglob('*')
                if f.is_file() and not f.name.startswith('.')]
        progress = getattr(self, '_progress', None) or _Progress()
        progress.stage(f'cleaning {len(todo)} files (pass 1)')

        for file_index, staging_file in enumerate(todo, 1):
            progress.step(file_index, len(todo), staging_file.name)

            if staging_file.suffix.lower() in IMAGE_EXTS:
                images_cleaned += 1

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
                cleaned_files.append((staging_file, rel_path))
            else:
                failed += 1
                _logger.warning("Failed to clean %s", rel_path)
                # Quarantine: move the original file outside the deliverable
                self._quarantine_file(staging_file, rel_path)

        # RETROACTIVE PASSES: entities discovered mid-run (emails inside a
        # workbook, authors in document metadata) were unknown when earlier
        # files were cleaned. Re-clean the already-cleaned outputs in place
        # until the mapper stops growing, so every file reflects the full
        # entity set. (Accuracy over speed, per project policy; disabled
        # in one-pass mode via max_extra_passes=0.)
        prev_mapping_count = mapping_count_before
        extra_pass = 0
        while (self.mapper.mapping_count != prev_mapping_count
               and extra_pass < max_extra_passes and cleaned_files):
            prev_mapping_count = self.mapper.mapping_count
            extra_pass += 1
            _logger.info(
                "Retroactive cleaning pass %d over %d files "
                "(mapper has %d entities)",
                extra_pass, len(cleaned_files), self.mapper.mapping_count,
            )
            progress.stage(
                f'retroactive re-clean pass {extra_pass} '
                f'({len(cleaned_files)} files, '
                f'{self.mapper.mapping_count} entities)')
            still_clean: List[Tuple[Path, Path]] = []
            for retro_index, (staging_file, rel_path) in enumerate(
                    cleaned_files, 1):
                progress.step(retro_index, len(cleaned_files),
                              staging_file.name)
                if not staging_file.exists():
                    continue
                success = router.clean_file(
                    input_path=staging_file,
                    output_path=staging_file,
                    entity_spans=None,
                )
                if success:
                    self._normalize_mtime(staging_file)
                    still_clean.append((staging_file, rel_path))
                else:
                    cleaned -= 1
                    failed += 1
                    _logger.warning(
                        "Re-clean pass failed for %s; quarantining", rel_path,
                    )
                    self._quarantine_file(staging_file, rel_path)
            cleaned_files = still_clean

        # Surface OCR blindness: images cleaned WITHOUT OCR screening may
        # ship sensitive pixel text (screenshots of receipts, invoices...).
        # This must fail the run, not scroll by as a log line.
        # NOTE: image_ocr is an instance even when tesseract is missing,
        # so check the functional `available` flag rather than `is None`.
        ocr = router.image_cleaner.image_ocr
        ocr_functional = bool(ocr) and getattr(ocr, 'available', False)
        if images_cleaned and not ocr_functional:
            self._images_without_ocr = images_cleaned
        else:
            self._images_without_ocr = 0

        return cleaned, failed

    # Placeholder tokens and separators that may remain in a fully
    # anonymized name; anything else is identifying residue.
    # MUST be the prefix-anchored pattern — a loose [A-Z]+_\d+ regex let
    # real stems like IMG_20200615 pass as "placeholders" and ship verbatim.
    _PLACEHOLDER_TOKEN_RE = PLACEHOLDER_TOKEN_RE
    _NAME_SEPARATOR_RE = re.compile(r'[\s\-_.,;()+&#@!~\[\]{}]+')

    def _anonymize_name(self, name: str, entity_type: str) -> str:
        """Fully anonymize one file or directory name (STRICT policy).

        1. Replace mapped entities with placeholders.
        2. If the remaining stem still carries ANY residue beyond
           placeholders and separators — dates ("20200615"), personal
           acronyms ("JCW"), app/vendor names ("WeChat"), unmapped CJK,
           version markers — the whole stem is replaced with a generic
           pseudonym (FILE_001 / DIR_001) registered in the audit mapper,
           because filename fragments are identifying even when no mapped
           entity matches.

        The extension is always preserved.
        """
        from .anonymizer import anonymize_path_component
        return anonymize_path_component(self.mapper, name, entity_type)

    def _anonymize_paths(self) -> None:
        """Rename files and directories in staging so NO identifying
        information remains in any path component.

        Walks the staging directory bottom-up so that child paths are renamed
        before their parents, avoiding broken intermediate paths.

        STRICT policy: names are first entity-replaced; any name still
        carrying non-placeholder residue (dates, initials like "JCW",
        app names like "WeChat", unmapped CJK text) is replaced wholesale
        with FILE_nnn/DIR_nnn pseudonyms. The original names are preserved
        in the audit mapper for traceability.
        """
        # Collect all directories and files
        dirs_to_rename: List[Path] = []
        files_to_rename: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(self.staging_dir, topdown=False):
            current_dir = Path(dirpath)

            # Check files in this directory (ALWAYS, including staging root)
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                files_to_rename.append(current_dir / filename)

            # Skip staging root itself for directory renaming
            if current_dir == self.staging_dir:
                continue
            dirs_to_rename.append(current_dir)

        # Rename files first
        for file_path in files_to_rename:
            if not file_path.exists():
                continue
            anonymized_name = self._anonymize_name(file_path.name, 'filename')
            if anonymized_name != file_path.name:
                new_path = file_path.parent / anonymized_name
                self._safe_rename(file_path, new_path, "file")

        # Rename directories bottom-up
        for dir_path in dirs_to_rename:
            if not dir_path.exists():
                continue
            anonymized_name = self._anonymize_name(dir_path.name, 'directory')
            if anonymized_name != dir_path.name:
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
        if not staging_file.exists():
            # The cleaner already removed its output (fail-closed) —
            # nothing left in the deliverable, so nothing to move.
            _logger.debug(
                "No staged file to quarantine for %s (already removed "
                "fail-closed by the cleaner).", rel_path,
            )
            return

        quarantine_dir = self._quarantine_root() / self.project_name
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        quarantine_path = quarantine_dir / rel_path
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            staging_file.rename(quarantine_path)
            _logger.info(
                "Quarantined %s -> %s", rel_path, quarantine_path,
            )
        except OSError as e:
            # FAIL CLOSED: if we cannot move the raw original out of the
            # deliverable, delete it — a warning alone left it shipping.
            _logger.warning(
                "Failed to quarantine %s (%s); deleting from staging "
                "instead (fail-closed).", rel_path, e,
            )
            try:
                staging_file.unlink()
            except OSError as unlink_err:
                _logger.error(
                    "Could not delete %s from staging either: %s — "
                    "MANUAL REMOVAL REQUIRED.", rel_path, unlink_err,
                )

    def _upfront_discovery(self, result: 'CleanResult') -> None:
        """One-pass mode: populate the mapper COMPLETELY before cleaning.

        Runs on the staged copies while they still hold original content:
        1. Deterministic identifier registration (emails) from every
           file's extracted text — replaces the retroactive passes'
           mid-run discoveries.
        2. A single LLM discovery scan (mode-aware, see
           PROJECT_P_LLM_VERIFY). The LLM Cleanliness Check in
           verification remains the final gate; anything missed here
           fails the run instead of triggering re-cleans.
        """
        from .llm_detect import (LLMEntityDetector, LocalLLM,
                                 extract_scannable_text, llm_verify_mode)
        from .cleaners.text import TextCleaner

        progress = getattr(self, '_progress', None)
        if progress:
            progress.stage('up-front discovery: extracting text')

        # Deterministic pre-registration from extracted text (also warms
        # the extraction cache for the LLM scan below).
        text_cleaner = TextCleaner(self.mapper)
        todo = [f for f in sorted(self.staging_dir.rglob('*'))
                if f.is_file() and not f.name.startswith('.')]
        texts: Dict[Path, str] = {}
        for index, file_path in enumerate(todo, 1):
            if progress:
                progress.step(index, len(todo), file_path.name)
            text = extract_scannable_text(file_path)
            if text and text.strip():
                texts[file_path] = text
                try:
                    text_cleaner._register_emails(
                        text, source=str(file_path))
                except Exception as e:
                    _logger.debug(
                        "Email pre-registration failed for %s: %s",
                        file_path.name, e)

        self._gliner_discovery(texts, progress)

        mode = llm_verify_mode()
        if mode in ('off', 'judge', 'sample'):
            # 'judge'/'sample' = LLM audits the finished output only;
            # discovery stays deterministic + CV.
            return
        llm = LocalLLM()
        if not llm.available():
            if mode == 'required':
                result.errors.append(
                    f"LLM endpoint {llm.base_url} unreachable and "
                    f"PROJECT_P_LLM_VERIFY=required — up-front entity "
                    f"discovery could NOT run."
                )
            else:
                _logger.info(
                    "LLM endpoint %s unreachable; up-front discovery is "
                    "deterministic-only.", llm.base_url)
            return

        if progress:
            progress.stage(
                f'up-front LLM discovery ({llm.model} @ {llm.base_url})')
        try:
            new_count = LLMEntityDetector(self.mapper, llm).scan_directory(
                self.staging_dir)
            _logger.info(
                "Up-front LLM discovery: %d new entities "
                "(mapper now %d)", new_count, self.mapper.mapping_count)
        except Exception as e:
            result.errors.append(f"Up-front LLM discovery failed: {e}")

    def _llm_discovery_loop(self, result: 'CleanResult',
                            max_rounds: int = 3) -> None:
        """Backstop entity discovery with the local LLM (Qwen at
        PROJECT_P_LLM_BASE), used ONLY for what deterministic detection
        cannot express. Each round: scan cleaned text, register findings,
        re-clean in place; stop when the model finds nothing new.

        Mode (PROJECT_P_LLM_VERIFY): off = skip; auto = use when the
        endpoint answers; required = endpoint unreachable appends an error
        (run fails). The LLM Cleanliness Check in verification is the
        final gate either way.
        """
        from .llm_detect import (LLMEntityDetector, LocalLLM,
                                 llm_verify_mode)

        mode = llm_verify_mode()
        if mode in ('off', 'judge', 'sample'):
            return

        llm = LocalLLM()
        if not llm.available():
            if mode == 'required':
                result.errors.append(
                    f"LLM endpoint {llm.base_url} unreachable and "
                    f"PROJECT_P_LLM_VERIFY=required — entities beyond "
                    f"deterministic detection were NOT discovered."
                )
            else:
                _logger.info(
                    "LLM endpoint %s unreachable; skipping LLM discovery "
                    "(deterministic detection only).", llm.base_url,
                )
            return

        detector = LLMEntityDetector(self.mapper, llm)
        progress = getattr(self, '_progress', None)
        for round_num in range(1, max_rounds + 1):
            if progress:
                progress.stage(
                    f'LLM discovery round {round_num} '
                    f'({llm.model} @ {llm.base_url})')
            try:
                new_count = detector.scan_directory(self.staging_dir)
            except Exception as e:
                result.errors.append(
                    f"LLM discovery failed mid-scan (round {round_num}): {e}"
                )
                return
            _logger.info(
                "LLM discovery round %d: %d new entities", round_num, new_count,
            )
            if new_count == 0:
                break
            if progress:
                progress.stage(
                    f're-cleaning after LLM round {round_num} '
                    f'({self.mapper.mapping_count} entities)')
            failures = self._inplace_reclean()
            if failures:
                _logger.warning(
                    "%d file(s) failed the post-LLM re-clean and were "
                    "quarantined", failures,
                )

    # NER discoveries these types may auto-register; GLiNER's other labels
    # (invoice/date/money) are far too generic to enter the mapper.
    _GLINER_REGISTER_TYPES = frozenset(
        {'person', 'company', 'email', 'phone', 'address'})

    def _gliner_discovery(self, texts: Dict[Path, str], progress) -> None:
        """Promote GLiNER to a first-class up-front discovery detector.

        Person/company prose names are invisible to the deterministic
        detectors, so with the LLM off they only entered the mapper via
        seeds — the LLM judge kept finding exactly this class of leak
        (CJK names/companies in the docx), and GLiNER's pixel-redaction
        finds never propagated beyond their own image/page.

        Confidence split: hits >= GLINER_AUTO_THRESHOLD (default 0.80)
        are registered in the mapper (blocked everywhere, one pass);
        hits between the base threshold and that are collected as
        SUGGESTIONS surfaced in review_candidates.json for the human
        loop. Skipped silently when GLiNER is not installed — the
        deterministic detectors, sample audit, and review loop remain.
        """
        self.suggested_entities: List[Dict] = []
        if not texts:
            return
        try:
            from acquire.catalog import (_get_gliner_model, _gliner_chunks,
                                         _gliner_predict_many,
                                         _hits_from_predictions)
            model = _get_gliner_model()
        except Exception as e:
            _logger.debug("GLiNER unavailable for discovery: %s", e)
            return
        if model is None:
            return

        from .anonymizer import PLACEHOLDER_VALUE_RE
        from .llm_detect import _stoplisted
        try:
            auto_threshold = float(
                os.environ.get('GLINER_AUTO_THRESHOLD', '0.80'))
        except ValueError:
            auto_threshold = 0.80

        if progress:
            progress.stage(
                f'up-front NER discovery (GLiNER, {len(texts)} files)')
        registered = 0
        for file_path, text in texts.items():
            chunks = list(_gliner_chunks(text))
            seen: set = set()
            hits = []
            for predictions in _gliner_predict_many(model, chunks):
                hits.extend(_hits_from_predictions(
                    predictions, str(file_path), seen))
            for hit in hits:
                if hit.entity_type not in self._GLINER_REGISTER_TYPES:
                    continue
                value = hit.value.strip()
                has_cjk = re.search(r'[぀-ヿ㐀-䶿一-鿿가-힯]', value)
                if len(value) < (2 if has_cjk else 3) or len(value) > 120:
                    continue
                if _stoplisted(value):
                    continue
                if PLACEHOLDER_VALUE_RE.fullmatch(value):
                    continue
                if hit.confidence >= auto_threshold:
                    self.mapper.get_or_create(
                        hit.entity_type, value,
                        source=f'gliner_discovery:{file_path.name}')
                    registered += 1
                else:
                    self.suggested_entities.append({
                        'type': hit.entity_type,
                        'value': value,
                        'confidence': hit.confidence,
                        'file': file_path.name,
                    })
        # Dedupe suggestions across files/chunks: one review line per
        # (type, value), keeping the best confidence and every file.
        merged: Dict[Tuple[str, str], Dict] = {}
        for suggestion in self.suggested_entities:
            key = (suggestion['type'], suggestion['value'].lower())
            kept = merged.get(key)
            if kept is None:
                kept = dict(suggestion, files=[suggestion.pop('file')])
                merged[key] = kept
            else:
                kept['confidence'] = max(
                    kept['confidence'], suggestion['confidence'])
                if suggestion['file'] not in kept['files']:
                    kept['files'].append(suggestion['file'])
        self.suggested_entities = sorted(
            merged.values(), key=lambda s: -s['confidence'])
        _logger.info(
            "GLiNER discovery: %d value(s) auto-registered (>=%.2f), "
            "%d suggestion(s) for review",
            registered, auto_threshold, len(self.suggested_entities))

    def _llm_sample_audit(self, result: 'CleanResult') -> None:
        """PROJECT_P_LLM_VERIFY=sample: audit ONE cleaned file per
        extension with the LLM, FIX what it finds, re-check.

        1. Sample: per extension, the cleaned file with the most
           extractable text.
        2. Round 1: LLM-scan the samples. Every reported value is a leak
           in cleaned output — register it in the mapper.
        3. Targeted re-clean: only files whose extracted text contains a
           found value (cheap substring test over cached extractions).
        4. Round 2: LLM-scan the samples plus the re-cleaned files.
           Residual findings (or failed scans) FAIL the run.

        LLM cost is two bounded rounds over ~one file per type instead of
        a judge pass over every file: O(types), not O(files).
        """
        from .llm_detect import (LocalLLM, detect_entities_batch,
                                 extract_scannable_text, llm_verify_mode,
                                 _llm_scan_cap)
        from .verifier import VerificationResult, LeakageHit

        if llm_verify_mode() != 'sample':
            return
        progress = getattr(self, '_progress', None)
        llm = LocalLLM()
        if not llm.available():
            result.errors.append(
                f"LLM endpoint {llm.base_url} unreachable — sample audit "
                f"could not run (PROJECT_P_LLM_VERIFY=sample).")
            self._sample_audit_result = VerificationResult(
                check_name="LLM Sample Audit", passed=False,
                details=f"Endpoint {llm.base_url} unreachable — "
                        f"output NOT LLM-audited.")
            return

        cap = _llm_scan_cap()

        def _texts(paths):
            out = {}
            for path in paths:
                text = extract_scannable_text(path)
                if text and text.strip():
                    out[str(path)] = text[:cap] if cap > 0 else text
            return out

        all_files = [f for f in sorted(self.staging_dir.rglob('*'))
                     if f.is_file() and not f.name.startswith('.')]
        texts_all = _texts(all_files)

        # One sample per extension: the file with the MOST extractable
        # text (best odds of exposing that type's leak pattern).
        best: Dict[str, Path] = {}
        for f in all_files:
            text = texts_all.get(str(f))
            if not text:
                continue
            ext = f.suffix.lower()
            if ext not in best or len(text) > len(texts_all[str(best[ext])]):
                best[ext] = f
        samples = list(best.values())
        if not samples:
            self._sample_audit_result = VerificationResult(
                check_name="LLM Sample Audit", passed=True,
                details="No text-extractable files to sample")
            return

        hits: List = []

        def _rel(key: str) -> str:
            return str(Path(key).relative_to(self.staging_dir))

        if progress:
            progress.stage(
                f'LLM sample audit round 1 '
                f'({len(samples)} samples, {llm.model})')
        round1, errors1 = detect_entities_batch(
            llm, {str(p): texts_all[str(p)] for p in samples})
        for key, err in errors1.items():
            hits.append(LeakageHit(
                file_path=_rel(key), entity_type='unverifiable',
                original=f'LLM sample scan failed: {err}'))

        # Every round-1 value is identifying text that SURVIVED cleaning:
        # register it (idempotent for known entities) so the re-clean
        # replaces it everywhere, not just in the sampled file.
        leak_values: List[str] = []
        for key, entities in round1.items():
            for entity_type, value in entities:
                self.mapper.get_or_create(
                    entity_type, value,
                    source=f'llm_sample_audit:{Path(key).name}')
                leak_values.append(value)

        recleaned: List[Path] = []
        if leak_values:
            _logger.info(
                "Sample audit round 1: %d leak value(s) across %d sample(s)",
                len(leak_values), len(samples))
            router = FileCleanerRouter(self.mapper)
            needles = {v.lower() for v in leak_values}
            if progress:
                progress.stage(
                    f're-cleaning files containing {len(needles)} '
                    f'audited value(s)')
            for f in all_files:
                text_lower = texts_all.get(str(f), '').lower()
                if not any(needle in text_lower for needle in needles):
                    continue
                ok = router.clean_file(
                    input_path=f, output_path=f, entity_spans=None)
                if ok:
                    self._normalize_mtime(f)
                    recleaned.append(f)
                else:
                    _logger.warning(
                        "Sample-audit re-clean failed for %s; "
                        "quarantining", f.name)
                    self._quarantine_file(
                        f, f.relative_to(self.staging_dir))

            # Round 2: re-check samples + everything re-cleaned.
            targets = [p for p in dict.fromkeys(samples + recleaned)
                       if p.exists()]
            if progress:
                progress.stage(
                    f'LLM sample audit round 2 ({len(targets)} files)')
            round2, errors2 = detect_entities_batch(llm, _texts(targets))
            for key, err in errors2.items():
                hits.append(LeakageHit(
                    file_path=_rel(key), entity_type='unverifiable',
                    original=f'LLM sample re-scan failed: {err}'))
            for key, entities in round2.items():
                for entity_type, value in entities:
                    hits.append(LeakageHit(
                        file_path=_rel(key),
                        entity_type=f'llm_{entity_type}',
                        original=value,
                        context='Still identifiable after sample-audit '
                                're-clean'))

        self._sample_audit_result = VerificationResult(
            check_name="LLM Sample Audit",
            passed=len(hits) == 0,
            details=(f"Sampled {len(samples)} file(s) (one per type) via "
                     f"{llm.model}; round 1 registered {len(leak_values)} "
                     f"leak value(s); re-cleaned {len(recleaned)} file(s)"),
            hits=hits,
        )

    def _inplace_reclean(self) -> int:
        """Re-clean every file currently in staging, in place.

        Used after new entities enter the mapper (LLM discovery). Files
        that fail are quarantined (fail-closed). Returns failure count.
        """
        router = FileCleanerRouter(self.mapper)
        failures = 0
        for staging_file in list(self.staging_dir.rglob('*')):
            if not staging_file.is_file() or staging_file.name.startswith('.'):
                continue
            rel_path = staging_file.relative_to(self.staging_dir)
            success = router.clean_file(
                input_path=staging_file,
                output_path=staging_file,
                entity_spans=None,
            )
            if success:
                self._normalize_mtime(staging_file)
            else:
                failures += 1
                _logger.warning(
                    "In-place re-clean failed for %s; quarantining", rel_path,
                )
                self._quarantine_file(staging_file, rel_path)
        return failures

    def _normalize_all_mtimes(self) -> None:
        """Normalize mtimes of EVERYTHING left in staging (files and dirs).

        Per-file normalization during cleaning misses directories and any
        path touched by the rename pass; a final sweep closes the gap.
        """
        for path in self.staging_dir.rglob('*'):
            self._normalize_mtime(path)
        self._normalize_mtime(self.staging_dir)

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
        audit_dir = self._audit_root() / self.project_name
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