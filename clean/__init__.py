"""Data cleaning module.

Provides tools for anonymizing sensitive information in project files:
- Consistent entity-to-placeholder mapping across all file types
- File-type-specific cleaners (text, PDF, images, audio, video)
- Deterministic verification (leakage detection, re-scanning, legibility)
- Difference tracking between original and cleaned documents
- Full pipeline orchestration

Workflow:
    1. Files are copied to /tmp/clean/{project_name}/
    2. Entities are replaced with consistent placeholders
    3. Verification ensures no sensitive data remains
    4. Diff report shows all changes made
    5. Cleaned files remain in /tmp/ until compact step consumes them
"""

from .anonymizer import (
    EntityMapper,
    EntityMapping,
    SpanBasedReplacer,
)

from .cleaner import (
    TextCleaner,
    PDFCleaner,
    ImageCleaner,
    XLSXCleaner,
    DOCXCleaner,
    CADCleaner,
    PPTXCleaner,
    ZipCleaner,
    FileCleanerRouter,
)

from .diff import (
    ChangeRecord,
    FileDiff,
    DiffReport,
    ChangeTracker,
    StructuralIntegrityCheck,
    compute_file_hash,
    compute_text_hash,
)

from .verifier import (
    LeakageHit,
    VerificationResult,
    LeakageReport,
    LeakageChecker,
    ReScanner,
    LegibilityChecker,
    ConsistencyChecker,
    verify_clean,
)

from .pipeline import (
    CleanPipeline,
    CleanResult,
    clean_project,
    clean_directory,
    DEFAULT_STAGING_DIR,
)

__all__ = [
    # Anonymizer
    "EntityMapper",
    "EntityMapping",
    "SpanBasedReplacer",
    # Cleaner
    "TextCleaner",
    "PDFCleaner",
    "ImageCleaner",
    "XLSXCleaner",
    "DOCXCleaner",
    "CADCleaner",
    "PPTXCleaner",
    "ZipCleaner",
    "FileCleanerRouter",
    # Diff
    "ChangeRecord",
    "FileDiff",
    "DiffReport",
    "ChangeTracker",
    "StructuralIntegrityCheck",
    "compute_file_hash",
    "compute_text_hash",
    # Verifier
    "LeakageHit",
    "VerificationResult",
    "LeakageReport",
    "LeakageChecker",
    "ReScanner",
    "LegibilityChecker",
    "ConsistencyChecker",
    "verify_clean",
    # Pipeline
    "CleanPipeline",
    "CleanResult",
    "clean_project",
    "clean_directory",
    "DEFAULT_STAGING_DIR",
]