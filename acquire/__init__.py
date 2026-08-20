"""Data acquisition module."""

from .read import (
    ProjectFile,
    ProjectGroup,
    discover_projects,
    scan_directory,
    find_projects_by_company,
    find_projects_by_product,
    get_all_data_types,
    print_project_summary,
    parse_filename,
    is_project_file,
    DEFAULT_WATCH_DIR,
    COMMON_MEDIA_PATHS,
)

from .catalog import (
    FileCatalog,
    ProjectCatalog,
    SensitivityFlag,
    ConsolidationSuggestion,
    catalog_project,
    catalog_project_group,
    classify_file,
    suggest_consolidations,
    print_consolidation_suggestions,
)

from .metadata import (
    ImageOCR,
    SensitiveDocFlagger,
    FilenamePatternDetector,
    CADMetadataExtractor,
)

__all__ = [
    # From read
    "ProjectFile",
    "ProjectGroup",
    "discover_projects",
    "scan_directory",
    "find_projects_by_company",
    "find_projects_by_product",
    "get_all_data_types",
    "print_project_summary",
    "parse_filename",
    "is_project_file",
    "DEFAULT_WATCH_DIR",
    "COMMON_MEDIA_PATHS",
    # From catalog
    "FileCatalog",
    "ProjectCatalog",
    "SensitivityFlag",
    "ConsolidationSuggestion",
    "catalog_project",
    "catalog_project_group",
    "classify_file",
    "suggest_consolidations",
    "print_consolidation_suggestions",
    # From metadata
    "ImageOCR",
    "SensitiveDocFlagger",
    "FilenamePatternDetector",
    "CADMetadataExtractor",
]