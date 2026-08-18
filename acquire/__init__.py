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

__all__ = [
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
]
