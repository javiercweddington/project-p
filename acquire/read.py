"""
Read module for acquiring data from USB-mounted project files.

Scans a watch directory for project archives (zip) and unpacked directories.
Parses filenames to extract project identity, timestamp, and version.
Discovers data type categories (subdirectories) within each project.

READ-ONLY GUARANTEE:
    This module is strictly READ-ONLY with respect to source data.
    - No files are created, modified, or deleted on disk.
    - Zip files are opened in read mode ('r') only.
    - Directory operations are limited to listing and reading metadata.
    - All data structures (ProjectFile, ProjectGroup) are in-memory representations.
    - The add_file method only manipulates in-memory lists and caches.

Naming convention:
    {Company} - {Product}[-YYYYMMDDTHHMMSSZ][-NNN.zip]

Examples:
    Ampere - Shower Power-20210425T055243Z-001.zip
    Clipsy LLC - GOLIGO Walker  (unzipped directory)
    RMI - Xenxo S-Ring 1.0-20210425T055329Z-005.zip
"""

import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Default watch directory (USB mount point)
DEFAULT_WATCH_DIR = Path.home() / "mnt" / "sparrows"

# Common USB media locations
COMMON_MEDIA_PATHS = [
    Path.home() / "mnt" / "sparrows",
    Path("/media/sparrows/KINGSTON"),
]

# Timestamp pattern: 20210425T055243Z
_TIMESTAMP_RE = re.compile(r'-\d{8}T\d{6}Z')
# Version pattern at end: -001.zip or -005.zip
_VERSION_RE = re.compile(r'-\d{3}\.zip$')


def parse_filename(filename: str) -> Optional[Tuple[str, str, str, int, Optional[datetime]]]:
    """
    Parse a project filename to extract its components.

    Expected format: {Company} - {Product}[-YYYYMMDDTHHMMSSZ][-NNN.zip]

    Strategy: Strip timestamp and version from the end first (they have
    unique patterns), then parse what remains as "Company - Product".

    Args:
        filename: The name of the file or directory.

    Returns:
        Tuple of (project_name, company, product, version, timestamp) or None.
    """
    name = filename

    # Extract timestamp
    timestamp = None
    ts_match = _TIMESTAMP_RE.search(name)
    if ts_match:
        ts_str = ts_match.group(0)[1:]  # Remove leading '-'
        try:
            timestamp = datetime.strptime(ts_str, '%Y%m%dT%H%M%SZ')
        except ValueError:
            pass
        name = name[:ts_match.start()] + name[ts_match.end():]

    # Extract version (only for .zip files)
    version = 0
    if filename.lower().endswith('.zip'):
        v_match = _VERSION_RE.search(filename)
        if v_match:
            version_str = v_match.group(0)[1:-4]  # Remove '-' and '.zip'
            version = int(version_str)

    # Remove .zip suffix if still present
    if name.lower().endswith('.zip'):
        name = name[:-4]

    # Now remove version from name (e.g., "Ampere - Shower Power-001" -> "Ampere - Shower Power")
    if version > 0:
        name = re.sub(r'-\d{3}$', '', name)

    # Clean up any double separators or trailing dashes
    name = name.replace('--', '-').rstrip('-')

    # Try "Company - Product" pattern (use first " - " as separator)
    if ' - ' in name:
        idx = name.index(' - ')
        company = name[:idx].strip()
        product = name[idx+3:].strip()
    else:
        company = name.strip()
        product = ""

    if not company:
        return None

    project_name = f"{company} - {product}" if product else company

    return (project_name, company, product, version, timestamp)

# Timestamp format
_TIMESTAMP_FORMAT = '%Y%m%dT%H%M%SZ'

# macOS metadata files to ignore
_MACOS_METADATA_PREFIX = '._'


class ProjectFile:
    """Represents a single file or directory belonging to a project."""

    def __init__(self, filepath: Path, project_name: str, company: str,
                 product: str, version: int = 0,
                 timestamp: Optional[datetime] = None,
                 is_zipped: bool = False):
        self.filepath = Path(filepath)
        self.project_name = project_name
        self.company = company
        self.product = product
        self.version = version
        self.timestamp = timestamp
        self._is_zipped = is_zipped
        self._data_types: Optional[Set[str]] = None

    @property
    def is_zipped(self) -> bool:
        return self._is_zipped

    @property
    def name(self) -> str:
        return self.filepath.name

    @property
    def data_types(self) -> Set[str]:
        """Discover data type categories in this file/directory."""
        if self._data_types is not None:
            return self._data_types

        self._data_types = set()

        if self.is_zipped:
            self._data_types = self._get_data_types_from_zip()
        else:
            self._data_types = self._get_data_types_from_directory()

        return self._data_types

    def _get_data_types_from_zip(self) -> Set[str]:
        """Extract data type categories from a zip file.

        Zip files typically have structure:
            ProjectName/DataType1/file.pdf
            ProjectName/DataType2/file.pdf

        Data types are at the second level (inside the project folder).
        """
        data_types = set()
        try:
            with zipfile.ZipFile(self.filepath, 'r') as zf:
                for name in zf.namelist():
                    parts = [p for p in name.split('/') if p]  # Non-empty parts
                    if len(parts) >= 2:
                        # Second level is the data type category
                        data_types.add(parts[1])
                    elif len(parts) == 1 and parts[0]:
                        # File directly in project root (no subdirectory)
                        # Could be a top-level file, skip or mark as 'root'
                        pass
        except (zipfile.BadZipFile, Exception):
            pass
        return data_types

    def _get_data_types_from_directory(self) -> Set[str]:
        """Get immediate subdirectory names as data types."""
        data_types = set()
        try:
            for item in self.filepath.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    data_types.add(item.name)
        except PermissionError:
            pass
        return data_types

    @property
    def data_type_count(self) -> int:
        return len(self.data_types)

    def __repr__(self):
        return (f"ProjectFile(name={self.project_name!r}, v{self.version}, "
                f"zipped={self.is_zipped}, types={self.data_type_count})")

    def __eq__(self, other):
        if not isinstance(other, ProjectFile):
            return False
        return self.filepath == other.filepath

    def __hash__(self):
        return hash(self.filepath)


class ProjectGroup:
    """Groups multiple versions of the same project."""

    def __init__(self, project_name: str, company: str, product: str):
        self.project_name = project_name
        self.company = company
        self.product = product
        self.files: List[ProjectFile] = []
        self._all_data_types: Optional[Set[str]] = None

    def add_file(self, file: ProjectFile):
        """Add a file/version to this project group.

        READ-ONLY: This method only manipulates in-memory data structures.
        - self.files is an in-memory list of ProjectFile references.
        - self._all_data_types is an in-memory cache (invalidated here).
        - No files on disk are created, modified, or deleted.
        """
        if file not in self.files:
            self.files.append(file)
            self._all_data_types = None  # Invalidate in-memory cache
        # Sort by version (highest last) - in-memory only
        self.files.sort(key=lambda f: f.version)

    @property
    def latest_file(self) -> Optional[ProjectFile]:
        """Return the file with the highest version number."""
        if not self.files:
            return None
        return self.files[-1]

    @property
    def version_count(self) -> int:
        return len(self.files)

    @property
    def data_types(self) -> Set[str]:
        """All unique data types across all versions of this project."""
        if self._all_data_types is not None:
            return self._all_data_types

        self._all_data_types = set()
        for f in self.files:
            self._all_data_types.update(f.data_types)
        return self._all_data_types

    @property
    def data_type_count(self) -> int:
        return len(self.data_types)

    def get_data_types_for_version(self, version: int) -> Set[str]:
        """Get data types for a specific version."""
        for f in self.files:
            if f.version == version:
                return f.data_types
        return set()

    def __repr__(self):
        return (f"ProjectGroup({self.project_name!r}, "
                f"versions={self.version_count}, "
                f"data_types={self.data_type_count})")


# (parse_filename is now defined above with the regex patterns)


def is_project_file(filepath: Path) -> bool:
    """
    Check if a file/directory looks like a project file.

    Projects are either:
    - .zip files matching the naming pattern
    - Directories matching the naming pattern
    - Not macOS metadata files (._ prefix)
    - Not random data files (xlsx, docx, etc. at top level)
    """
    # Ignore macOS metadata
    if filepath.name.startswith(_MACOS_METADATA_PREFIX):
        return False

    # Ignore hidden items
    if filepath.name.startswith('.'):
        return False

    # Zip files are candidates
    if filepath.suffix.lower() == '.zip':
        return parse_filename(filepath.name) is not None

    # Directories are candidates if they match the pattern
    if filepath.is_dir():
        return parse_filename(filepath.name) is not None

    return False


def scan_directory(watch_dir: Path) -> Dict[str, ProjectGroup]:
    """
    Scan the watch directory for project files and directories.

    Args:
        watch_dir: Directory to scan.

    Returns:
        Dictionary mapping project names to ProjectGroup objects.
    """
    if not watch_dir.exists() or not watch_dir.is_dir():
        return {}

    projects: Dict[str, ProjectGroup] = {}

    for item in watch_dir.iterdir():
        if not is_project_file(item):
            continue

        parsed = parse_filename(item.name)
        if parsed is None:
            continue

        project_name, company, product, version, timestamp = parsed
        is_zip = item.suffix.lower() == '.zip'

        file_obj = ProjectFile(
            filepath=item,
            project_name=project_name,
            company=company,
            product=product,
            version=version,
            timestamp=timestamp,
            is_zipped=is_zip,
        )

        if project_name not in projects:
            projects[project_name] = ProjectGroup(project_name, company, product)

        projects[project_name].add_file(file_obj)

    return projects


def discover_projects(watch_dir: Optional[Path] = None) -> List[ProjectGroup]:
    """
    Discover all projects, searching common locations if watch_dir is None.

    Main entry point for the read module.

    Args:
        watch_dir: Directory to scan. If None, searches common USB locations.

    Returns:
        List of ProjectGroup objects, sorted by project name.
    """
    if watch_dir is None:
        # Search common locations
        all_projects = {}
        for path in COMMON_MEDIA_PATHS:
            if path.exists():
                projects = scan_directory(path)
                for name, group in projects.items():
                    if name not in all_projects:
                        all_projects[name] = group
                    else:
                        # Merge files from same project found in different locations
                        for f in group.files:
                            all_projects[name].add_file(f)
        projects = all_projects
    else:
        projects = scan_directory(watch_dir)

    return sorted(projects.values(), key=lambda p: p.project_name.lower())


def find_projects_by_company(watch_dir: Path, company: str) -> List[ProjectGroup]:
    """
    Find all projects for a specific company.

    Args:
        watch_dir: Directory to scan.
        company: Company name to filter by (case-insensitive partial match).

    Returns:
        List of ProjectGroup objects for matching companies.
    """
    all_projects = discover_projects(watch_dir)
    company_lower = company.lower()
    return [p for p in all_projects if company_lower in p.company.lower()]


def find_projects_by_product(watch_dir: Path, product: str) -> List[ProjectGroup]:
    """
    Find all projects containing a specific product keyword.

    Args:
        watch_dir: Directory to scan.
        product: Product keyword to search for (case-insensitive partial match).

    Returns:
        List of ProjectGroup objects for matching products.
    """
    all_projects = discover_projects(watch_dir)
    product_lower = product.lower()
    return [p for p in all_projects if product_lower in p.product.lower()]


def get_all_data_types(watch_dir: Path) -> Dict[str, Set[str]]:
    """
    Get all data types for all projects.

    Args:
        watch_dir: Directory to scan.

    Returns:
        Dictionary mapping project names to their data type sets.
    """
    result = {}
    for project in discover_projects(watch_dir):
        result[project.project_name] = project.data_types
    return result


def print_project_summary(watch_dir: Optional[Path] = None) -> None:
    """
    Print a human-readable summary of all discovered projects.

    Args:
        watch_dir: Directory to scan. If None, searches common locations.
    """
    projects = discover_projects(watch_dir)

    if not projects:
        print("No projects found.")
        return

    print(f"Found {len(projects)} project(s):\n")
    print("-" * 80)

    for project in projects:
        print(f"\nProject: {project.project_name}")
        print(f"  Company: {project.company}")
        print(f"  Product: {project.product}")
        print(f"  Versions: {project.version_count}")

        for f in project.files:
            ts = f.timestamp.strftime('%Y-%m-%d') if f.timestamp else 'N/A'
            zip_str = " (zip)" if f.is_zipped else " (dir)"
            print(f"    v{f.version:03d} - {f.filepath.name}{zip_str} "
                  f"[{ts}]")

        print(f"  Data Types ({project.data_type_count}):")
        for dt in sorted(project.data_types):
            print(f"    - {dt}")

    print("\n" + "-" * 80)
    print(f"\nTotal: {len(projects)} projects")

    # Count total unique data types across all projects
    all_types = set()
    for p in projects:
        all_types.update(p.data_types)
    print(f"Unique data types across all projects: {len(all_types)}")