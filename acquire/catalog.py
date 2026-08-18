"""
Catalog module - enriched file analysis built on top of the read module.

Provides a Catalog class that attaches to ProjectFile/ProjectGroup objects
from the read module and provides:
- File type catalog (images, videos, PDFs, documents, CAD files, etc.)
- File counts by type
- Sensitive information detection (company names, people, locations)

READ-ONLY: This module is strictly READ-ONLY. It only reads file metadata
and content for analysis purposes. No files are created, modified, or deleted.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import zipfile

from .read import ProjectFile, ProjectGroup


# File type classifications
IMAGE_EXTS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.tiff', '.tif', '.ico', '.heic', '.heif', '.raw', '.cr2',
}

VIDEO_EXTS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp',
}

AUDIO_EXTS = {
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a',
}

DOCUMENT_EXTS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.txt', '.rtf', '.odt', '.ods', '.odp', '.csv', '.tsv',
}

CAD_EXTS = {
    '.step', '.stp', '.stl', '.obj', '.fbx', '.blend',
    '.dwg', '.dxf', '.sldprt', '.sldasm', '.ipt', '.iam',
    '.prt', '.asm', '.x_t', '.x_b', '.iges', '.igs',
}

ELECTRONICS_EXTS = {
    '.brd', '.sch', '.pcb', '.kicad_pcb', '.kicad_sch',
    '.fzz', '.hex', '.bin', '.elf',
}

ARCHIVE_EXTS = {
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
}

FIRMWARE_EXTS = {
    '.hex', '.bin', '.elf', '.fw', '.uif', '.dfu',
}

CODE_EXTS = {
    '.py', '.js', '.c', '.cpp', '.h', '.java', '.go', '.rs',
    '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.toml',
}


FILE_TYPE_MAP = {
    'image': IMAGE_EXTS,
    'video': VIDEO_EXTS,
    'audio': AUDIO_EXTS,
    'document': DOCUMENT_EXTS,
    'cad': CAD_EXTS,
    'electronics': ELECTRONICS_EXTS,
    'archive': ARCHIVE_EXTS,
    'firmware': FIRMWARE_EXTS,
    'code': CODE_EXTS,
}


def classify_file(filename: str) -> str:
    """Classify a file by extension into a type category."""
    ext = Path(filename).suffix.lower()
    for file_type, exts in FILE_TYPE_MAP.items():
        if ext in exts:
            return file_type
    return 'other'


class SensitivityFlag:
    """A flag indicating potentially sensitive information found in a file."""

    def __init__(self, flag_type: str, value: str, source: str,
                 context: str = "", confidence: float = 1.0):
        self.flag_type = flag_type  # 'company', 'person', 'location', 'email', 'phone'
        self.value = value          # The detected sensitive value
        self.source = source        # Filename or path where it was found
        self.context = context      # Surrounding text/context
        self.confidence = confidence  # 0.0 to 1.0

    def __repr__(self):
        return (f"SensitivityFlag(type={self.flag_type!r}, value={self.value!r}, "
                f"source={self.source!r})")


class FileCatalog:
    """Catalog of file types within a single ProjectFile.

    Attaches to a ProjectFile and provides detailed file type analysis.
    """

    def __init__(self, project_file: ProjectFile):
        self.project_file = project_file
        self._files_by_type: Optional[Dict[str, List[str]]] = None
        self._file_count_by_type: Optional[Dict[str, int]] = None
        self._total_files: Optional[int] = None
        self._sensitivity_flags: Optional[List[SensitivityFlag]] = None

    @property
    def files_by_type(self) -> Dict[str, List[str]]:
        """Dictionary mapping file type to list of filenames."""
        if self._files_by_type is None:
            self._build_catalog()
        return self._files_by_type

    @property
    def file_count_by_type(self) -> Dict[str, int]:
        """Dictionary mapping file type to count of files."""
        if self._file_count_by_type is None:
            self._build_catalog()
        return self._file_count_by_type

    @property
    def total_files(self) -> int:
        if self._total_files is None:
            self._build_catalog()
        return self._total_files

    @property
    def has_images(self) -> bool:
        return self.file_count_by_type.get('image', 0) > 0

    @property
    def has_videos(self) -> bool:
        return self.file_count_by_type.get('video', 0) > 0

    @property
    def has_documents(self) -> bool:
        return self.file_count_by_type.get('document', 0) > 0

    @property
    def has_cad(self) -> bool:
        return self.file_count_by_type.get('cad', 0) > 0

    @property
    def has_audio(self) -> bool:
        return self.file_count_by_type.get('audio', 0) > 0

    @property
    def has_electronics(self) -> bool:
        return self.file_count_by_type.get('electronics', 0) > 0

    @property
    def has_firmware(self) -> bool:
        return self.file_count_by_type.get('firmware', 0) > 0

    def _build_catalog(self):
        """Build the file type catalog."""
        self._files_by_type = defaultdict(list)
        self._file_count_by_type = defaultdict(int)
        self._total_files = 0

        if self.project_file.is_zipped:
            self._catalog_zip()
        else:
            self._catalog_directory()

        # Convert defaultdicts to regular dicts
        self._files_by_type = dict(self._files_by_type)
        self._file_count_by_type = dict(self._file_count_by_type)

    def _catalog_zip(self):
        """Catalog files in a zip archive."""
        try:
            with zipfile.ZipFile(self.project_file.filepath, 'r') as zf:
                for name in zf.namelist():
                    if not name.endswith('/'):  # Skip directory entries
                        file_type = classify_file(name)
                        self._files_by_type[file_type].append(name)
                        self._file_count_by_type[file_type] += 1
                        self._total_files += 1
        except (zipfile.BadZipFile, Exception):
            pass

    def _catalog_directory(self):
        """Catalog files in a directory."""
        try:
            for filepath in self.project_file.filepath.rglob('*'):
                if filepath.is_file() and not filepath.name.startswith('._'):
                    # Get relative path
                    rel = str(filepath.relative_to(self.project_file.filepath))
                    file_type = classify_file(rel)
                    self._files_by_type[file_type].append(rel)
                    self._file_count_by_type[file_type] += 1
                    self._total_files += 1
        except PermissionError:
            pass

    def get_sensitivity_flags(self) -> List[SensitivityFlag]:
        """Analyze filenames for sensitive information.

        Scans filenames for patterns that indicate:
        - Company names
        - Person names
        - Email addresses
        - Phone numbers
        - Document types that may contain sensitive data
        """
        if self._sensitivity_flags is not None:
            return self._sensitivity_flags

        self._sensitivity_flags = []

        # Patterns for sensitive information detection
        email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        # Phone pattern: require at least one non-digit separator or + prefix
        # to avoid matching dates (e.g., 20210113)
        phone_pattern = re.compile(
            r'(?:'
            r'\+\d{1,3}[-.\s]\d{3,4}[-.\s]\d{3,8}'  # +1 555-123-4567
            r'|\(?\d{3}\)?[-.\s]\d{3,4}[-.\s]\d{4}'  # (555) 123-4567 or 555-123-4567
            r')'
        )

        # Known document types that are typically sensitive
        sensitive_doc_types = [
            'nda', 'non-disclosure', 'confidential',
            'invoice', 'quotation', 'proposal', 'contract',
            'salary', 'payment', 'receipt',
        ]

        files_to_scan = []
        if self.project_file.is_zipped:
            try:
                with zipfile.ZipFile(self.project_file.filepath, 'r') as zf:
                    files_to_scan = zf.namelist()
            except (zipfile.BadZipFile, Exception):
                pass
        else:
            try:
                for filepath in self.project_file.filepath.rglob('*'):
                    if filepath.is_file():
                        files_to_scan.append(
                            str(filepath.relative_to(self.project_file.filepath))
                        )
            except PermissionError:
                pass

        for filename in files_to_scan:
            # Skip directory entries
            if filename.endswith('/'):
                continue

            # Check for email addresses
            for match in email_pattern.finditer(filename):
                self._sensitivity_flags.append(SensitivityFlag(
                    flag_type='email',
                    value=match.group(),
                    source=filename,
                ))

            # Check for phone numbers
            for match in phone_pattern.finditer(filename):
                self._sensitivity_flags.append(SensitivityFlag(
                    flag_type='phone',
                    value=match.group(),
                    source=filename,
                ))

            # Check for sensitive document types in filename
            filename_lower = filename.lower()
            for doc_type in sensitive_doc_types:
                if doc_type in filename_lower:
                    self._sensitivity_flags.append(SensitivityFlag(
                        flag_type='sensitive_doc',
                        value=doc_type,
                        source=filename,
                        confidence=0.7,
                    ))
                    break  # One flag per file for doc type

        return self._sensitivity_flags

    def summary(self) -> str:
        """Get a human-readable summary of the catalog."""
        lines = [
            f"File Catalog for: {self.project_file.project_name} (v{self.project_file.version})",
            f"Total files: {self.total_files}",
            "",
            "File types:",
        ]

        for file_type in sorted(self.file_count_by_type.keys()):
            count = self.file_count_by_type[file_type]
            lines.append(f"  {file_type}: {count}")

        if self._sensitivity_flags:
            lines.append("")
            lines.append(f"Sensitivity flags: {len(self._sensitivity_flags)}")
            # Count by type
            flag_counts = defaultdict(int)
            for flag in self._sensitivity_flags:
                flag_counts[flag.flag_type] += 1
            for flag_type, count in sorted(flag_counts.items()):
                lines.append(f"  {flag_type}: {count}")

        return '\n'.join(lines)


class ProjectCatalog:
    """Catalog for an entire ProjectGroup (multiple versions).

    Aggregates FileCatalog objects from all versions of a project.
    """

    def __init__(self, project_group: ProjectGroup):
        self.project_group = project_group
        self._version_catalogs: Dict[int, FileCatalog] = {}
        for f in project_group.files:
            self._version_catalogs[f.version] = FileCatalog(f)

    def get_version_catalog(self, version: int) -> Optional[FileCatalog]:
        """Get the catalog for a specific version."""
        return self._version_catalogs.get(version)

    @property
    def latest_catalog(self) -> Optional[FileCatalog]:
        """Get the catalog for the latest version."""
        if not self._version_catalogs:
            return None
        latest_version = max(self._version_catalogs.keys())
        return self._version_catalogs[latest_version]

    @property
    def all_sensitivity_flags(self) -> List[SensitivityFlag]:
        """Get all sensitivity flags across all versions."""
        flags = []
        for catalog in self._version_catalogs.values():
            flags.extend(catalog.get_sensitivity_flags())
        return flags

    def summary(self) -> str:
        """Get a human-readable summary of the project catalog."""
        lines = [
            f"Project Catalog: {self.project_group.project_name}",
            f"Company: {self.project_group.company}",
            f"Product: {self.project_group.product}",
            f"Versions: {self.project_group.version_count}",
            "",
        ]

        for version, catalog in sorted(self._version_catalogs.items()):
            lines.append(catalog.summary())
            lines.append("")

        return '\n'.join(lines)


def catalog_project(project_file: ProjectFile) -> FileCatalog:
    """Convenience function to create a FileCatalog for a ProjectFile."""
    return FileCatalog(project_file)


def catalog_project_group(project_group: ProjectGroup) -> ProjectCatalog:
    """Convenience function to create a ProjectCatalog for a ProjectGroup."""
    return ProjectCatalog(project_group)