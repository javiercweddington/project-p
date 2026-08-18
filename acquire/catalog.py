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

from __future__ import annotations

import re
import io
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import zipfile

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from .read import ProjectFile, ProjectGroup


# ---- Sensitivity detection patterns ----

# Common company/organization name patterns (keywords that suggest a company)
_COMPANY_KEYWORDS = [
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'gmbh', 'sa', 'plc',
    'company', 'co.', 'industries', 'technologies', 'systems', 'labs',
    'medical', 'solutions', 'group', 'holdings', 'partners',
]

# Title patterns that often precede person names
_PERSON_TITLE_PATTERN = re.compile(
    r'(?:Mr|Mrs|Ms|Dr|Prof|CEO|COO|CTO|CFO|VP|Director|Manager|Engineer)'
    r'[.\s]+([A-Z][a-z]+)\s+([A-Z][a-z]+)',
)

# Location patterns
_LOCATION_PATTERNS = [
    # US States
    r'\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b',
    # Countries (English)
    r'\b(?:United States|USA|America|China|Japan|Germany|France|United Kingdom|UK|Canada|Australia|India|Brazil|Mexico|Italy|Spain|Korea|Hong Kong|Singapore|Switzerland|Netherlands|Sweden|Norway|Denmark|Finland|Belgium|Austria|Ireland|Portugal|Greece|Turkey|Russia|South Africa|Argentina|Chile|Colombia)\b',
    # Countries (Chinese)
    r'(?:中国|美国|日本|德国|法国|英国|加拿大|澳大利亚|印度|巴西|墨西哥|意大利|西班牙|韩国|香港|新加坡|瑞士|荷兰|瑞典|挪威|丹麦|芬兰|比利时|奥地利|爱尔兰|葡萄牙|希腊|土耳其|俄罗斯|南非|阿根廷|智利|哥伦比亚)',
    # Chinese Provinces
    r'(?:北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)',
    # Chinese Cities
    r'(?:北京|上海|广州|深圳|杭州|南京|成都|重庆|武汉|西安|长沙|郑州|济南|青岛|大连|沈阳|哈尔滨|长春|昆明|贵阳|南宁|福州|厦门|南昌|合肥|石家庄|太原|兰州|银川|西宁|乌鲁木齐|拉萨|呼和浩特|海口|三亚|东莞|佛山|苏州|无锡|常州|南通|徐州|扬州|镇江|泰州|盐城|淮安|宿迁|连云港|常州|温州|宁波|绍兴|嘉兴|湖州|金华|衢州|舟山|台州|丽水|杭州|南京|苏州|无锡|常州|徐州|南通|扬州|镇江|泰州|盐城|淮安|宿迁|连云港)',
    # Common city patterns (City, State or City, Country)
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+,?\s+(?:[A-Z]{2}|\w+)\b',
]

# Address pattern (street addresses - English)
_ADDRESS_PATTERN_EN = re.compile(
    r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|St|Ave|Blvd|Dr|Ln|Rd|Way|Pl|Ct|Ter|Cir)\.?\b',
    re.IGNORECASE,
)

# Address pattern (Chinese addresses)
_ADDRESS_PATTERN_CN = re.compile(
    r'(?:[^\s]{2,10}(?:省|市|区|县|镇|乡|村|路|街|大道|巷|弄|号|室|楼|栋|座|层|楼|单元))',
)

# ZIP/Postal code patterns (US)
_US_ZIP_PATTERN = re.compile(r'\b\d{5}(?:-\d{4})?\b')

# ZIP/Postal code patterns (China - 6 digits)
_CN_ZIP_PATTERN = re.compile(r'\b\d{6}\b')


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


def extract_pdf_text(filepath: Path, max_pages: int = 5) -> str:
    """Extract text from a PDF file.

    Args:
        filepath: Path to the PDF file.
        max_pages: Maximum number of pages to extract (to limit processing).

    Returns:
        Extracted text as a string, or empty string if extraction fails.
    """
    if not HAS_PYPDF:
        return ""
    try:
        reader = PdfReader(str(filepath))
        pages = []
        for i in range(min(max_pages, len(reader.pages))):
            pages.append(reader.pages[i].extract_text() or "")
        return '\n'.join(pages)
    except Exception:
        return ""


def extract_pdf_text_from_zip(zip_path: Path, zip_entry: str, max_pages: int = 5) -> str:
    """Extract text from a PDF file inside a zip archive.

    Args:
        zip_path: Path to the zip file.
        zip_entry: Path of the PDF entry inside the zip.
        max_pages: Maximum number of pages to extract.

    Returns:
        Extracted text as a string, or empty string if extraction fails.
    """
    if not HAS_PYPDF:
        return ""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(zip_entry) as pdf_file:
                reader = PdfReader(io.BytesIO(pdf_file.read()))
                pages = []
                for i in range(min(max_pages, len(reader.pages))):
                    pages.append(reader.pages[i].extract_text() or "")
                return '\n'.join(pages)
    except Exception:
        return ""


def scan_text_for_sensitivity(text: str, source: str) -> List[SensitivityFlag]:
    """Scan extracted text for sensitive information.

    Args:
        text: Text content to scan.
        source: Source filename/path for attribution.

    Returns:
        List of SensitivityFlag objects for detected sensitive information.
    """
    flags = []
    text_lower = text.lower()

    # Email addresses
    for match in re.finditer(r'[\w\.-]+@[\w\.-]+\.\w+', text):
        flags.append(SensitivityFlag(
            flag_type='email',
            value=match.group(),
            source=source,
            context=_get_context(text, match.start()),
        ))

    # Phone numbers
    for match in re.finditer(
        r'(?:\+\d{1,3}[-.\s]\d{3,4}[-.\s]\d{3,8}|\(?\d{3}\)?[-.\s]\d{3,4}[-.\s]\d{4})',
        text
    ):
        flags.append(SensitivityFlag(
            flag_type='phone',
            value=match.group(),
            source=source,
            context=_get_context(text, match.start()),
        ))

    # Company names (look for patterns like "Company Name LLC" or "Name Technologies")
    for keyword in _COMPANY_KEYWORDS:
        pattern = re.compile(
            r'([A-Z][A-Za-z\s.&]+(?:' + keyword + r'))',
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            company = match.group(1).strip()
            if 2 <= len(company.split()) <= 5:  # Reasonable company name length
                flags.append(SensitivityFlag(
                    flag_type='company',
                    value=company,
                    source=source,
                    context=_get_context(text, match.start()),
                    confidence=0.6,
                ))

    # Person names via titles (e.g., "CEO Lech Murawski", "Dr. John Smith")
    for match in _PERSON_TITLE_PATTERN.finditer(text):
        first, last = match.group(1), match.group(2)
        name = f"{first} {last}"
        flags.append(SensitivityFlag(
            flag_type='person',
            value=name,
            source=source,
            context=_get_context(text, match.start()),
            confidence=0.8,
        ))

    # Locations (states, countries, Chinese provinces/cities)
    for pattern in _LOCATION_PATTERNS:
        for match in re.finditer(pattern, text):
            location = match.group().strip()
            if len(location) > 2:  # Avoid false positives
                flags.append(SensitivityFlag(
                    flag_type='location',
                    value=location,
                    source=source,
                    context=_get_context(text, match.start()),
                    confidence=0.7,
                ))

    # US ZIP codes (indicate addresses)
    for match in _US_ZIP_PATTERN.finditer(text):
        flags.append(SensitivityFlag(
            flag_type='address',
            value=match.group(),
            source=source,
            context=_get_context(text, match.start()),
            confidence=0.5,
        ))

    # Chinese ZIP codes (6 digits)
    for match in _CN_ZIP_PATTERN.finditer(text):
        flags.append(SensitivityFlag(
            flag_type='address',
            value=match.group(),
            source=source,
            context=_get_context(text, match.start()),
            confidence=0.4,  # Lower confidence - 6 digits could be other things
        ))

    # Chinese addresses
    for match in _ADDRESS_PATTERN_CN.finditer(text):
        flags.append(SensitivityFlag(
            flag_type='address',
            value=match.group(),
            source=source,
            context=_get_context(text, match.start()),
            confidence=0.6,
        ))

    return flags


def _get_context(text: str, pos: int, window: int = 50) -> str:
    """Get surrounding context text around a position."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end].replace('\n', ' ').strip()
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."
    return context


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

    def get_sensitivity_flags(self, scan_pdf_content: bool = True) -> List[SensitivityFlag]:
        """Analyze filenames AND PDF content for sensitive information.

        Scans for patterns that indicate:
        - Company names
        - Person names
        - Email addresses
        - Phone numbers
        - Locations / addresses
        - Document types that may contain sensitive data

        Args:
            scan_pdf_content: If True, also extract and scan PDF text content.
        """
        if self._sensitivity_flags is not None:
            return self._sensitivity_flags

        self._sensitivity_flags = []

        # --- Phase 1: Scan filenames ---
        email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        phone_pattern = re.compile(
            r'(?:'
            r'\+\d{1,3}[-.\s]\d{3,4}[-.\s]\d{3,8}'
            r'|\(?\d{3}\)?[-.\s]\d{3,4}[-.\s]\d{4}'
            r')'
        )
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
            if filename.endswith('/'):
                continue

            for match in email_pattern.finditer(filename):
                self._sensitivity_flags.append(SensitivityFlag(
                    flag_type='email', value=match.group(), source=filename,
                ))
            for match in phone_pattern.finditer(filename):
                self._sensitivity_flags.append(SensitivityFlag(
                    flag_type='phone', value=match.group(), source=filename,
                ))
            filename_lower = filename.lower()
            for doc_type in sensitive_doc_types:
                if doc_type in filename_lower:
                    self._sensitivity_flags.append(SensitivityFlag(
                        flag_type='sensitive_doc', value=doc_type,
                        source=filename, confidence=0.7,
                    ))
                    break

        # --- Phase 2: Scan PDF content ---
        if scan_pdf_content and HAS_PYPDF:
            pdf_files = []
            if self.project_file.is_zipped:
                try:
                    with zipfile.ZipFile(self.project_file.filepath, 'r') as zf:
                        pdf_files = [
                            n for n in zf.namelist()
                            if n.lower().endswith('.pdf') and not n.endswith('/')
                        ]
                except (zipfile.BadZipFile, Exception):
                    pass
                for pdf_entry in pdf_files:
                    text = extract_pdf_text_from_zip(self.project_file.filepath, pdf_entry)
                    if text:
                        self._sensitivity_flags.extend(
                            scan_text_for_sensitivity(text, pdf_entry)
                        )
            else:
                try:
                    for filepath in self.project_file.filepath.rglob('*'):
                        if filepath.is_file() and filepath.suffix.lower() == '.pdf':
                            text = extract_pdf_text(filepath)
                            if text:
                                rel = str(filepath.relative_to(self.project_file.filepath))
                                self._sensitivity_flags.extend(
                                    scan_text_for_sensitivity(text, rel)
                                )
                except PermissionError:
                    pass

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