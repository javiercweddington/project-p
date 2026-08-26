"""
FileCleanerRouter - route files to the appropriate cleaner based on file type.

Dispatches files to format-specific cleaners using extension-based matching.
All extensions are casefolded before dispatch to handle .STEP vs .step, etc.

Extension sets are defined here for centralized reference.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from ..anonymizer import EntityMapper
from .text import TextCleaner
from .pdf import PDFCleaner
from .image import ImageCleaner
from .xlsx import XLSXCleaner
from .docx import DOCXCleaner
from .cad import CADCleaner, CAD_ASCII_TEXT, CAD_BINARY_COPY, CAD_ZIP_BASED
from .pptx import PPTXCleaner
# ZipCleaner imported lazily to avoid circular import with zip_cleaner.py

_logger = logging.getLogger(__name__)


# File extension sets for routing
TEXT_EXTS = {
    '.txt', '.csv', '.tsv', '.log', '.md', '.rst',
    '.py', '.js', '.c', '.cpp', '.h', '.java', '.go', '.rs',
    '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.toml',
    '.sql', '.sh', '.bash', '.zsh',
    '.err',                # Creo error logs (text-based)
    '.plist',              # Apple property lists (XML or binary)
    '.rels',               # OPC relationship parts (XML) inside archives
    '.model',              # 3MF 3D model parts (XML) inside archives
}

IMAGE_EXTS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.ico', '.heic', '.heif',
}

DOCUMENT_EXTS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.rtf', '.odt', '.ods', '.odp',
}

# All CAD extensions (union of ASCII, binary, and ZIP-based)
CAD_EXTS = CAD_ASCII_TEXT | CAD_BINARY_COPY | CAD_ZIP_BASED

# ZIP-based extensions (including generic .zip)
ZIP_EXTS = {
    '.zip',
    '.3mf',                # 3D Manufacturing Format
}


class FileCleanerRouter:
    """Route files to the appropriate cleaner based on file type."""

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)
        self.pdf_cleaner = PDFCleaner(mapper)
        self.image_cleaner = ImageCleaner(mapper)
        self.xlsx_cleaner = XLSXCleaner(mapper)
        self.docx_cleaner = DOCXCleaner(mapper)
        self.cad_cleaner = CADCleaner(mapper)
        self.pptx_cleaner = PPTXCleaner(mapper)
        # Lazy import to avoid circular dependency
        from .zip_cleaner import ZipCleaner
        self.zip_cleaner = ZipCleaner(mapper)

    def clean_file(self, input_path: Path, output_path: Path,
                   entity_spans: Optional[List[Tuple[int, int, str, str]]] = None) -> bool:
        """Clean a file using the appropriate cleaner for its type.

        Args:
            input_path: Source file path
            output_path: Destination file path
            entity_spans: Entity spans for text-based cleaners

        Returns:
            True if cleaning was successful
        """
        ext = input_path.suffix.lower()

        # OPC '_rels/.rels' files are dotfiles with an EMPTY suffix —
        # route them as XML text explicitly or archives lose them.
        if not ext and input_path.name.lower().endswith('.rels'):
            return self.text_cleaner.clean_file(
                input_path, output_path, entity_spans,
            )

        # ZIP-based formats (recurse, clean members, repack)
        if ext in ZIP_EXTS:
            return self.zip_cleaner.clean_file(input_path, output_path)

        # CAD formats
        if ext in CAD_EXTS:
            return self.cad_cleaner.clean_file(input_path, output_path)

        # Text files
        if ext in TEXT_EXTS:
            return self.text_cleaner.clean_file(
                input_path, output_path, entity_spans,
            )

        # PDF
        if ext == '.pdf':
            return self.pdf_cleaner.clean_file(input_path, output_path)

        # Images
        if ext in IMAGE_EXTS:
            return self.image_cleaner.clean_file(input_path, output_path)

        # Excel
        if ext in ('.xlsx', '.xlsm', '.xls'):
            return self.xlsx_cleaner.clean_file(input_path, output_path)

        # Word
        if ext in ('.docx', '.doc'):
            return self.docx_cleaner.clean_file(input_path, output_path)

        # PowerPoint
        if ext in ('.pptx', '.ppt'):
            return self.pptx_cleaner.clean_file(input_path, output_path)

        # .dat files - sniff magic bytes to determine type
        if ext == '.dat':
            return self._handle_dat_file(input_path, output_path)

        # Unknown type: fail-closed, let pipeline quarantine
        _logger.debug("Unknown file type %s; fail-closed", ext)
        return False

    def _handle_dat_file(self, input_path: Path, output_path: Path) -> bool:
        """Handle .dat files by sniffing magic bytes.

        .dat extension means nothing - could be anything.
        Sniff magic bytes to determine actual format.
        """
        try:
            with open(input_path, 'rb') as f:
                header = f.read(16)

            # Check for ZIP magic bytes
            if header[:2] == b'PK':
                _logger.info(
                    "Detected ZIP magic bytes in .dat file: %s",
                    input_path.name,
                )
                return self.zip_cleaner.clean_file(input_path, output_path)

            # Check for PDF magic bytes
            if header[:4] == b'%PDF':
                _logger.info(
                    "Detected PDF magic bytes in .dat file: %s",
                    input_path.name,
                )
                return self.pdf_cleaner.clean_file(input_path, output_path)

            # Check for PNG magic bytes
            if header[:8] == b'\x89PNG\r\n\x1a\n':
                _logger.info(
                    "Detected PNG magic bytes in .dat file: %s",
                    input_path.name,
                )
                return self.image_cleaner.clean_file(input_path, output_path)

            # Check for JPEG magic bytes
            if header[:2] == b'\xff\xd8':
                _logger.info(
                    "Detected JPEG magic bytes in .dat file: %s",
                    input_path.name,
                )
                return self.image_cleaner.clean_file(input_path, output_path)

            # Check for GIF magic bytes
            if header[:4] in (b'GIF8', ):
                _logger.info(
                    "Detected GIF magic bytes in .dat file: %s",
                    input_path.name,
                )
                return self.image_cleaner.clean_file(input_path, output_path)

            # Check for OLE compound file (SolidWorks, legacy Office)
            if header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                _logger.info(
                    "Detected OLE compound file in .dat file: %s",
                    input_path.name,
                )
                return self.cad_cleaner.clean_file(input_path, output_path)

            # Default: try text cleaning
            _logger.info(
                "Unknown .dat format: %s — fail-closed for quarantine "
                "(latin-1 text cleaning of unknown binaries ships them "
                "nearly raw and can corrupt them).",
                input_path.name,
            )
            return False

        except Exception as e:
            _logger.error("Failed to handle .dat file %s: %s", input_path, e)
            return False

    def get_cleaner_for_ext(self, ext: str):
        """Get the appropriate cleaner for a file extension.

        Args:
            ext: File extension (lowercase)

        Returns:
            Cleaner instance or None if no cleaner available
        """
        if ext in TEXT_EXTS:
            return self.text_cleaner
        elif ext == '.pdf':
            return self.pdf_cleaner
        elif ext in IMAGE_EXTS:
            return self.image_cleaner
        elif ext in ('.xlsx', '.xlsm', '.xls'):
            return self.xlsx_cleaner
        elif ext in ('.docx', '.doc'):
            return self.docx_cleaner
        elif ext in ('.pptx', '.ppt'):
            return self.pptx_cleaner
        elif ext in CAD_EXTS:
            return self.cad_cleaner
        elif ext in ZIP_EXTS:
            return self.zip_cleaner
        return None