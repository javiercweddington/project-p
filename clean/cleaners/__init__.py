"""
Cleaners subpackage - file-type-specific cleaning implementations.

Each format gets its own module for clarity and testability:
- text.py: Plain text files using entity spans
- pdf.py: PDF documents (extract, clean, rebuild)
- image.py: Remove EXIF/metadata from images, OCR
- xlsx.py: Remove metadata from Excel files
- docx.py: Remove metadata from Word documents
- cad.py: CAD files (STEP, IGES, STL, OBJ, DXF, SolidWorks, etc.)
- pptx.py: PowerPoint presentations
- zip_cleaner.py: Archive recursion (clean members, repack)
- router.py: FileCleanerRouter dispatch logic
"""

from .text import TextCleaner
from .pdf import PDFCleaner
from .image import ImageCleaner
from .xlsx import XLSXCleaner
from .docx import DOCXCleaner
from .cad import CADCleaner
from .pptx import PPTXCleaner
from .zip_cleaner import ZipCleaner
from .router import (
    FileCleanerRouter,
    TEXT_EXTS,
    IMAGE_EXTS,
    AUDIO_EXTS,
    VIDEO_EXTS,
    DOCUMENT_EXTS,
    CAD_EXTS,
    ZIP_EXTS,
)

__all__ = [
    'TextCleaner',
    'PDFCleaner',
    'ImageCleaner',
    'XLSXCleaner',
    'DOCXCleaner',
    'CADCleaner',
    'PPTXCleaner',
    'ZipCleaner',
    'FileCleanerRouter',
    'TEXT_EXTS',
    'IMAGE_EXTS',
    'AUDIO_EXTS',
    'VIDEO_EXTS',
    'DOCUMENT_EXTS',
    'CAD_EXTS',
    'ZIP_EXTS',
]