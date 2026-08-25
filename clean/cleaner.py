"""
Cleaner module - file-type-specific cleaning implementations.

This module re-exports all cleaner classes and extension sets from the
cleaners subpackage for backward compatibility.

The actual implementations are in clean/cleaners/:
- text.py: Plain text files using entity spans
- pdf.py: PDF documents (extract, clean, rebuild)
- image.py: Remove EXIF/metadata from images, OCR (uses ImageOCR from acquire)
- xlsx.py: Remove metadata from Excel files
- docx.py: Remove metadata from Word documents
- cad.py: CAD files (uses CADMetadataExtractor from acquire)
- pptx.py: PowerPoint presentations
- zip_cleaner.py: Archive recursion (clean members, repack)
- router.py: FileCleanerRouter dispatch logic

Option A: Cleaners import extractors from acquire/metadata.py:
- CADCleaner uses CADMetadataExtractor for format-specific metadata
- ImageCleaner uses ImageOCR for text detection in images

All cleaners work on copies in /tmp/ and never modify originals.
"""

from __future__ import annotations

# Re-export cleaner classes from subpackage
from .cleaners import (
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

# Re-export extension sets from subpackage
from .cleaners import (
    TEXT_EXTS,
    IMAGE_EXTS,
    DOCUMENT_EXTS,
    CAD_EXTS,
    ZIP_EXTS,
)

# Re-export from anonymizer for backward compatibility
from .anonymizer import EntityMapper, SpanBasedReplacer

__all__ = [
    # Cleaner classes
    'TextCleaner',
    'PDFCleaner',
    'ImageCleaner',
    'XLSXCleaner',
    'DOCXCleaner',
    'CADCleaner',
    'PPTXCleaner',
    'ZipCleaner',
    'FileCleanerRouter',
    # Extension sets
    'TEXT_EXTS',
    'IMAGE_EXTS',
    'DOCUMENT_EXTS',
    'CAD_EXTS',
    'ZIP_EXTS',
    # From anonymizer
    'EntityMapper',
    'SpanBasedReplacer',
]
