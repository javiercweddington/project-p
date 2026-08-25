"""
Metadata module - specialized extraction and flagging for sensitive content discovery.

Provides four classes/functions to address gaps identified in cross-product analysis:
1. ImageOCR - Extract text from images using Tesseract OCR
2. SensitiveDocFlagger - Flag files as sensitive document types based on filename patterns
3. FilenamePatternDetector - Detect sensitive document indicators in filenames
4. CADMetadataExtractor - Extract metadata from CAD/PCB/maker files

These are used by the catalog module during the acquire phase to discover
sensitive content that would otherwise be missed by text-only analysis.
"""

from __future__ import annotations

import re
import logging
import subprocess
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)


# ---- 1. ImageOCR ----

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    try:
        from PIL import Image
    except ImportError:
        pass


class ImageOCR:
    """Extract text from image files using Tesseract OCR.

    Used to detect sensitive content in images that are screenshots/scans
    of documents (invoices, receipts, payment confirmations, etc.).

    Falls back gracefully if Tesseract or PIL is not available.
    """

    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self):
        if not HAS_TESSERACT:
            _logger.warning(
                "Tesseract OCR not available. Install with: "
                "sudo apt install tesseract-ocr && pip install pytesseract pillow"
            )

    def extract_text(self, image_path: Path, lang: str = 'eng+chi_sim') -> Optional[str]:
        """Extract text from an image file using OCR.

        Args:
            image_path: Path to the image file.
            lang: Language code for Tesseract (default: 'eng+chi_sim' for bilingual support).

        Returns:
            Extracted text, or None if OCR failed or is unavailable.
        """
        if not HAS_TESSERACT:
            return None

        if image_path.suffix.lower() not in self.SUPPORTED_EXTS:
            _logger.debug("Unsupported image format for OCR: %s", image_path.suffix)
            return None

        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip() if text else None
        except Exception as e:
            _logger.debug("OCR failed on %s: %s", image_path, e)
            return None

    def extract_text_from_bytes(self, image_bytes: bytes, lang: str = 'eng+chi_sim') -> Optional[str]:
        """Extract text from image bytes using OCR.

        Args:
            image_bytes: Raw image data.
            lang: Language code for Tesseract (default: 'eng+chi_sim' for bilingual support).

        Returns:
            Extracted text, or None if OCR failed.
        """
        if not HAS_TESSERACT:
            return None

        try:
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip() if text else None
        except Exception as e:
            _logger.debug("OCR failed on image bytes: %s", e)
            return None


# ---- 2. SensitiveDocFlagger (sklearn-based) ----
#
# Purpose: Flag files that contain sensitive content based on filename
# patterns, using a trained ML classifier.
#
# Design: Uses TF-IDF (character n-grams) + LinearSVC to classify filenames
# as "sensitive" or "normal". Character n-grams are ideal for short text
# like filenames because they capture substrings that generalize across
# unseen document types.
#
# This can be combined with GLiNER: sklearn routes filenames → GLiNER
# analyzes extracted text content for entities.


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Default training data for sensitive document detection.
# These are example filenames that represent each class.
# In production, this should be expanded with real project data.
_DEFAULT_SENSITIVE_EXAMPLES = [
    # Financial documents
    "invoice_2020.pdf", "invoice_acme_march.pdf", "INV-12345.pdf",
    "quotation_v2.pdf", "quote_response.pdf",
    "payment_receipt_jan.png", "payment_confirmation.pdf",
    "receipt_001.jpg", "paid_invoice.pdf",
    "budget_2021.xlsx", "financial_report_q4.pdf",
    "tax_filing_2020.pdf", "vat_return.pdf",
    "purchase_order_1001.pdf", "PO-5432.pdf",
    # Legal/contractual
    "nda_acme_signed.pdf", "non-disclosure_agreement.pdf",
    "contract_v3.pdf", "service_agreement.pdf",
    "confidential_strategy.pdf", "proprietary_design.pdf",
    "license_agreement.pdf", "terms_of_service.pdf",
    # HR/personnel
    "resume_john_doe.pdf", "cv_smith.pdf",
    "employee_handbook.pdf", "salary_review.xlsx",
    "performance_review_2020.pdf",
    # Identity
    "passport_scan.jpg", "id_card_front.png",
    "driver_license.jpg", "ssn_form.pdf",
    # Project sensitive
    "bom_revision3.xlsx", "bill_of_materials.pdf",
    "prototype_design.pdf", "specification_v2.pdf",
    "roadmap_2021.pdf", "strategy_deck.pptx",
]

_DEFAULT_NORMAL_EXAMPLES = [
    # Generic images
    "photo_001.jpg", "screenshot.png", "image_123.bmp",
    "vacation_photo.jpg", "family_pic.png",
    # Technical but not sensitive
    "readme.txt", "changelog.md", "todo.txt",
    "test_results.log", "build_output.txt",
    # Generic documents
    "meeting_notes.pdf", "agenda.pdf", "minutes.docx",
    "presentation.pptx", "draft.docx",
    # CAD/technical
    "part_001.step", "assembly.dxf", "board_rev2.sch",
    "firmware.hex", "config.json",
]


class SensitiveDocFlagger:
    """Flag files as sensitive using a trained ML classifier.

    Uses TF-IDF with character n-grams + LinearSVC to classify filenames.
    Character n-grams (2-5) are ideal for short text like filenames because
    they capture substrings that generalize across unseen document types.

    Can be trained on custom data or uses built-in default examples.
    Falls back to minimal regex if sklearn is unavailable.
    """

    # Fallback regex pattern (only used when sklearn is unavailable)
    _FALLBACK_PATTERN = re.compile(
        r'(invoice|quotation|proposal|quote|contract|agreement|'
        r'receipt|payment|paid|confidential|nda|non-disclosure|'
        r'report|specification|bom|bill\s*of\s*materials|'
        r'resume|cv|salary|passport|id\s*card|license)',
        re.IGNORECASE,
    )

    def __init__(self, paths_train: List[str] = None,
                 y_train: List[str] = None):
        """Initialize the classifier.

        Args:
            paths_train: List of training filenames (stems or full names).
                        If None, uses default examples.
            y_train: List of labels ("sensitive" or "normal").
                    If None, uses default examples.
        """
        self._model = None
        self._use_sklearn = HAS_SKLEARN

        if self._use_sklearn:
            # Use provided training data or defaults
            X = paths_train or (_DEFAULT_SENSITIVE_EXAMPLES + _DEFAULT_NORMAL_EXAMPLES)
            y = y_train or (["sensitive"] * len(_DEFAULT_SENSITIVE_EXAMPLES) +
                           ["normal"] * len(_DEFAULT_NORMAL_EXAMPLES))

            try:
                self._model = make_pipeline(
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=1,  # Lower min_df for small training sets
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                    LinearSVC(C=1.0, class_weight="balanced", max_iter=10000),
                )
                self._model.fit(X, y)
            except Exception as e:
                _logger.warning("Failed to train sklearn model: %s", e)
                self._use_sklearn = False

    def is_sensitive_doc(self, filename: str) -> Tuple[bool, List[str]]:
        """Check if a filename indicates a sensitive document.

        Args:
            filename: The filename to check.

        Returns:
            Tuple of (is_sensitive, detected_labels).
            detected_labels contains the predicted class if sensitive.
        """
        path = Path(filename)
        text = path.stem  # Filename without extension

        if self._use_sklearn and self._model:
            try:
                prediction = self._model.predict([text])[0]
                is_sensitive = prediction == "sensitive"
                return (is_sensitive, [prediction] if is_sensitive else [])
            except Exception as e:
                _logger.debug("sklearn prediction failed: %s", e)
                return self._fallback_check(text)
        else:
            return self._fallback_check(text)

    def _fallback_check(self, text: str) -> Tuple[bool, List[str]]:
        """Fallback: regex pattern matching when sklearn unavailable."""
        matched = self._FALLBACK_PATTERN.findall(text)
        return (len(matched) > 0, [m.lower() for m in matched])

    def flag_files(self, filenames: List[str]) -> Dict[str, List[str]]:
        """Flag multiple files at once.

        Returns:
            Dict mapping filename -> list of detected labels.
            Only includes files classified as sensitive.
        """
        results = {}
        for filename in filenames:
            is_sensitive, detected = self.is_sensitive_doc(filename)
            if is_sensitive:
                results[filename] = detected
        return results

    def get_sensitivity_score(self, filename: str) -> float:
        """Get a sensitivity score (0.0 to 1.0).

        For sklearn model, returns the decision function value
        normalized to [0, 1]. For fallback, uses keyword weights.
        """
        path = Path(filename)
        text = path.stem

        if self._use_sklearn and self._model:
            try:
                # decision_function returns distance from hyperplane
                # positive = sensitive, negative = normal
                score = self._model.decision_function([text])[0]
                # Normalize using sigmoid-like transformation
                return 1.0 / (1.0 + math.exp(-score))
            except Exception:
                return 0.5
        else:
            _, matched = self._fallback_check(text)
            if not matched:
                return 0.0
            high_weight = {'confidential', 'nda', 'non-disclosure'}
            score = sum(0.5 if kw in high_weight else 0.2 for kw in matched)
            return min(score, 1.0)

    def train(self, paths: List[str], labels: List[str]) -> None:
        """Retrain the classifier with new data.

        Args:
            paths: List of training filenames.
            labels: List of labels ("sensitive" or "normal").
        """
        if not self._use_sklearn:
            _logger.warning("sklearn not available; cannot train")
            return

        try:
            self._model = make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    sublinear_tf=True,
                    lowercase=True,
                ),
                LinearSVC(C=1.0, class_weight="balanced", max_iter=10000),
            )
            self._model.fit(paths, labels)
            _logger.info("Retrained SensitiveDocFlagger with %d samples", len(paths))
        except Exception as e:
            _logger.error("Training failed: %s", e)


# ---- 3. FilenamePatternDetector ----

# Patterns for common sensitive information in filenames
_FILENAME_PATTERNS = {
    # Email addresses in filenames
    'email': re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    # Phone numbers in filenames
    'phone': re.compile(
        r'(?:\+\d{1,3}[-.\s]\d{3,4}[-.\s]\d{3,8}|\(?\d{3}\)?[-.\s]\d{3,4}[-.\s]\d{4})'
    ),
    # Person names (Title Case, two+ words) in filenames
    'person_name': re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'),
    # Date patterns (common in document filenames)
    'date': re.compile(r'\b(20\d{2}[-./]\d{1,2}[-./]\d{1,2})\b'),
    # Invoice/PO numbers
    'invoice_number': re.compile(r'\b(INV|PO|PR|SO)[-#]?\d{4,}\b', re.IGNORECASE),
    # Part numbers
    'part_number': re.compile(r'\b[P/N|PN|P/N][:.]?\s*[\w-]{3,}\b', re.IGNORECASE),
}


class FilenamePatternDetector:
    """Detect sensitive information patterns in filenames.

    Scans filenames for patterns like email addresses, phone numbers,
    person names, dates, invoice numbers, etc.

    Complements the SensitiveDocFlagger by detecting actual PII patterns
    rather than just document type keywords.
    """

    def __init__(self):
        self._patterns = _FILENAME_PATTERNS

    def detect_patterns(self, filename: str) -> List[Dict[str, str]]:
        """Detect sensitive patterns in a filename.

        Args:
            filename: The filename to scan.

        Returns:
            List of dictionaries with keys:
            - pattern_type: Type of pattern detected
            - value: The matched text
            - confidence: Confidence score (0.0 to 1.0)
        """
        results = []
        path = Path(filename)
        # Check both full path and just the filename
        texts_to_check = [str(filename), path.name, path.stem]

        seen = set()  # Deduplicate

        for pattern_type, pattern in self._patterns.items():
            for text in texts_to_check:
                for match in pattern.finditer(text):
                    value = match.group().strip()
                    # Skip very short matches
                    if len(value) < 3:
                        continue
                    # Deduplicate
                    key = (pattern_type, value.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    # Assign confidence based on pattern type
                    confidence = self._get_confidence(pattern_type, value)
                    results.append({
                        'pattern_type': pattern_type,
                        'value': value,
                        'confidence': confidence,
                    })

        return results

    def _get_confidence(self, pattern_type: str, value: str) -> float:
        """Get confidence score for a detected pattern.

        Args:
            pattern_type: Type of pattern.
            value: The matched value.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        confidence_map = {
            'email': 0.95,
            'phone': 0.85,
            'person_name': 0.6,  # Lower due to false positives
            'date': 0.5,
            'invoice_number': 0.75,
            'part_number': 0.7,
        }
        return confidence_map.get(pattern_type, 0.5)

    def has_sensitive_patterns(self, filename: str,
                                min_confidence: float = 0.7) -> bool:
        """Check if a filename contains sensitive patterns above a confidence threshold.

        Args:
            filename: The filename to check.
            min_confidence: Minimum confidence threshold.

        Returns:
            True if any sensitive patterns found above threshold.
        """
        patterns = self.detect_patterns(filename)
        return any(p['confidence'] >= min_confidence for p in patterns)


# ---- 4. CADMetadataExtractor ----

# STEP file metadata patterns
_STEP_METADATA_PATTERNS = {
    'product': re.compile(
        r'PRODUCT\([^)]*\'([^\']+)\'', re.IGNORECASE
    ),
    'description': re.compile(
        r'DESCRIPTION\([^)]*\'([^\']+)\'', re.IGNORECASE
    ),
    'directory': re.compile(
        r'DIRECTORY\([^)]*\'([^\']+)\'', re.IGNORECASE
    ),
    'designer': re.compile(
        r'DESIGNER\([^)]*\'([^\']+)\'', re.IGNORECASE
    ),
    'organization': re.compile(
        r'ORGANIZATION\([^)]*\'([^\']+)\'', re.IGNORECASE
    ),
    'time': re.compile(
        r'TIME_TIMESTAMP\([^)]*\'([^\']+)\'', re.IGNORECASE
    ),
}

# DXF file metadata patterns (in HEADER section)
_DXF_METADATA_KEYS = [
    'ACAD_MAINTVER', 'ACAD_VER', 'DWGCODEPAGE',
    'INSBASE', 'EXTMIN', 'EXTMAX', 'LIMMIN', 'LIMMAX',
    'AUTHOR', 'LAST_SAVED_BY', 'COMPANY', 'COMMENT',
]

# Eagle SCH/BRD metadata patterns (XML-based)
_EAGLE_METADATA_PATTERNS = {
    'name': re.compile(r'<name>\s*([^<]+)\s*</name>'),
    'value': re.compile(r'<value>\s*([^<]+)\s*</value>'),
    'description': re.compile(r'<description>\s*([^<]+)\s*</description>'),
    'author': re.compile(r'<author>\s*([^<]+)\s*</author>'),
    'date': re.compile(r'<date>\s*([^<]+)\s*</date>'),
    'revision': re.compile(r'<revision>\s*([^<]+)\s*</revision>'),
    'company': re.compile(r'<company>\s*([^<]+)\s*</company>'),
}


class CADMetadataExtractor:
    """Extract metadata from CAD/PCB/maker files.

    Supports:
    - STEP (.step, .stp): Text-based 3D CAD format with PRODUCT/DESCRIPTION headers
    - DXF (.dxf): AutoCAD Drawing Exchange Format with HEADER metadata
    - Eagle SCH/BRD (.sch, .brd): XML-based schematic/board files
    - Gerber (.gbr, .gtl, etc.): Text-based PCB aperture files
    - SolidWorks (.sldprt, .sldasm): Binary format, uses exiftool if available

    Falls back gracefully if exiftool is not available for binary formats.
    """

    def __init__(self):
        self._has_exiftool = self._check_exiftool()

    def _check_exiftool(self) -> bool:
        """Check if exiftool is available."""
        try:
            result = subprocess.run(
                ['exiftool', '-version'],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def extract_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from a CAD/PCB/maker file.

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary of metadata key-value pairs.
        """
        ext = file_path.suffix.lower()

        if ext in ('.step', '.stp'):
            return self._extract_step_metadata(file_path)
        elif ext == '.dxf':
            return self._extract_dxf_metadata(file_path)
        elif ext in ('.sch', '.brd'):
            return self._extract_eagle_metadata(file_path)
        elif ext in ('.gbr', '.gtl', '.gbl', '.gto', '.gbo', '.gts', '.gbs',
                     '.gps', '.gpt', '.gm1', '.gm2', '.gm3', '.gm4', '.gm5'):
            return self._extract_gerber_metadata(file_path)
        elif ext in ('.sldprt', '.sldasm'):
            return self._extract_solidworks_metadata(file_path)
        else:
            _logger.debug("No metadata extractor for extension: %s", ext)
            return {}

    def _extract_step_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from STEP files."""
        metadata = {}
        try:
            # Read only the header section (first few KB)
            with open(file_path, 'r', errors='ignore') as f:
                # Read until we find ENDSEC
                content = ''
                while len(content) < 50000:  # Limit to 50KB
                    chunk = f.read(1000)
                    if not chunk:
                        break
                    content += chunk
                    if 'ENDSEC;' in content:
                        break

            for key, pattern in _STEP_METADATA_PATTERNS.items():
                match = pattern.search(content)
                if match:
                    metadata[key] = match.group(1).strip()

        except Exception as e:
            _logger.debug("Failed to extract STEP metadata from %s: %s", file_path, e)

        return metadata

    def _extract_dxf_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from DXF files."""
        metadata = {}
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(100000)  # Read first 100KB (header section)

            # DXF format uses alternating code/value lines
            lines = content.split('\n')
            i = 0
            while i < len(lines) - 1:
                # Look for metadata keys
                if lines[i].strip() in _DXF_METADATA_KEYS:
                    key = lines[i].strip()
                    value = lines[i + 1].strip() if i + 1 < len(lines) else ''
                    if value:
                        metadata[key.lower()] = value
                i += 1

        except Exception as e:
            _logger.debug("Failed to extract DXF metadata from %s: %s", file_path, e)

        return metadata

    def _extract_eagle_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from Eagle SCH/BRD files (XML-based)."""
        metadata = {}
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(50000)  # Read first 50KB

            for key, pattern in _EAGLE_METADATA_PATTERNS.items():
                matches = pattern.findall(content)
                if matches:
                    # Take the first meaningful match
                    for match in matches:
                        value = match.strip()
                        if value and value not in ('', ' '):
                            metadata[key] = value
                            break

        except Exception as e:
            _logger.debug("Failed to extract Eagle metadata from %s: %s", file_path, e)

        return metadata

    def _extract_gerber_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from Gerber files.

        Gerber files may contain job information in format commands
        and Slader job tickets (optional).
        """
        metadata = {}
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB

            # Look for format command (e.g., %FSLAX26Y26E*%)
            format_match = re.search(r'%FSLA\w+(\d+)Y(\d+)E*%', content)
            if format_match:
                metadata['x_format'] = format_match.group(1)
                metadata['y_format'] = format_match.group(2)

            # Look for Slader job ticket (between SLDR+ and SLDR-)
            slader_match = re.search(r'SLDR\+(.*?)SLDR-', content, re.DOTALL)
            if slader_match:
                ticket = slader_match.group(1).strip()
                metadata['job_ticket'] = ticket[:200]  # Limit length

            # Look for lamp table or other job info
            for line in content.split('\n')[:20]:  # First 20 lines
                line = line.strip()
                if line.startswith('') and not line.startswith('%'):
                    # Non-format command in header might be job info
                    if len(line) > 2 and len(line) < 100:
                        metadata.get('header_comments', []).append(line)

        except Exception as e:
            _logger.debug("Failed to extract Gerber metadata from %s: %s", file_path, e)

        return metadata

    def _extract_solidworks_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from SolidWorks files using exiftool.

        SolidWorks files are binary and require exiftool for metadata extraction.
        """
        metadata = {}

        if not self._has_exiftool:
            _logger.debug(
                "exiftool not available; cannot extract SolidWorks metadata from %s",
                file_path
            )
            return metadata

        try:
            result = subprocess.run(
                ['exiftool', '-json', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if data:
                    # exiftool returns a list of file metadata
                    file_meta = data[0]
                    # Map common metadata fields
                    field_mapping = {
                        'Author': 'author',
                        'LastSavedBy': 'last_saved_by',
                        'CreatorTool': 'creator_tool',
                        'CreateDate': 'create_date',
                        'ModifyDate': 'modify_date',
                        'Description': 'description',
                        'Subject': 'subject',
                        'Comments': 'comments',
                        'Company': 'company',
                        'Manager': 'manager',
                        'Title': 'title',
                    }
                    for exif_key, our_key in field_mapping.items():
                        if exif_key in file_meta:
                            metadata[our_key] = file_meta[exif_key]

        except Exception as e:
            _logger.debug(
                "Failed to extract SolidWorks metadata from %s: %s", file_path, e
            )

        return metadata

    def extract_all_metadata(self, file_paths: List[Path]) -> Dict[str, Dict[str, str]]:
        """Extract metadata from multiple files.

        Args:
            file_paths: List of file paths to extract metadata from.

        Returns:
            Dictionary mapping filepath -> metadata dict.
            Only includes files that had extractable metadata.
        """
        results = {}
        for path in file_paths:
            metadata = self.extract_metadata(path)
            if metadata:
                results[str(path)] = metadata
        return results

    def has_sensitive_metadata(self, file_path: Path) -> Tuple[bool, Dict[str, str]]:
        """Check if a file has sensitive metadata (author, company, etc.).

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (has_sensitive, metadata_dict).
        """
        metadata = self.extract_metadata(file_path)
        sensitive_keys = {'author', 'last_saved_by', 'company', 'designer',
                         'organization', 'manager', 'comments'}
        sensitive_metadata = {
            k: v for k, v in metadata.items()
            if k.lower() in sensitive_keys and v
        }
        return (len(sensitive_metadata) > 0, sensitive_metadata)