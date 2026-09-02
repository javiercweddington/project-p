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

import os as _os

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

# RapidOCR (PaddleOCR detection/recognition models on ONNX Runtime):
# persistent in-memory engine (no per-call subprocess), substantially
# better CJK + screenshot accuracy than tesseract, deterministic.
try:
    try:
        from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    except ImportError:
        from rapidocr import RapidOCR as _RapidOCR  # newer package name
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False
    _RapidOCR = None

_rapidocr_engine = None
_rapidocr_failed = False


def _get_rapidocr():
    """Lazy RapidOCR engine singleton.

    PROJECT_P_OCR_DEVICE=auto|cuda|cpu (default auto): 'cuda' asks ONNX
    Runtime for the GPU provider — the OCR models are tiny (~20MB + CUDA
    context), sized to fit the headroom vLLM leaves on a card. Any GPU
    init failure falls back to CPU rather than failing the run.
    """
    global _rapidocr_engine, _rapidocr_failed
    if _rapidocr_engine is not None or _rapidocr_failed:
        return _rapidocr_engine
    device = _os.environ.get('PROJECT_P_OCR_DEVICE', 'auto').strip().lower()
    want_cuda = device in ('auto', 'cuda')
    try:
        if want_cuda:
            try:
                _rapidocr_engine = _RapidOCR(
                    det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
                _logger.info("RapidOCR engine initialized (CUDA requested)")
                return _rapidocr_engine
            except TypeError:
                # Engine version without *_use_cuda kwargs: provider
                # selection happens inside onnxruntime; plain init below.
                pass
            except Exception as e:
                if device == 'cuda':
                    _logger.warning(
                        "RapidOCR CUDA init failed (%s); falling back "
                        "to CPU.", e)
        _rapidocr_engine = _RapidOCR()
        _logger.info("RapidOCR engine initialized")
    except Exception as e:
        _logger.warning("RapidOCR unavailable (%s); falling back to "
                        "tesseract if installed.", e)
        _rapidocr_failed = True
        _rapidocr_engine = None
    return _rapidocr_engine


class ImageOCR:
    """Extract text (and word boxes) from images.

    Backend (PROJECT_P_OCR_BACKEND=auto|rapidocr|tesseract, default auto):
    RapidOCR when importable, else Tesseract. Falls back gracefully when
    neither is available — callers must check `self.available`.
    """

    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self):
        pref = _os.environ.get(
            'PROJECT_P_OCR_BACKEND', 'auto').strip().lower()
        if pref == 'tesseract':
            self.backend = 'tesseract' if HAS_TESSERACT else None
        elif pref == 'rapidocr':
            self.backend = 'rapidocr' if HAS_RAPIDOCR else None
        else:  # auto
            self.backend = ('rapidocr' if HAS_RAPIDOCR
                            else 'tesseract' if HAS_TESSERACT else None)
        # Functional availability flag: the INSTANCE exists even when no
        # OCR backend is installed, so callers must check this, not `is None`.
        self.available = self.backend is not None
        if not self.available:
            _logger.warning(
                "No OCR backend available. Install RapidOCR "
                "(pip install rapidocr-onnxruntime) or Tesseract "
                "(sudo apt install tesseract-ocr && pip install pytesseract)"
            )

    # -- text extraction --

    def extract_text(self, image_path: Path, lang: str = 'eng+chi_sim') -> Optional[str]:
        """Extract text from an image file using OCR.

        Returns extracted text, or None if OCR failed or is unavailable.
        """
        if image_path.suffix.lower() not in self.SUPPORTED_EXTS:
            _logger.debug("Unsupported image format for OCR: %s", image_path.suffix)
            return None

        if self.backend == 'rapidocr':
            text = self._rapidocr_text(image_path)
            if text is not None:
                return text
            # engine init/inference failure: fall through to tesseract

        if HAS_TESSERACT:
            try:
                img = Image.open(image_path)
                return self._tesseract_text(img, lang)
            except Exception as e:
                _logger.debug("OCR failed on %s: %s", image_path, e)
        return None

    def _rapidocr_text(self, image_path: Path) -> Optional[str]:
        engine = _get_rapidocr()
        if engine is None:
            return None
        try:
            result = engine(str(image_path))
            detections = result[0] if isinstance(result, tuple) else result
            if not detections:
                return ''
            lines = [str(det[1]) for det in detections if len(det) >= 2]
            return '\n'.join(lines).strip()
        except Exception as e:
            _logger.debug("RapidOCR failed on %s: %s", image_path, e)
            return None

    def _tesseract_text(self, img, lang: str = 'eng+chi_sim') -> Optional[str]:
        try:
            text = pytesseract.image_to_string(img, lang=lang)
        except Exception as lang_err:
            # A missing traineddata (e.g. chi_sim not installed) raises a
            # TesseractError; without this fallback ALL OCR silently
            # fails on machines lacking the extra language pack.
            if lang != 'eng':
                _logger.warning(
                    "OCR with lang=%r failed (%s); retrying with 'eng'. "
                    "Install the missing traineddata for CJK coverage.",
                    lang, lang_err,
                )
                text = pytesseract.image_to_string(img, lang='eng')
            else:
                raise
        return text.strip() if text else None

    def extract_text_from_bytes(self, image_bytes: bytes, lang: str = 'eng+chi_sim') -> Optional[str]:
        """Extract text from image bytes using OCR."""
        try:
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            _logger.debug("Could not decode image bytes: %s", e)
            return None
        if self.backend == 'rapidocr':
            lines = self.ocr_lines(img)
            if lines is not None:
                return '\n'.join(
                    ' '.join(w[0] for w in words)
                    for words in lines.values()).strip()
        if HAS_TESSERACT:
            try:
                return self._tesseract_text(img, lang)
            except Exception as e:
                _logger.debug("OCR failed on image bytes: %s", e)
        return None

    # -- word boxes (for pixel redaction) --

    def ocr_lines(self, image) -> Optional[dict]:
        """OCR a PIL image into {line_key: [(word, x, y, w, h), ...]}.

        The unified structure the pixel-redaction code consumes.
        Tesseract yields true per-word boxes; RapidOCR yields line-level
        boxes that are split into per-word boxes proportionally by
        character count (redaction padding absorbs the estimation error;
        over-coverage is fail-safe here).

        Returns None when no OCR backend is functional.
        """
        if self.backend == 'rapidocr':
            lines = self._rapidocr_lines(image)
            if lines is not None:
                return lines
        if HAS_TESSERACT:
            return self._tesseract_lines(image)
        return None

    def _rapidocr_lines(self, image) -> Optional[dict]:
        engine = _get_rapidocr()
        if engine is None:
            return None
        try:
            import numpy as np
            arr = np.array(image.convert('RGB'))
            result = engine(arr)
            detections = result[0] if isinstance(result, tuple) else result
            lines = {}
            if not detections:
                return lines
            for i, det in enumerate(detections):
                if len(det) < 2:
                    continue
                box, text = det[0], str(det[1])
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                left, top = int(min(xs)), int(min(ys))
                width = max(1, int(max(xs) - min(xs)))
                height = max(1, int(max(ys) - min(ys)))
                words = text.split() or [text]
                total_chars = sum(len(w) for w in words)
                # Proportional split of the line box across its words
                # (gaps between words share the space of their letters).
                entries = []
                cursor = 0
                for word in words:
                    frac_start = cursor / max(1, total_chars)
                    cursor += len(word)
                    frac_end = cursor / max(1, total_chars)
                    wx = left + int(frac_start * width)
                    ww = max(1, int((frac_end - frac_start) * width))
                    entries.append((word, wx, top, ww, height))
                lines[(0, 0, i)] = entries
            return lines
        except Exception as e:
            _logger.debug("RapidOCR line extraction failed: %s", e)
            return None

    def ocr_text_confidence(self, image):
        """OCR a small crop into (text, mean_confidence 0-100).

        Used to adjudicate logo-template matches: printed labels read
        at high confidence ('NUMBER' = 96) while stylized script marks
        half-read at low confidence ('Milas' = 0, 'ites' = 33 on live
        Milwaukee marks). Returns ('', None) when no backend can read
        the crop (None = confidence unknown, not zero).
        """
        if self.backend == 'rapidocr':
            engine = _get_rapidocr()
            if engine is not None:
                try:
                    import numpy as np
                    result = engine(np.array(image.convert('RGB')))
                    dets = result[0] if isinstance(result, tuple) else result
                    words, scores = [], []
                    for det in dets or []:
                        if len(det) >= 3:
                            words.append(str(det[1]))
                            scores.append(float(det[2]) * 100.0)
                        elif len(det) >= 2:
                            words.append(str(det[1]))
                    text = ' '.join(words)
                    conf = (sum(scores) / len(scores)) if scores else None
                    return text, conf
                except Exception as e:
                    _logger.debug("RapidOCR confidence read failed: %s", e)
        if HAS_TESSERACT:
            try:
                data = self._tess_data(image, config='--psm 7')
                words, confs = [], []
                for t, cf in zip(data['text'], data['conf']):
                    if not t.strip():
                        continue
                    words.append(t)
                    if (float(cf) >= 0
                            and sum(ch.isalnum() for ch in t) >= 2):
                        confs.append(float(cf))
                conf = (sum(confs) / len(confs)) if confs else None
                return ' '.join(words), conf
            except Exception as e:
                _logger.debug("Tesseract confidence read failed: %s", e)
        return '', None

    def _tess_data(self, image, config: str = ''):
        try:
            return pytesseract.image_to_data(
                image, lang='eng+chi_sim', config=config,
                output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError:
            return pytesseract.image_to_data(
                image, lang='eng', config=config,
                output_type=pytesseract.Output.DICT)

    def _tesseract_lines(self, image) -> Optional[dict]:
        try:
            data = self._tess_data(image)
            lines = {}
            covered = []
            for i, word in enumerate(data['text']):
                if not word.strip():
                    continue
                key = (data['block_num'][i], data['par_num'][i],
                       data['line_num'][i])
                box = (data['left'][i], data['top'][i],
                       data['width'][i], data['height'][i])
                lines.setdefault(key, []).append((word,) + box)
                covered.append(box)

            # Sparse second pass. Full-auto segmentation routinely skips
            # isolated title-block cells on busy drawing sheets (live:
            # 'B.WARD' read at conf 91 from a crop of the DWN BY cell
            # but absent from the page-level pass — the drafter name
            # then SHIPPED on 2 of 19 sheets). --psm 11 treats the page
            # as sparse text; words whose center no existing box covers
            # are merged in under synthetic line keys.
            try:
                sparse = self._tess_data(image, config='--psm 11')
                for i, word in enumerate(sparse['text']):
                    if not word.strip():
                        continue
                    x, y = sparse['left'][i], sparse['top'][i]
                    w, h = sparse['width'][i], sparse['height'][i]
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if any(bx <= cx <= bx + bw and by <= cy <= by + bh
                           for bx, by, bw, bh in covered):
                        continue
                    lines.setdefault((99, 99, i), []).append(
                        (word, x, y, w, h))
            except Exception as e:
                _logger.debug("Sparse OCR pass failed: %s", e)

            return lines
        except Exception as e:
            _logger.debug("Tesseract line extraction failed: %s", e)
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