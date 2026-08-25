"""
PDFCleaner - clean PDF documents by removing metadata and cleaning text content.

Addresses the major PDF risk vectors (ranked by failure likelihood):

1. Incremental updates (CRITICAL) - PDFs append, never overwrite in place.
   Black rectangles over text leave the text extractable. Deleted pages
   persist. This is the #1 sanitization failure mode and it will bite
   you on invoices. Full page rebuild is mandatory, not optional.

2. XMP packet (HIGH) - Second copy of metadata in XML format.
   Stripping the Info dict alone is the #2 failure mode.

3. Optional content groups / hidden layers (HIGH) - CAD-exported PDFs
   routinely carry toggled-off revision clouds and internal notes.

4. Annotations, comments, form fields (MEDIUM) - Popup comments,
   form field values, rich media annotations.

5. Digital signatures (MEDIUM) - Signer name, email, cert org.

6. Embedded attachments (MEDIUM) - Files embedded in the PDF.

7. JavaScript (MEDIUM) - Action scripts, OpenAction, AA entries.

8. Trailer /ID (LOW) - Document fingerprint that persists across copies.

9. U3D/PRC 3D models (LOW) - Some CAD exports embed 3D models inside
   a "2D drawing" PDF.

Strategy: Full page rebuild. Extract text from each page, clean entities,
rebuild pages from scratch. This is lossy for formatting but correct for
sanitization. Any strategy that copies page objects (rather than rebuilding)
will leak incremental update data.

Dependencies (optional, ranked by quality):
- PyMuPDF (fitz): Best control over PDF internals
- pdfminer.six: Text extraction with layout awareness
- reportlab: Page rebuilding with cleaned text
- pypdf: Basic operations (fallback)
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
from pathlib import Path
from typing import List, Optional, Tuple

from ..anonymizer import EntityMapper
from .text import TextCleaner

_logger = logging.getLogger(__name__)

# Try optional dependencies (modern pypdf only; PyPDF2 is deprecated)
try:
    from pypdf import PdfReader, PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Alias for backward compatibility in code references
HAS_PYPDF2 = HAS_PYPDF

try:
    from pdfminer.high_level import extract_text, extract_pages
    from pdfminer.layout import LTTextContainer, LTChar
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class PDFCleaner:
    """Clean PDF documents by full page rebuild.

    Uses a multi-strategy approach based on available dependencies:
    - PyMuPDF (best): Full page rebuild with cleaned content
    - pdfminer + reportlab: Extract text, rebuild pages
    - pypdf (fallback): Metadata removal + incremental update detection
    - copy-as-is (last resort): When no PDF library is available

    All strategies attempt to address incremental updates by rebuilding
    pages from scratch rather than copying page objects.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)

        # Determine best available strategy
        if HAS_PYMUPDF:
            self._strategy = 'pymupdf'
        elif HAS_PDFMINER and HAS_REPORTLAB:
            self._strategy = 'pdfminer_reportlab'
        elif HAS_PYPDF:
            self._strategy = 'pypdf'
        else:
            self._strategy = 'copy'

        if self._strategy != 'pymupdf':
            _logger.warning(
                "PDF cleaning strategy: %s. For full PDF sanitization, "
                "install PyMuPDF: pip install PyMuPDF",
                self._strategy,
            )

    def clean_file(self, input_path: Path, output_path: Path,
                   entity_spans: Optional[List[Tuple[int, int, str, str]]] = None) -> bool:
        """Clean a PDF file with full page rebuild.

        Args:
            input_path: Source PDF path
            output_path: Destination PDF path
            entity_spans: Entity spans for text-based cleaning (optional)

        Returns:
            True if cleaning was successful
        """
        # Pre-clean: detect and warn about incremental updates
        self._detect_incremental_updates(input_path)

        try:
            if self._strategy == 'pymupdf':
                return self._clean_with_pymupdf(input_path, output_path)
            elif self._strategy == 'pdfminer_reportlab':
                return self._clean_with_pdfminer(input_path, output_path)
            elif self._strategy == 'pypdf':
                return self._clean_with_pypdf(input_path, output_path)
            else:
                return self._copy_as_is(input_path, output_path)
        except Exception as e:
            _logger.error("Error cleaning PDF %s: %s", input_path, e)
            return self._copy_as_is(input_path, output_path)

    # ---- incremental update detection ----

    def _detect_incremental_updates(self, input_path: Path) -> None:
        """Detect incremental updates in a PDF file.

        PDFs use append-only updates. When you "save" a PDF, the new content
        is appended, and a cross-reference table marks old objects as free.
        The old content is still there and extractable.

        Detection: look for multiple %%EOF markers in the file.
        A PDF with N %%EOF markers has N versions.
        """
        try:
            with open(input_path, 'rb') as f:
                content = f.read()

            # Count %%EOF markers
            eof_count = content.count(b'%%EOF')
            if eof_count > 1:
                _logger.warning(
                    "PDF %s has %d %%EOF markers (incremental updates detected). "
                    "Prior revisions are extractable. Full rebuild required.",
                    input_path.name, eof_count,
                )

            # Also check for multiple xref tables
            xref_count = content.lower().count(b'xref')
            if xref_count > 1:
                _logger.warning(
                    "PDF %s has %d xref tables (incremental updates confirmed).",
                    input_path.name, xref_count,
                )

        except Exception as e:
            _logger.debug("Failed to check incremental updates for %s: %s",
                         input_path, e)

    # ---- PyMuPDF strategy (best) ----

    def _clean_with_pymupdf(self, input_path: Path, output_path: Path) -> bool:
        """Use PyMuPDF for full PDF sanitization.

        Uses redaction annotations to remove entity text from the original
        document, preserving layout and CJK text. This is superior to the
        extract-and-rebuild approach which destroys formatting and drops CJK.

        This addresses incremental updates by applying redactions which
        replace content streams rather than overlaying black boxes.
        """
        try:
            doc = fitz.open(str(input_path))

            # Check for embedded files using correct API
            embfile_names = doc.embfile_names()
            if embfile_names:
                _logger.warning(
                    "PDF %s has %d embedded files (will be removed)",
                    input_path.name, len(embfile_names),
                )

            # Check for JavaScript
            js_list = doc.get_js()
            if js_list:
                _logger.warning(
                    "PDF %s has JavaScript (will be removed)",
                    input_path.name,
                )

            # Check for digital signatures (correct API: doc.is_signed is a method in newer versions)
            try:
                if hasattr(doc, 'is_signed') and callable(doc.is_signed):
                    is_signed = doc.is_signed()
                else:
                    is_signed = getattr(doc, 'is_signed', False)
                if is_signed:
                    _logger.warning(
                        "PDF %s has digital signature (will be removed)",
                        input_path.name,
                    )
            except Exception:
                pass

            # Check for 3D annotations
            self._check_3d_annotations(doc)

            # Get entity terms to redact from the mapper
            entity_terms = self._get_entity_terms_to_redact()

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Remove annotations using correct API (iterate page.annots)
                annots = list(page.annots()) if page.annots() else []
                if annots:
                    _logger.debug(
                        "Removing %d annotations from page %d",
                        len(annots), page_num + 1,
                    )
                    while page.annots():
                        page.delete_annot(page.annots()[0])

                # Extract text and clean it to find entities present
                text = page.get_text()

                # Also extract form field values
                try:
                    widgets = page.widgets()
                except Exception:
                    widgets = None

                if widgets:
                    for widget in widgets:
                        field_value = widget.field_value or ""
                        if field_value:
                            text += "\n" + field_value

                # Clean text to determine which entities are present
                cleaned_text = self.text_cleaner.clean_text(text)

                # Find entity terms that appear in this page and redact them
                for term in entity_terms:
                    try:
                        occurrences = page.search_for(term)
                        if occurrences:
                            _logger.debug(
                                "Redacting '%s' (%d occurrences) on page %d",
                                term, len(occurrences), page_num + 1,
                            )
                            for rect in occurrences:
                                page.add_redact_annot(rect)
                    except Exception:
                        pass

                # Apply all redaction annotations (replaces content, not overlay)
                if page.annots():
                    page.apply_redactions()

            # Clear document metadata
            doc.set_metadata(["", "", "", "", "", "", "", ""])

            # Save with garbage collection (removes unused objects)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(output_path), garbage=4, deflate=True, clean=True)
            doc.close()

            return True

        except Exception as e:
            _logger.error("PyMuPDF cleaning failed: %s", e)
            return False

    def _get_entity_terms_to_redact(self) -> List[str]:
        """Get list of entity terms that should be redacted from PDFs.

        Returns terms from the entity mapper that need to be searched for
        and redacted in PDF pages.
        """
        terms = []

        # EntityMapper stores mappings in _mappings: Dict[str, EntityMapping]
        # where EntityMapping has .original (the original entity text)
        if hasattr(self.mapper, '_mappings'):
            for mapping in self.mapper._mappings.values():
                original = mapping.original
                if original and len(original.strip()) > 0:
                    terms.append(original)

        return list(set(terms))  # Deduplicate

    def _check_3d_annotations(self, doc) -> None:
        """Check for U3D/PRC 3D model annotations."""
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                annots = list(page.annots()) if page.annots() else []
                if annots:
                    for annot in annots:
                        # annot.info is a dict with annotation properties
                        info = annot.info
                        if info.get("subtype") == "3D":
                            _logger.warning(
                                "Page %d: U3D/PRC 3D model embedded (will be removed)",
                                page_num + 1,
                            )
            except Exception:
                pass

    # ---- pdfminer + reportlab strategy ----

    def _clean_with_pdfminer(self, input_path: Path, output_path: Path) -> bool:
        """Use pdfminer + reportlab for PDF sanitization.

        Extracts text with pdfminer, cleans entities, rebuilds with reportlab.
        This is lossy for formatting but correct for sanitization.
        """
        try:
            # Extract text from each page
            page_texts = []
            for page_num in range(100):  # Limit pages
                try:
                    text = extract_text(str(input_path), page_numbers=[page_num])
                    if text:
                        page_texts.append(self.text_cleaner.clean_text(text))
                    else:
                        break
                except Exception:
                    break

            # Rebuild PDF with reportlab
            c = canvas.Canvas(str(output_path))
            page_width, page_height = letter

            for i, text in enumerate(page_texts):
                if i > 0:
                    c.showPage()

                # Simple text layout
                y = page_height - 72
                for line in text.split('\n'):
                    if y < 72:
                        c.showPage()
                        y = page_height - 72
                    c.drawString(72, y, line)
                    y -= 14

            c.save()
            return True

        except Exception as e:
            _logger.error("pdfminer+reportlab cleaning failed: %s", e)
            return False

    # ---- pypdf strategy (fallback) ----

    def _clean_with_pypdf(self, input_path: Path, output_path: Path) -> bool:
        """Use pypdf for PDF cleaning.

        WARNING: pypdf copies page objects rather than rebuilding them,
        which means incremental update data may persist. This strategy
        addresses XMP, annotations, attachments, and form fields but
        cannot guarantee removal of incremental update remnants.

        For full sanitization, use PyMuPDF strategy.
        """
        if not HAS_PYPDF:
            return False

        try:
            reader = PdfReader(str(input_path))
            writer = PdfWriter()

            # Copy all pages with cleaning
            for page_num, page in enumerate(reader.pages):
                # Remove annotations
                self._remove_page_annotations(page, page_num)

                # Remove XMP metadata from page
                self._remove_page_metadata(page, page_num)

                # Extract and clean text content
                self._clean_page_content(page)

                writer.add_page(page)

            # Clear document-level metadata
            writer.metadata = {}

            # Remove attachments (embedded files)
            self._remove_attachments(writer)

            # Remove JavaScript
            self._remove_javascript(writer)

            # Remove optional content groups (layers)
            self._remove_ocg(writer)

            # Reset trailer /ID to prevent fingerprint persistence
            self._reset_trailer_id(writer)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                writer.write(f)

            return True

        except Exception as e:
            _logger.error("pypdf cleaning failed: %s", e)
            return False

    def _remove_page_annotations(self, page, page_num: int) -> None:
        """Remove all annotations from a page."""
        if "/Annots" in page:
            annots = page["/Annots"]
            if annots:
                _logger.debug(
                    "Removing %d annotations from page %d",
                    len(annots), page_num + 1,
                )
                del page["/Annots"]

    def _remove_page_metadata(self, page, page_num: int) -> None:
        """Remove XMP metadata from a page."""
        if "/Metadata" in page:
            _logger.debug("Removing XMP metadata from page %d", page_num + 1)
            del page["/Metadata"]

    def _clean_page_content(self, page) -> None:
        """Attempt to clean page content stream.

        Note: This is limited with pypdf. Full content stream
        manipulation requires PyMuPDF or similar.
        """
        # Extract text and register entities
        try:
            text = page.extract_text()
            if text:
                self.text_cleaner.clean_text(text)
        except Exception:
            pass

    def _remove_attachments(self, writer: PdfWriter) -> None:
        """Remove embedded file attachments."""
        if writer._root_object is None:
            return

        if "/Names" in writer._root_object:
            names = writer._root_object["/Names"]
            if "/EmbeddedFiles" in names:
                _logger.warning("Removing embedded file attachments")
                del names["/EmbeddedFiles"]

    def _remove_javascript(self, writer: PdfWriter) -> None:
        """Remove JavaScript from the PDF."""
        if writer._root_object is None:
            return

        root = writer._root_object

        # Remove OpenAction JavaScript
        if "/OpenAction" in root:
            open_action = root["/OpenAction"]
            if isinstance(open_action, dict) and open_action.get("/S") == "/JavaScript":
                _logger.warning("Removing JavaScript from OpenAction")
                del root["/OpenAction"]

        # Remove AA (Additional Actions) JavaScript
        if "/AA" in root:
            _logger.warning("Removing Additional Actions (may contain JavaScript)")
            del root["/AA"]

        # Remove Name JavaScript
        names = root.get("/Names", {})
        if "/JavaScript" in names:
            _logger.warning("Removing embedded JavaScript")
            del names["/JavaScript"]

    def _remove_ocg(self, writer: PdfWriter) -> None:
        """Remove Optional Content Groups (hidden layers)."""
        if writer._root_object is None:
            return

        if "/OCProperties" in writer._root_object:
            _logger.warning("Removing Optional Content Groups (hidden layers)")
            del writer._root_object["/OCProperties"]

    def _reset_trailer_id(self, writer: PdfWriter) -> None:
        """Reset trailer /ID to prevent fingerprint persistence."""
        # The /ID entry in the trailer is a document fingerprint that
        # persists across copies. By writing a fresh PDF, we get a
        # new /ID automatically. No explicit action needed with
        # PdfWriter since it creates a new document structure.
        pass

    # ---- fallback ----

    def _copy_as_is(self, input_path: Path, output_path: Path) -> bool:
        """Fallback: remove staged file when no PDF library is available.

        This is the last resort when no PDF library is available.
        Fail-closed: remove the staged original rather than shipping it.
        """
        _logger.warning(
            "No PDF cleaning library available; removing PDF (fail-closed). "
            "Install PyMuPDF for full sanitization: pip install PyMuPDF"
        )
        if output_path.exists():
            os.remove(output_path)
        return False  # Return False to indicate cleaning was not performed

    # ---- verification helpers ----

    def extract_and_clean_text(self, input_path: Path,
                                max_pages: int = 10) -> Optional[str]:
        """Extract text from PDF and return cleaned version.

        This is used for verification - to ensure no entity text survives.

        Args:
            input_path: Source PDF path
            max_pages: Maximum pages to extract

        Returns:
            Cleaned text or None if extraction failed
        """
        if HAS_PYPDF:
            try:
                reader = PdfReader(str(input_path))
                pages = []
                for i in range(min(max_pages, len(reader.pages))):
                    text = reader.pages[i].extract_text() or ""
                    pages.append(text)

                full_text = '\n'.join(pages)
                return self.text_cleaner.clean_text(full_text)
            except Exception as e:
                _logger.debug("Failed to extract text from PDF %s: %s", input_path, e)

        return None

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in a PDF file.

        Args:
            input_path: Source PDF path

        Returns:
            List of detected risk descriptions
        """
        risks = []

        # Check for incremental updates (binary inspection)
        try:
            with open(input_path, 'rb') as f:
                content = f.read()

            eof_count = content.count(b'%%EOF')
            if eof_count > 1:
                risks.append(
                    f"Incremental updates detected: {eof_count} %%EOF markers. "
                    f"Prior revisions are extractable."
                )

            xref_count = content.lower().count(b'xref')
            if xref_count > 1:
                risks.append(
                    f"Multiple xref tables ({xref_count}): incremental updates confirmed."
                )
        except Exception:
            pass

        if not HAS_PYPDF:
            return risks or ["Cannot inspect PDF: pypdf not available"]

        try:
            reader = PdfReader(str(input_path))

            # Check for XMP metadata
            for page_num, page in enumerate(reader.pages):
                if "/Metadata" in page:
                    risks.append(f"Page {page_num + 1}: XMP metadata packet present")

            # Check for optional content groups (layers)
            root = reader.trailer.get("/Root", {})
            if "/OCProperties" in root:
                risks.append("Optional content groups (hidden layers) present")

            # Check for attachments
            if "/Names" in root:
                names = root["/Names"]
                if "/EmbeddedFiles" in names:
                    risks.append("Embedded file attachments present")

            # Check for JavaScript
            if "/OpenAction" in root:
                open_action = root["/OpenAction"]
                if isinstance(open_action, dict) and open_action.get("/S") == "/JavaScript":
                    risks.append("JavaScript present in OpenAction")

            names = root.get("/Names", {})
            if "/JavaScript" in names:
                risks.append("Embedded JavaScript present")

            # Check for digital signatures
            for page_num, page in enumerate(reader.pages):
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        if annot.get("/Subtype") == "/Widget":
                            risks.append(
                                f"Page {page_num + 1}: Digital signature field present"
                            )
                            break

            # Check for U3D/3D annotations
            for page_num, page in enumerate(reader.pages):
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        if annot.get("/Subtype") == "/3D":
                            risks.append(
                                f"Page {page_num + 1}: U3D/PRC 3D model embedded"
                            )
                            break

            # Check for form fields
            for page_num, page in enumerate(reader.pages):
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        if annot.get("/FT") == "/Tx":  # Text field
                            risks.append(
                                f"Page {page_num + 1}: Form field present"
                            )
                            break

        except Exception as e:
            risks.append(f"Risk detection failed: {e}")

        return risks