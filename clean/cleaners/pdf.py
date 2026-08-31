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
        self._image_cleaner = None  # lazy; used by the raster strategy

        # Determine best available strategy
        if HAS_PYMUPDF:
            self._strategy = 'pymupdf'
        elif HAS_PDFMINER and HAS_REPORTLAB:
            self._strategy = 'pdfminer_reportlab'
        elif HAS_PYPDF:
            self._strategy = 'pypdf'
        else:
            self._strategy = 'copy'

        # Raster mode (PROJECT_P_PDF_MODE=auto|raster|redact, default auto):
        # render pages to pixels, redact on pixels, rebuild an image-only
        # PDF. By construction the output can contain NO ghost content
        # (incremental updates, hidden layers, invisible text, XMP).
        # 'auto' uses raster whenever PyMuPDF + a functional OCR backend
        # exist; 'redact' forces the in-place text-redaction strategy.
        self._pdf_mode = os.environ.get(
            'PROJECT_P_PDF_MODE', 'auto').strip().lower()

        if self._strategy != 'pymupdf':
            _logger.warning(
                "PDF cleaning strategy: %s. For full PDF sanitization, "
                "install PyMuPDF: pip install PyMuPDF",
                self._strategy,
            )

    def _get_image_cleaner(self):
        """ImageCleaner instance shared with the raster strategy (same
        mapper, same OCR/GLiNER redaction machinery as standalone images)."""
        if self._image_cleaner is None:
            from .image import ImageCleaner
            self._image_cleaner = ImageCleaner(self.mapper)
        return self._image_cleaner

    def _raster_available(self) -> bool:
        if not HAS_PYMUPDF:
            return False
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            return False
        cleaner = self._get_image_cleaner()
        ocr = getattr(cleaner, 'image_ocr', None)
        return bool(ocr) and getattr(ocr, 'available', False)

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
            if (self._pdf_mode in ('auto', 'raster')
                    and self._raster_available()):
                if self._clean_raster(input_path, output_path):
                    return True
                # NO fallback when raster fails: raster failure means the
                # page could not be verifiably redacted, and the text
                # strategies below are a NO-OP on vector/scanned PDFs with
                # no text layer — falling back shipped a Milwaukee drawing
                # completely raw (FILE_040 live). Fail closed; the
                # pipeline quarantines.
                _logger.warning(
                    "Raster PDF cleaning failed for %s — fail-closed "
                    "(no text-strategy fallback).", input_path.name)
                return False
            # Text strategies only act on the TEXT LAYER. A vector/scanned
            # PDF with no extractable text would pass through content-
            # untouched (metadata scrub only) and "succeed" — the second
            # flavor of the FILE_040 fail-open. Refuse it.
            if not self._has_text_layer(input_path):
                _logger.warning(
                    "%s has no extractable text layer and raster mode is "
                    "unavailable — text strategies cannot clean page "
                    "content; fail-closed.", input_path.name)
                return False
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

    def _has_text_layer(self, input_path: Path) -> bool:
        """True when any page yields extractable text (first 10 pages).

        Errors count as False: if we cannot READ the text layer we
        cannot claim to have cleaned it.
        """
        try:
            if HAS_PYMUPDF:
                import fitz
                with fitz.open(input_path) as doc:
                    return any(page.get_text().strip()
                               for page in doc.pages(0, min(len(doc), 10)))
            from pypdf import PdfReader
            reader = PdfReader(str(input_path))
            return any((page.extract_text() or '').strip()
                       for page in reader.pages[:10])
        except Exception as e:
            _logger.warning("Text-layer probe failed for %s: %s",
                            input_path.name, e)
            return False

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

            # Also check for multiple xref tables. Do NOT count the 'xref'
            # inside 'startxref' (present in every PDF) — that false-flagged
            # every single-revision file.
            import re as _re
            xref_count = len(_re.findall(rb'(?<!start)xref', content.lower()))
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

        Uses redaction annotations to REPLACE entity text with its placeholder
        in the original document, preserving layout and CJK text. This is
        superior to the extract-and-rebuild approach, which destroys
        formatting and drops CJK.

        Steps:
        1. doc.scrub() when available (metadata, XMP, JavaScript, attachments,
           hidden text, thumbnails) with explicit fallbacks otherwise.
        2. Delete annotations and form widgets (popup text / field values leak).
        3. Redact every mapped entity (plus its no-space/hyphen/underscore
           variants) with the placeholder as replacement text.
        4. Full fresh save (garbage=4) — discards incremental-update history.
        5. Self-verify the OUTPUT: if any mapped entity is still extractable,
           delete the output and fail closed.
        """
        doc = None
        try:
            doc = fitz.open(str(input_path))

            if doc.is_encrypted:
                if not doc.authenticate(""):
                    _logger.warning(
                        "PDF %s is password-protected; cannot clean. "
                        "Fail-closed for pipeline quarantine.", input_path.name,
                    )
                    return False

            # Step 1: scrub document-level dangerous content.
            # Document.scrub() removes metadata, XML/XMP metadata, JavaScript,
            # attached/embedded files, hidden text, and thumbnails.
            try:
                doc.scrub()
            except AttributeError:
                # Older PyMuPDF without scrub(): explicit fallbacks.
                try:
                    for name in list(doc.embfile_names()):
                        doc.embfile_del(name)
                        _logger.info("Removed embedded file: %s", name)
                except Exception as e:
                    _logger.debug("Embedded-file removal failed: %s", e)

            # Auto-register detectable identifiers (emails, doc codes) from
            # the PDF's own text BEFORE building the redaction term list —
            # otherwise a code like JCW20191226A that appears only in this
            # PDF never enters the mapper and never gets redacted.
            try:
                full_text = '\n'.join(pg.get_text() for pg in doc)
                self.text_cleaner._register_emails(
                    full_text, source=str(input_path))
            except Exception as e:
                _logger.debug("PDF text pre-registration failed: %s", e)

            # Step 2 + 3: per-page widget/annotation removal and redaction.
            entity_terms = self._get_entity_terms_to_redact()

            # Embedded raster images (logos, stamps, signatures, photos)
            # carry identifying content that text redaction can never reach
            # ("NOA LABS" logo). Default: remove them all. Set
            # PROJECT_P_PDF_KEEP_IMAGES=1 to keep (e.g. pure technical
            # drawings you have separately vetted).
            keep_images = os.environ.get(
                'PROJECT_P_PDF_KEEP_IMAGES', '0') == '1'

            for page_num in range(len(doc)):
                page = doc[page_num]

                if not keep_images:
                    for img_info in list(page.get_images(full=True)):
                        xref = img_info[0]
                        try:
                            page.delete_image(xref)
                            _logger.info(
                                "Removed embedded image xref %d from page %d",
                                xref, page_num + 1,
                            )
                            continue
                        except Exception:
                            pass
                        # Fallback 1: swap in a blank 1x1 pixmap. Never use
                        # rect-redaction here — apply_redactions() also wipes
                        # TEXT intersecting the image bbox (logos overlapping
                        # the header ate the words next to them).
                        try:
                            blank = fitz.Pixmap(fitz.csGRAY, (0, 0, 1, 1), 0)
                            blank.clear_with(255)
                            page.replace_image(xref, pixmap=blank)
                            _logger.info(
                                "Blanked embedded image xref %d on page %d",
                                xref, page_num + 1,
                            )
                        except Exception as e:
                            _logger.warning(
                                "Could not remove or blank image xref %d on "
                                "page %d (%s) — failing closed.",
                                xref, page_num + 1, e,
                            )
                            # Unremovable image content = unverifiable PDF
                            return False

                # Delete form widgets first (field values leak).
                try:
                    for widget in list(page.widgets() or []):
                        try:
                            page.delete_widget(widget)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Delete existing annotations (comments/popups leak).
                try:
                    for annot in list(page.annots() or []):
                        try:
                            page.delete_annot(annot)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Word boxes for match expansion: search_for() returns the
                # rect of the SUBSTRING only, which leaves fragments like
                # 'ander' behind when 'Alex' matches inside 'Alexander'.
                # Expanding each match to the full word boxes it touches
                # removes the whole word cleanly.
                try:
                    word_boxes = page.get_text('words')
                except Exception:
                    word_boxes = []

                def _expand_to_words(rect):
                    """Expand a match rect to the word boxes it touches,
                    returning (expanded_rect, joined_word_text)."""
                    expanded = fitz.Rect(rect)
                    touched = []
                    for wb in word_boxes:
                        wrect = fitz.Rect(wb[:4])
                        if wrect.intersects(rect):
                            expanded |= wrect
                            touched.append(wb[4])
                    return expanded, ' '.join(touched)

                # Add a redaction for every occurrence of every entity term,
                # inserting the placeholder as the replacement text.
                # search_for() is a boundary-FREE substring search (token
                # 'ann' matches inside 'planning'), so every hit must be
                # validated against the anonymizer's boundary-aware pattern
                # on the words it actually touches before we redact them —
                # otherwise ordinary words get destroyed.
                added_redactions = False
                for term, placeholder, boundary_pattern in entity_terms:
                    try:
                        occurrences = page.search_for(term)
                    except Exception:
                        continue
                    for rect in occurrences:
                        target, touched_text = _expand_to_words(rect)
                        if (boundary_pattern is not None and touched_text
                                and not boundary_pattern.search(touched_text)):
                            # Substring-only hit inside larger words
                            # ('ann' in 'planning'): skip.
                            continue
                        try:
                            page.add_redact_annot(
                                target, text=placeholder, fontsize=6,
                            )
                            added_redactions = True
                        except Exception:
                            # Fall back to plain removal if replacement
                            # text insertion is unsupported.
                            page.add_redact_annot(target)
                            added_redactions = True
                if added_redactions:
                    page.apply_redactions()

            # Step 4: clear Info dict + XMP (again, post-scrub for safety)
            doc.set_metadata({})
            try:
                doc.del_xml_metadata()
            except Exception:
                pass

            # Full fresh save discards prior revisions (incremental updates).
            # Save via a temp file: PyMuPDF refuses a full save onto the
            # file it has open (input == output on re-clean passes).
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_name(output_path.name + '.cleantmp')
            try:
                doc.save(str(tmp_path), garbage=4, deflate=True, clean=True)
                doc.close()
                doc = None
                os.replace(tmp_path, output_path)
            except Exception:
                # Never leave a partial temp file inside the deliverable
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

            # Step 5: self-verify the output; fail closed on any residual.
            residual = self._pdf_residual_entities(output_path)
            if residual:
                _logger.warning(
                    "PDF %s still contains entity text after redaction "
                    "(%s); deleting output, fail-closed for quarantine. "
                    "(Likely split across line breaks or embedded in images.)",
                    input_path.name, ', '.join(sorted(residual)[:5]),
                )
                output_path.unlink(missing_ok=True)
                return False

            return True

        except Exception as e:
            _logger.error("PyMuPDF cleaning failed: %s", e)
            return False
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    # ---- raster strategy (render -> redact pixels -> image-only PDF) ----

    def _clean_raster(self, input_path: Path, output_path: Path) -> bool:
        """Rasterize every page, redact on pixels, rebuild an image-only PDF.

        By construction the output contains ONLY rendered pixels: no
        incremental-update history, hidden layers, invisible text, form
        fields, embedded files, JavaScript, or XMP can survive.

        Two redaction belts per page:
        1. Text layer (exact, free): every mapped entity located via
           search_for(), expanded to full word boxes, blacked out —
           boundary-validated exactly like the redact strategy.
        2. Pixels: the shared ImageCleaner redaction (OCR words + mapper
           patterns + shape rules + GLiNER), including its re-OCR
           verification. Covers text that exists only as pixels (stamps,
           logos, embedded scans) and anything extraction missed.

        Tuning: PROJECT_P_PDF_RASTER_DPI (default 200),
        PROJECT_P_PDF_RASTER_QUALITY (JPEG quality, default 80).
        """
        import io
        from PIL import Image, ImageDraw

        try:
            dpi = int(os.environ.get('PROJECT_P_PDF_RASTER_DPI', '200'))
        except ValueError:
            dpi = 200
        try:
            jpeg_quality = int(os.environ.get(
                'PROJECT_P_PDF_RASTER_QUALITY', '80'))
        except ValueError:
            jpeg_quality = 80
        zoom = dpi / 72.0

        doc = None
        newdoc = None
        try:
            doc = fitz.open(str(input_path))
            if doc.is_encrypted:
                if not doc.authenticate(""):
                    _logger.warning(
                        "PDF %s is password-protected; cannot rasterize. "
                        "Fail-closed for pipeline quarantine.",
                        input_path.name,
                    )
                    return False
            if len(doc) == 0:
                _logger.warning("PDF %s has no pages; fail-closed.",
                                input_path.name)
                return False

            entity_terms = self._get_entity_terms_to_redact()
            image_cleaner = self._get_image_cleaner()
            pages_out = []

            # Embedded raster images (signatures, stamps, logos, photos)
            # would otherwise be RENDERED INTO the page pixels — a scanned
            # signature is identifying and OCR cannot read cursive to
            # redact it. Same policy as the redact strategy: remove them
            # before rendering (PROJECT_P_PDF_KEEP_IMAGES=1 to keep).
            keep_images = os.environ.get(
                'PROJECT_P_PDF_KEEP_IMAGES', '0') == '1'

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Images that can be neither deleted nor blanked (e.g.
                # inside Form XObjects in soffice-converted PDFs — seen
                # live: 'xref not an image' on a converted .ppt) are
                # instead COVERED on the rendered pixels below, which is
                # security-equivalent in the raster path.
                cover_rects = []
                if not keep_images:
                    for img_info in list(page.get_images(full=True)):
                        xref = img_info[0]
                        try:
                            page.delete_image(xref)
                            continue
                        except Exception:
                            pass
                        try:
                            blank = fitz.Pixmap(fitz.csGRAY, (0, 0, 1, 1), 0)
                            blank.clear_with(255)
                            page.replace_image(xref, pixmap=blank)
                            continue
                        except Exception:
                            pass
                        try:
                            rects = page.get_image_rects(xref)
                        except Exception:
                            rects = []
                        if rects:
                            cover_rects.extend(rects)
                            continue
                        # get_images() sometimes lists xrefs that are not
                        # actual image objects ('xref not an image' on all
                        # three operations — seen live on a soffice-
                        # converted deck, xref 97). A non-image xref
                        # renders no pixels; skipping it strips nothing.
                        # Only a REAL image we cannot locate fails closed.
                        try:
                            obj = doc.xref_object(xref, compressed=True)
                        except Exception:
                            obj = ''
                        if '/Image' not in obj:
                            _logger.info(
                                "Skipping non-image xref %d on page %d of "
                                "%s (listed by get_images but not an "
                                "image object).",
                                xref, page_num + 1, input_path.name)
                            continue
                        _logger.warning(
                            "Could not remove, blank, or locate image "
                            "xref %d on page %d of %s — failing closed.",
                            xref, page_num + 1, input_path.name,
                        )
                        return False

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes(
                    'RGB', (pix.width, pix.height), pix.samples)

                if cover_rects:
                    cover_draw = ImageDraw.Draw(img)
                    for rect in cover_rects:
                        cover_draw.rectangle(
                            [rect.x0 * zoom, rect.y0 * zoom,
                             rect.x1 * zoom, rect.y1 * zoom],
                            fill=(0, 0, 0))
                    _logger.info(
                        "Covered %d undeletable image placement(s) on "
                        "page %d of %s at render time.",
                        len(cover_rects), page_num + 1, input_path.name)

                # Belt 1: text-layer entity boxes (exact glyph geometry).
                draw = ImageDraw.Draw(img)
                try:
                    word_boxes = page.get_text('words')
                except Exception:
                    word_boxes = []
                for term, _placeholder, boundary_pattern in entity_terms:
                    try:
                        occurrences = page.search_for(term)
                    except Exception:
                        continue
                    for rect in occurrences:
                        expanded = fitz.Rect(rect)
                        touched = []
                        for wb in word_boxes:
                            wrect = fitz.Rect(wb[:4])
                            if wrect.intersects(rect):
                                expanded |= wrect
                                touched.append(wb[4])
                        if (boundary_pattern is not None and touched
                                and not boundary_pattern.search(
                                    ' '.join(touched))):
                            # Substring-only hit inside larger words
                            continue
                        draw.rectangle(
                            [expanded.x0 * zoom - 2, expanded.y0 * zoom - 2,
                             expanded.x1 * zoom + 2, expanded.y1 * zoom + 2],
                            fill=(0, 0, 0))

                # Belt 2: pixel-level redaction shared with image files
                # (OCR + mapper patterns + shape rules + GLiNER + verify).
                redaction = image_cleaner.redact_pil(
                    img,
                    source_name=f'{input_path.name}#page{page_num + 1}')
                if redaction is None:
                    _logger.warning(
                        "Could not verify pixel redaction for %s page %d "
                        "— fail-closed.", input_path.name, page_num + 1,
                    )
                    return False
                redacted_img, _had_redactions, _word_count = redaction
                pages_out.append(
                    (redacted_img, page.rect.width, page.rect.height))

            doc.close()
            doc = None

            # Rebuild: one image per page, original page dimensions.
            newdoc = fitz.open()
            for page_img, width_pt, height_pt in pages_out:
                new_page = newdoc.new_page(width=width_pt, height=height_pt)
                buf = io.BytesIO()
                page_img.save(buf, 'JPEG', quality=jpeg_quality)
                new_page.insert_image(new_page.rect, stream=buf.getvalue())

            # Neutral metadata with FIXED dates: a fresh timestamp would
            # leak when the cleaning run happened (same policy as the
            # pipeline's mtime normalization).
            newdoc.set_metadata({
                'creationDate': 'D:20240101000000Z',
                'modDate': 'D:20240101000000Z',
                'producer': '', 'creator': '',
                'title': '', 'author': '', 'subject': '', 'keywords': '',
            })
            try:
                newdoc.del_xml_metadata()
            except Exception:
                pass

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_name(output_path.name + '.cleantmp')
            try:
                newdoc.save(str(tmp_path), garbage=4, deflate=True)
                newdoc.close()
                newdoc = None
                os.replace(tmp_path, output_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

            _logger.info(
                "Rasterized %s: %d page(s) at %d dpi -> image-only PDF",
                input_path.name, len(pages_out), dpi,
            )
            return True

        except Exception as e:
            _logger.error(
                "Raster PDF cleaning failed for %s: %s", input_path.name, e)
            return False
        finally:
            for handle in (doc, newdoc):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

    def _get_entity_terms_to_redact(self) -> List[Tuple[str, str, object]]:
        """Get (search_term, placeholder, boundary_pattern) triples.

        Includes each mapped entity's original text plus its collapsed /
        hyphenated / underscored variants and (stopword-filtered) person
        name tokens. Each triple carries the anonymizer's boundary-aware
        compiled pattern so hits from PyMuPDF's boundary-free substring
        search can be validated before redacting. Longest terms first.
        """
        from ..anonymizer import NON_TEXT_ENTITY_TYPES, PERSON_TOKEN_STOPWORDS
        import re as _re

        triples = []
        seen = set()
        for mapping in self.mapper.mappings:
            if mapping.entity_type in NON_TEXT_ENTITY_TYPES:
                continue
            original = (mapping.original or '').strip()
            if len(original) < 2:
                continue

            build = getattr(self.mapper, '_build_pattern_cached', None)
            boundary_pattern = None
            if build is not None:
                try:
                    boundary_pattern = build(original, mapping.entity_type)
                except Exception:
                    boundary_pattern = None
            if boundary_pattern is None:
                continue  # degenerate value; nothing safe to search for

            variants = {original}
            generate = getattr(self.mapper, '_generate_variants', None)
            if generate is not None:
                try:
                    variants.update(generate(original))
                except Exception:
                    pass
            # Person entities also redact by individual name token
            # ("Alex Murawski" must catch "Lech Alexander Murawski").
            if mapping.entity_type == 'person':
                for token in _re.split(r'[\s\-_]+', original.lower()):
                    token = token.strip('.,;:')
                    if len(token) >= 3 and token not in PERSON_TOKEN_STOPWORDS:
                        variants.add(token)

            for variant in variants:
                key = variant.lower()
                if len(variant) >= 2 and key not in seen:
                    seen.add(key)
                    triples.append(
                        (variant, mapping.placeholder, boundary_pattern))
        triples.sort(key=lambda p: len(p[0]), reverse=True)
        return triples

    def _pdf_residual_entities(self, pdf_path: Path) -> set:
        """Extract all text from a PDF and return mapped entities still present."""
        residual = set()
        try:
            check_doc = fitz.open(str(pdf_path))
            try:
                full_text = '\n'.join(
                    page.get_text() for page in check_doc
                ).lower()
            finally:
                check_doc.close()
        except Exception as e:
            _logger.warning("Could not self-verify PDF %s: %s", pdf_path, e)
            return {'<unverifiable>'}

        seen_patterns = set()
        for term, _ph, boundary_pattern in self._get_entity_terms_to_redact():
            if id(boundary_pattern) in seen_patterns:
                continue
            seen_patterns.add(id(boundary_pattern))
            # Boundary-aware residual check: a bare substring test would
            # flag 'ann' inside 'planning' and false-quarantine clean PDFs.
            if boundary_pattern.search(full_text):
                residual.add(term)
        return residual

    # ---- pdfminer + reportlab strategy ----

    def _clean_with_pdfminer(self, input_path: Path, output_path: Path) -> bool:
        """Use pdfminer + reportlab for PDF sanitization.

        Extracts text with pdfminer, cleans entities, rebuilds with reportlab.
        This is lossy for formatting but correct for sanitization.
        """
        try:
            # Extract the WHOLE document in one pass and split on form feeds.
            # (The previous per-page loop capped at 100 pages and stopped at
            # the first empty page, silently truncating documents.)
            full_text = extract_text(str(input_path)) or ""
            page_texts = [
                self.text_cleaner.clean_text(page_text,
                                             source=str(input_path))
                for page_text in full_text.split('\f')
            ]

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

            # Self-verify: pypdf cannot rewrite page content text, so if any
            # mapped entity remains extractable in the output, we must NOT
            # ship it. Delete the output and fail closed for quarantine.
            residual = self._pypdf_residual_entities(output_path)
            if residual:
                _logger.warning(
                    "pypdf strategy cannot remove entity text still present "
                    "in %s (%s); deleting output, fail-closed for quarantine. "
                    "Install PyMuPDF for content redaction.",
                    input_path.name, ', '.join(sorted(residual)[:5]),
                )
                output_path.unlink(missing_ok=True)
                return False

            return True

        except Exception as e:
            _logger.error("pypdf cleaning failed: %s", e)
            return False

    def _pypdf_residual_entities(self, pdf_path: Path) -> set:
        """Extract all text via pypdf and return mapped entities still present."""
        residual = set()
        try:
            reader = PdfReader(str(pdf_path))
            full_text = '\n'.join(
                (page.extract_text() or '') for page in reader.pages
            ).lower()
        except Exception as e:
            _logger.warning("Could not self-verify PDF %s: %s", pdf_path, e)
            return {'<unverifiable>'}

        seen_patterns = set()
        for term, _ph, boundary_pattern in self._get_entity_terms_to_redact():
            if id(boundary_pattern) in seen_patterns:
                continue
            seen_patterns.add(id(boundary_pattern))
            # Boundary-aware residual check: a bare substring test would
            # flag 'ann' inside 'planning' and false-quarantine clean PDFs.
            if boundary_pattern.search(full_text):
                residual.add(term)
        return residual

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
        """Fallback: fail-closed when no PDF library is available.

        This is the last resort when no PDF library is available.
        Fail-closed: return False so the pipeline can quarantine the file.
        """
        _logger.warning(
            "No PDF cleaning library available; PDF fail-closed. "
            "Install PyMuPDF for full sanitization: pip install PyMuPDF"
        )
        return False  # Return False so pipeline can quarantine

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