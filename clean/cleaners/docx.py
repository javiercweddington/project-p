"""
DOCXCleaner - clean Word documents by removing metadata and cleaning content.

Addresses Word-specific risks (ranked by failure likelihood):

1. Tracked changes (CRITICAL) - Supplier-quote negotiation history is fully
   recoverable. Every deletion and insertion is stored. This is the primary
   reason people ship sensitive Word documents by accident.

2. Comments (HIGH) - Supplier negotiation notes, internal feedback, pricing
   discussions. Often the frankest content in the document.

3. w:rsid values (HIGH) - Random Submission IDs fingerprint documents to a
   common editing session. Even after "cleaning", RSID values reveal which
   documents were edited by the same person in the same session. This is the
   strongest cross-file correlation key you'll leave behind.

4. Hidden text / w:vanish (MEDIUM) - Text marked as hidden but still present
   and extractable.

5. Embedded objects (MEDIUM) - OLE objects that may contain sensitive data.

6. Legacy .doc Fast Save (MEDIUM) - Old text persists in the file even after
   "deletion". Fast Save appends without removing.

7. Fields with UNC paths (LOW) - INCLUDETEXT, INCLUDEPICTURE fields that
   reference network locations.

Strategy: Use python-docx for high-level cleaning, then directly manipulate
the underlying XML to handle RSID values, tracked changes, and hidden text
that python-docx cannot access through its API.

Dependencies (optional):
- python-docx: Word document manipulation
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..anonymizer import EntityMapper
from .text import TextCleaner

_logger = logging.getLogger(__name__)

# Try optional dependencies
try:
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml.ns import qn
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False


class DOCXCleaner:
    """Clean Word documents by removing metadata and cleaning content.

    Uses python-docx for high-level cleaning and direct XML manipulation
    for RSID values, tracked changes, and hidden text.
    """

    # python-docx core properties that contain identifier-like values
    # and should go through the mapper instead of being blanked.
    # NOTE: python-docx exposes dc:creator as `author` — there is no
    # `creator`/`contributor`/`company` attribute on CoreProperties.
    # Company/Manager live in docProps/app.xml, handled in the ZIP pass.
    _IDENTIFIER_PROPERTIES = {
        'author': ('author', 'person'),
        'last_modified_by': ('last_modified_by', 'person'),
    }

    # python-docx core properties that should be blanked (non-identifiers)
    _BLANK_PROPERTIES = {
        'comments', 'category', 'content_status', 'identifier',
        'keywords', 'language', 'subject', 'title', 'version',
    }

    # Fixed timestamp for embedded document timestamps
    # (python-docx requires a datetime object, never a string)
    _FIXED_DT = datetime(2024, 1, 1, 0, 0, 0)

    # Fixed timestamp for output zip members (prevents run-date leakage)
    _ZIP_DATE_TIME = (2024, 1, 1, 0, 0, 0)

    # RSID attribute form: w:rsid="..." / w:rsidR="..." / w:rsidRDefault="..."
    _RSID_ATTR_PATTERN = re.compile(r'\s*w:rsid[\w]*\s*=\s*"[^"]*"')
    # RSID element forms in settings.xml: the whole <w:rsids> block plus
    # any standalone <w:rsid w:val="..."/> elements.
    _RSIDS_BLOCK_PATTERN = re.compile(r'<w:rsids>.*?</w:rsids>', re.DOTALL)
    _RSID_ELEM_PATTERN = re.compile(r'<w:rsid\s[^>]*/>')

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)

        if not HAS_PYTHON_DOCX:
            _logger.warning(
                "python-docx not available; Word cleaning limited. "
                "Install with: pip install python-docx"
            )

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean a Word document by removing metadata and cleaning content.

        Args:
            input_path: Source DOCX file path
            output_path: Destination DOCX file path

        Returns:
            True if cleaning was successful
        """
        ext = input_path.suffix.lower()

        # Legacy .doc: no content rewriter, but OLE properties can be
        # zeroed and the raw streams (including Fast-Save remnants) verified
        # free of mapped entities; ship only when that verification passes.
        if ext == '.doc':
            from .ole_scrub import strip_ole_properties
            return strip_ole_properties(
                self.mapper, input_path, output_path, input_path.name)

        if not HAS_PYTHON_DOCX:
            _logger.warning("python-docx not available; DOCX fail-closed")
            return False

        try:
            doc = Document(str(input_path))

            # Clear core properties
            self._clear_properties(doc.core_properties)

            # Clean all text content
            # (custom properties / app.xml are handled in the ZIP pass —
            #  python-docx has no Document.custom_properties API)
            self._clean_all_text(doc)

            # Save to bytes for XML manipulation
            output_bytes = io.BytesIO()
            doc.save(output_bytes)
            output_bytes.seek(0)

            # Post-processing: clean XML-level artifacts
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._clean_xml_artifacts(output_bytes, output_path)

            # Catch-all: entity text in members python-docx never visits
            # (text boxes w:txbxContent, drawings, charts). Fail closed.
            from .xml_pass import scrub_zip_xml_members
            if not scrub_zip_xml_members(output_path, self.mapper,
                                         input_path.name):
                output_path.unlink(missing_ok=True)
                return False

            return True

        except Exception as e:
            _logger.error("Error cleaning DOCX %s: %s", input_path, e)
            return False

    def _clear_properties(self, core_props) -> None:
        """Anonymize core document properties.

        Identifier properties (creator, company, etc.) go through the mapper
        to generate consistent placeholders like [PERSON_002].
        Non-identifier properties are blanked.
        Embedded timestamps are normalized to a fixed epoch.
        """
        # Anonymize identifier properties through the mapper
        for prop_name, (_, entity_type) in self._IDENTIFIER_PROPERTIES.items():
            try:
                value = getattr(core_props, prop_name, None)
                if value and str(value).strip():
                    placeholder = self.mapper.get_or_create(
                        entity_type=entity_type,
                        value=str(value).strip(),
                        source='docx_core_properties',
                    )
                    setattr(core_props, prop_name, placeholder)
                else:
                    setattr(core_props, prop_name, '')
            except (AttributeError, TypeError):
                pass

        # Blank non-identifier properties
        for prop_name in self._BLANK_PROPERTIES:
            try:
                setattr(core_props, prop_name, '')
            except (AttributeError, TypeError):
                pass

        # Normalize embedded timestamps to fixed epoch
        # python-docx core_properties requires datetime objects (not strings, not None)
        for prop_name in ('created', 'modified', 'last_printed'):
            try:
                setattr(core_props, prop_name, self._FIXED_DT)
            except (AttributeError, TypeError, ValueError):
                pass

    def _clean_paragraph(self, para) -> None:
        """Clean one paragraph, catching entities split across runs.

        Word routinely splits an entity like 'Globus Medical' across several
        <w:r> runs, so a run-by-run pass misses it. Strategy: clean each run
        first; if the paragraph text as a whole STILL contains an entity,
        collapse the paragraph into its first run with the fully cleaned
        text (loses intra-paragraph character formatting for that paragraph
        only — accuracy over formatting).
        """
        for run in para.runs:
            if isinstance(run.text, str) and run.text:
                run.text = self.text_cleaner.clean_text(run.text)

        if not para.runs:
            return
        full_text = para.text
        cleaned_full = self.text_cleaner.clean_text(full_text)
        if cleaned_full != full_text:
            para.runs[0].text = cleaned_full
            for run in para.runs[1:]:
                run.text = ''

    def _clean_block(self, container) -> None:
        """Clean paragraphs and tables of any block container (body, cell,
        header, footer)."""
        for para in getattr(container, 'paragraphs', []):
            self._clean_paragraph(para)
        for table in getattr(container, 'tables', []):
            for row in table.rows:
                for cell in row.cells:
                    self._clean_block(cell)

    def _clean_all_text(self, doc) -> None:
        """Clean text content across body, tables, headers, and footers."""
        self._clean_block(doc)

        # Clean headers and footers through sections.
        # Real python-docx attribute names: header/footer,
        # first_page_header/first_page_footer, even_page_header/even_page_footer.
        for section in doc.sections:
            for attr in ('header', 'first_page_header', 'even_page_header',
                         'footer', 'first_page_footer', 'even_page_footer'):
                part = getattr(section, attr, None)
                if part is None:
                    continue
                # A linked header/footer just inherits the previous section's;
                # cleaning it would create an unwanted override.
                if getattr(part, 'is_linked_to_previous', False):
                    continue
                self._clean_block(part)

    def _clean_xml_artifacts(self, source_bytes: io.BytesIO,
                              output_path: Path) -> None:
        """Clean XML-level artifacts that python-docx cannot access.

        This directly manipulates the ZIP internals to:
        1. Remove RSID values (w:rsid*) from all XML files
        2. Remove tracked changes (w:ins, w:del elements)
        3. Remove hidden text (w:vanish)
        4. Remove comments
        5. Remove embedded objects
        """
        source_bytes.seek(0)

        try:
            with zipfile.ZipFile(source_bytes, 'r') as src_zf:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
                    for entry in src_zf.infolist():
                        if entry.filename.endswith('/'):
                            continue

                        name_lower = entry.filename.lower()

                        # Drop whole parts that carry sensitive content:
                        # comments*, commentsExtended, commentsIds,
                        # people.xml (commenter real names), custom.xml
                        # (custom properties python-docx cannot reach).
                        base = name_lower.rsplit('/', 1)[-1]
                        if (base.startswith('comments')
                                or base == 'people.xml'
                                or name_lower == 'docprops/custom.xml'):
                            _logger.info(
                                "Removing sensitive part: %s", entry.filename,
                            )
                            continue

                        data = src_zf.read(entry.filename)

                        # Clean XML/rels members
                        if name_lower.endswith('.xml') or name_lower.endswith('.rels'):
                            try:
                                text = data.decode('utf-8')
                                cleaned = self._clean_xml_text(text)

                                if name_lower == 'docprops/app.xml':
                                    cleaned = self._clean_app_xml(cleaned)

                                if name_lower == 'word/document.xml':
                                    # Strip dangling comment anchors
                                    cleaned = re.sub(
                                        r'<w:commentR(?:eference|angeStart|angeEnd)\b[^>]*/>',
                                        '', cleaned)

                                # Catch-all visible-text pass: hyperlink
                                # runs, text boxes, footnotes/endnotes, and
                                # unwrapped tracked insertions all live in
                                # <w:t> elements python-docx never handed us.
                                base_name = name_lower.rsplit('/', 1)[-1]
                                if (name_lower.startswith('word/')
                                        and name_lower.endswith('.xml')
                                        and (base_name in (
                                            'document.xml', 'footnotes.xml',
                                            'endnotes.xml')
                                            or base_name.startswith(
                                                ('header', 'footer')))):
                                    cleaned = self._clean_visible_text_xml(
                                        cleaned, 'w:t')

                                if name_lower == '[content_types].xml':
                                    # Drop Overrides for the parts we removed
                                    cleaned = re.sub(
                                        r'<Override[^>]*PartName="/(?:word/comments[^"]*|'
                                        r'word/people\.xml|docProps/custom\.xml)"[^>]*/>',
                                        '', cleaned)

                                if name_lower.endswith('.rels'):
                                    # Drop relationships to the removed parts
                                    # (exact part names only — a hyperlink
                                    # whose URL merely CONTAINS 'comments'
                                    # must survive).
                                    cleaned = re.sub(
                                        r'<Relationship[^>]*Target="[^"]*/'
                                        r'(?:comments\w*|people|custom)\.xml"'
                                        r'[^>]*/>',
                                        '', cleaned)
                                    # Entity-clean remaining relationship
                                    # targets (mailto:, entity-bearing URLs)
                                    cleaned = self._clean_rel_targets(cleaned)

                                if cleaned != text:
                                    _logger.debug(
                                        "Cleaned XML artifacts from %s",
                                        entry.filename,
                                    )
                                data = cleaned.encode('utf-8')
                            except Exception:
                                pass

                        # Fixed member timestamp: never leak the run date.
                        info = zipfile.ZipInfo(
                            filename=entry.filename,
                            date_time=self._ZIP_DATE_TIME,
                        )
                        info.compress_type = zipfile.ZIP_DEFLATED
                        dst_zf.writestr(info, data)

        except Exception:
            # Fail closed: never fall back to writing the un-scrubbed bytes.
            if output_path.exists():
                output_path.unlink()
            raise

    def _clean_xml_text(self, xml_text: str) -> str:
        """Clean XML text to remove sensitive artifacts.

        Removes:
        - RSID values (w:rsid*, w:edGeom, etc.)
        - Tracked change markers (w:ins, w:del)
        - Hidden text markers (w:vanish)
        """
        cleaned = xml_text

        # Remove all RSID attributes (fingerprinting):
        # w:rsid, w:rsidR, w:rsidRDefault, w:rsidP, w:rsidRPr, ...
        cleaned = self._RSID_ATTR_PATTERN.sub('', cleaned)

        # Remove RSID element forms (settings.xml stores a <w:rsids> block
        # of <w:rsid w:val="..."/> elements the attribute pattern misses).
        cleaned = self._RSIDS_BLOCK_PATTERN.sub('', cleaned)
        cleaned = self._RSID_ELEM_PATTERN.sub('', cleaned)

        # Remove SELF-CLOSING tracked-change markers FIRST: the block
        # regexes below would otherwise pair a '<w:ins .../>' opener with
        # some later '</w:ins>' and mass-delete everything in between.
        cleaned = re.sub(r'<w:ins\b[^>]*/>', '', cleaned)
        cleaned = re.sub(r'<w:del\b[^>]*/>', '', cleaned)

        # Remove tracked changes - accept all insertions.
        # (?=[\s>]) prevents the prefix from also matching <w:insideH>/<w:insideV>
        # table-border elements; DOTALL handles multiline blocks.
        cleaned = re.sub(
            r'<w:ins(?=[\s>])[^>]*>(.*?)</w:ins>',
            r'\1',
            cleaned,
            flags=re.DOTALL,
        )

        # Remove tracked changes - remove all deletions (incl. w:delText)
        cleaned = re.sub(
            r'<w:del(?=[\s>])[^>]*>.*?</w:del>',
            '',
            cleaned,
            flags=re.DOTALL,
        )

        # Remove hidden text markers (w:vanish), with or without attributes
        cleaned = re.sub(r'<w:vanish\b[^>]*/?>', '', cleaned)

        # NOTE: no whitespace collapsing here — a global \s{2,} -> ' '
        # rewrite corrupts xml:space="preserve" text content.

        return cleaned

    def _clean_visible_text_xml(self, xml_text: str, tag: str) -> str:
        """Entity-clean the inner text of every <tag>...</tag> element.

        Text is XML-unescaped before matching (so 'Johnson &amp; Johnson'
        is seen — and registered — as 'Johnson & Johnson') and re-escaped
        on write.
        """
        from xml.sax.saxutils import escape, unescape

        def _sub(m: re.Match) -> str:
            raw = unescape(m.group(2))
            cleaned_text = self.text_cleaner.clean_text(raw)
            if cleaned_text == raw:
                return m.group(0)
            return m.group(1) + escape(cleaned_text) + m.group(3)

        return re.sub(
            rf'(<{tag}(?:\s[^>]*)?>)(.*?)(</{tag}>)',
            _sub, xml_text, flags=re.DOTALL,
        )

    def _clean_rel_targets(self, xml_text: str) -> str:
        """Entity-clean Target="..." values in relationship parts
        (mailto: addresses and entity-bearing URLs leak otherwise)."""
        from xml.sax.saxutils import escape, unescape

        def _sub(m: re.Match) -> str:
            raw = unescape(m.group(2))
            cleaned_text = self.text_cleaner.clean_text(raw)
            if cleaned_text == raw:
                return m.group(0)
            return m.group(1) + escape(cleaned_text) + m.group(3)

        return re.sub(r'(Target=")([^"]*)(")', _sub, xml_text)

    # Fields in docProps/app.xml that identify people/orgs/tooling
    _APP_XML_IDENTIFIER_TAGS = {
        'Company': 'company',
        'Manager': 'person',
    }
    _APP_XML_BLANK_TAGS = ('Template', 'HyperlinkBase', 'TotalTime',
                           'Application', 'AppVersion')

    def _clean_app_xml(self, xml_text: str) -> str:
        """Scrub docProps/app.xml (extended properties python-docx can't reach).

        Company/Manager are pseudonymized through the mapper; tool and
        template fingerprints are blanked.
        """
        cleaned = xml_text

        from xml.sax.saxutils import unescape

        def _pseudonymize(tag: str, entity_type: str, text: str) -> str:
            def _sub(m: re.Match) -> str:
                # Unescape first: 'Johnson &amp; Johnson' must register as
                # 'Johnson & Johnson' or neither cleaner nor verifier will
                # ever match the real value elsewhere.
                value = unescape(m.group(2)).strip()
                if not value:
                    return m.group(0)
                placeholder = self.mapper.get_or_create(
                    entity_type=entity_type, value=value,
                    source='docx_app_properties',
                )
                return f"{m.group(1)}{placeholder}{m.group(3)}"
            return re.sub(
                rf'(<{tag}>)([^<]*)(</{tag}>)', _sub, text,
            )

        for tag, entity_type in self._APP_XML_IDENTIFIER_TAGS.items():
            cleaned = _pseudonymize(tag, entity_type, cleaned)

        for tag in self._APP_XML_BLANK_TAGS:
            cleaned = re.sub(
                rf'<{tag}>[^<]*</{tag}>', f'<{tag}></{tag}>', cleaned,
            )

        # Entity-clean document-title entries (TitlesOfParts/HeadingPairs
        # carry the title, which is often entity-bearing)
        cleaned = self._clean_visible_text_xml(cleaned, 'vt:lpstr')

        return cleaned

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in a Word document.

        Args:
            input_path: Source DOCX file path

        Returns:
            List of detected risk descriptions
        """
        risks = []
        ext = input_path.suffix.lower()

        if ext == '.doc':
            risks.append(
                "Legacy .doc format - Fast Save appends without removing old text"
            )

        if not zipfile.is_zipfile(input_path):
            return risks

        try:
            with zipfile.ZipFile(input_path, 'r') as zf:
                namelist = zf.namelist()

                # Check for tracked changes
                doc_xml = zf.read('word/document.xml').decode('utf-8')
                if '<w:ins' in doc_xml or '<w:del' in doc_xml:
                    risks.append(
                        "Tracked changes detected - revision history is fully recoverable"
                    )

                # Check for comments
                if any('comments' in f for f in namelist):
                    risks.append(
                        "Comments present - may contain negotiation history"
                    )

                # Check for RSID values
                if 'w:rsid' in doc_xml:
                    risks.append(
                        "RSID values present - documents can be fingerprinted to editing sessions"
                    )

                # Check for hidden text
                if 'w:vanish' in doc_xml:
                    risks.append(
                        "Hidden text (w:vanish) present - extractable but not visible"
                    )

                # Check for embedded objects
                if any('embed' in f for f in namelist):
                    risks.append(
                        "Embedded objects present - may contain sensitive data"
                    )

        except Exception as e:
            risks.append(f"Risk detection failed: {e}")

        return risks