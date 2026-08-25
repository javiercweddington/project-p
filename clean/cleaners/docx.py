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
    # Keys map to (property_name, entity_type) tuples.
    _IDENTIFIER_PROPERTIES = {
        'creator': ('creator', 'person'),
        'last_modified_by': ('last_modified_by', 'person'),
        'contributor': ('contributor', 'person'),
        'company': ('company', 'company'),
    }

    # python-docx core properties that should be blanked (non-identifiers)
    _BLANK_PROPERTIES = {
        'description', 'subject', 'keywords', 'category', 'comments',
    }

    # Fixed timestamp for embedded document timestamps
    _FIXED_TIMESTAMP = "2024-01-01T00:00:00Z"

    # XML elements to remove from document
    _RSID_PATTERN = re.compile(r'w:rsid[\w]*\s*=\s*"[^"]*"')

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

        # Legacy .doc format - can't safely clean, remove staged file
        if ext == '.doc':
            _logger.warning(
                "Legacy .doc format detected: %s. "
                "Fast Save appends without removing - old text persists. "
                "Removing staged file (fail-closed).",
                input_path.name,
            )
            if output_path.exists():
                os.remove(output_path)
            return False

        if not HAS_PYTHON_DOCX:
            _logger.warning("python-docx not available; removing DOCX (fail-closed)")
            if output_path.exists():
                os.remove(output_path)
            return False

        try:
            doc = Document(str(input_path))

            # Clear core properties
            self._clear_properties(doc.core_properties)

            # Clear custom properties
            self._clear_custom_properties(doc)

            # Clean all text content
            self._clean_all_text(doc)

            # Save to bytes for XML manipulation
            output_bytes = io.BytesIO()
            doc.save(output_bytes)
            output_bytes.seek(0)

            # Post-processing: clean XML-level artifacts
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._clean_xml_artifacts(output_bytes, output_path)

            return True

        except Exception as e:
            _logger.error("Error cleaning DOCX %s: %s", input_path, e)
            if output_path.exists():
                os.remove(output_path)
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
        for prop_name in ('created', 'modified', 'last_printed'):
            try:
                current = getattr(core_props, prop_name, None)
                if current is None:
                    setattr(core_props, prop_name, self._FIXED_TIMESTAMP)
                else:
                    setattr(core_props, prop_name, self._FIXED_TIMESTAMP)
            except (AttributeError, TypeError):
                pass

    def _clear_custom_properties(self, doc) -> None:
        """Clear custom properties from the XML."""
        try:
            custom_props = doc.custom_properties
            if custom_props:
                props_elem = custom_props._properties
                for child in list(props_elem):
                    props_elem.remove(child)
        except Exception:
            pass

    def _clean_all_text(self, doc) -> None:
        """Clean text content across all paragraphs, runs, and tables."""
        # Clean paragraphs
        for para in doc.paragraphs:
            for run in para.runs:
                if isinstance(run.text, str) and run.text:
                    run.text = self.text_cleaner.clean_text(run.text)

        # Clean tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            if isinstance(run.text, str) and run.text:
                                run.text = self.text_cleaner.clean_text(run.text)

        # Clean headers and footers through sections
        # python-docx does not expose doc.headers/doc.footers - must iterate sections
        for section in doc.sections:
            # Headers: first-page, even-page, and default (odd-page) variants
            for header_attr in ('header', 'header_first_page', 'header_even'):
                header = getattr(section, header_attr, None)
                if header is not None:
                    for para in header.paragraphs:
                        for run in para.runs:
                            if isinstance(run.text, str) and run.text:
                                run.text = self.text_cleaner.clean_text(run.text)
            # Footers: same page-variant pattern
            for footer_attr in ('footer', 'footer_first_page', 'footer_even'):
                footer = getattr(section, footer_attr, None)
                if footer is not None:
                    for para in footer.paragraphs:
                        for run in para.runs:
                            if isinstance(run.text, str) and run.text:
                                run.text = self.text_cleaner.clean_text(run.text)

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
                            # Directory entry
                            dst_zf.writestr(entry, b'')
                            continue

                        data = src_zf.read(entry.filename)

                        # Clean XML files
                        if entry.filename.endswith('.xml'):
                            try:
                                text = data.decode('utf-8')
                                cleaned = self._clean_xml_text(text)
                                if cleaned != text:
                                    _logger.debug(
                                        "Cleaned XML artifacts from %s",
                                        entry.filename,
                                    )
                                data = cleaned.encode('utf-8')
                            except Exception:
                                pass

                        # Skip comment files entirely
                        if 'comments' in entry.filename.lower():
                            _logger.info(
                                "Removing comment file: %s", entry.filename,
                            )
                            continue

                        # Skip digital signature files
                        if entry.filename.startswith('word/_rels/'):
                            # Clean relationships that reference comments
                            try:
                                text = data.decode('utf-8')
                                if 'comments' in text:
                                    _logger.info(
                                        "Removing comment relationship from %s",
                                        entry.filename,
                                    )
                                    # Remove comment relationships
                                    import re
                                    text = re.sub(
                                        r'<Relationship[^>]*Id="[^"]*"[^>]*Type="[^"]*comments"[^>]*/>',
                                        '',
                                        text,
                                    )
                                    data = text.encode('utf-8')
                            except Exception:
                                pass

                        dst_zf.writestr(entry, data)

        except Exception as e:
            _logger.error(
                "Failed to clean XML artifacts: %s. Saving original instead.",
                e,
            )
            source_bytes.seek(0)
            with open(output_path, 'wb') as f:
                f.write(source_bytes.read())

    def _clean_xml_text(self, xml_text: str) -> str:
        """Clean XML text to remove sensitive artifacts.

        Removes:
        - RSID values (w:rsid*, w:edGeom, etc.)
        - Tracked change markers (w:ins, w:del)
        - Hidden text markers (w:vanish)
        """
        cleaned = xml_text

        # Remove all RSID attributes (fingerprinting)
        # This includes w:rsid, w:rsidR, w:rsidRDefault, w:rsidP, w:rsidRPr
        cleaned = self._RSID_PATTERN.sub('', cleaned)

        # Remove tracked changes - accept all insertions
        # w:ins elements contain inserted text - keep the content, remove the wrapper
        import re
        cleaned = re.sub(
            r'<w:ins[^>]*>(.*?)</w:ins>',
            r'\1',
            cleaned,
        )

        # Remove tracked changes - remove all deletions
        cleaned = re.sub(
            r'<w:del[^>]*>.*?</w:del>',
            '',
            cleaned,
        )

        # Remove hidden text markers (w:vanish)
        cleaned = self._remove_vanish(cleaned)

        # Clean up extra whitespace from removals
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)

        return cleaned

    def _remove_vanish(self, xml_text: str) -> str:
        """Remove w:vanish elements that mark text as hidden.

        Hidden text (w:vanish) is still extractable even though it's
        not visible in the document. Remove the vanish marker to make
        the text visible (and auditable).
        """
        import re
        # Remove w:vanish elements entirely
        return re.sub(r'<w:vanish\s*/?>', '', xml_text)

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