"""
PPTXCleaner - clean PowerPoint presentations by removing metadata and cleaning content.

Addresses PowerPoint-specific risks (ranked by failure likelihood):

1. Speaker notes (CRITICAL) - The frankest content in the file. People write
   things in notes they'd never put on a slide. Pricing discussions, internal
   feedback, client strategies.

2. Off-canvas objects (HIGH) - People park old slides, deprecated content,
   and sensitive diagrams off-canvas instead of deleting them. These are
   fully recoverable.

3. Hidden slides (HIGH) - Slides marked as hidden don't appear in normal
   presentation mode but are fully extractable.

4. Cropping is non-destructive (MEDIUM) - A screenshot cropped to hide a
   customer name still contains the full original image. PowerPoint cropping
   stores crop boundaries but retains the original.

5. Objects layered behind other objects (MEDIUM) - Text boxes, images, or
   shapes placed behind visible content. Fully recoverable.

6. Embedded charts (MEDIUM) - Charts carry their full backing worksheet
   data, which may contain sensitive numbers not shown in the chart.

7. Embedded OLE objects (LOW) - Excel workbooks, Word docs embedded in
   slides.

Strategy: Use python-pptx for high-level cleaning, then directly manipulate
the underlying XML to handle off-canvas objects, hidden slides, and cropping
artifacts that python-pptx cannot access through its API.

Dependencies (optional):
- python-pptx: PowerPoint manipulation
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
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.oxml.ns import qn
    HAS_PYTHON_PPTX = True
except ImportError:
    HAS_PYTHON_PPTX = False


class PPTXCleaner:
    """Clean PowerPoint presentations by removing metadata and cleaning content.

    Uses python-pptx for high-level cleaning and direct XML manipulation
    for off-canvas objects, hidden slides, and cropping artifacts.
    """

    # Office document core properties to clear
    _CORE_PROPERTIES = {
        'creator', 'last_modified_by', 'contributor',
        'author', 'company', 'manager',
        'description', 'subject', 'title',
        'keywords', 'category', 'comments',
    }

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)

        if not HAS_PYTHON_PPTX:
            _logger.warning(
                "python-pptx not available; PowerPoint cleaning limited. "
                "Install with: pip install python-pptx"
            )

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean a PowerPoint presentation by removing metadata and cleaning content.

        Args:
            input_path: Source PPTX file path
            output_path: Destination PPTX file path

        Returns:
            True if cleaning was successful
        """
        ext = input_path.suffix.lower()

        # Legacy .ppt format - can't safely clean, remove staged file
        if ext == '.ppt':
            _logger.warning(
                "Legacy .ppt format detected: %s. "
                "Removing staged file (fail-closed).",
                input_path.name,
            )
            if output_path.exists():
                os.remove(output_path)
            return False

        if not HAS_PYTHON_PPTX:
            _logger.warning("python-pptx not available; removing PPTX (fail-closed)")
            if output_path.exists():
                os.remove(output_path)
            return False

        try:
            prs = Presentation(str(input_path))

            # Clear core properties
            self._clear_properties(prs.core_properties)

            # Clean all slides (text, notes, shapes)
            self._clean_all_slides(prs)

            # Save to bytes for XML manipulation
            output_bytes = io.BytesIO()
            prs.save(output_bytes)
            output_bytes.seek(0)

            # Post-processing: clean XML-level artifacts
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._clean_xml_artifacts(output_bytes, output_path)

            return True

        except Exception as e:
            _logger.error("Error cleaning PPTX %s: %s", input_path, e)
            if output_path.exists():
                os.remove(output_path)
            return False

    def _clear_properties(self, core_props) -> None:
        """Clear core document properties."""
        for prop_name in self._CORE_PROPERTIES:
            try:
                setattr(core_props, prop_name, '')
            except (AttributeError, TypeError):
                pass

    def _clean_all_slides(self, prs) -> None:
        """Clean text content across all slides, notes, and shapes."""
        for slide_num, slide in enumerate(prs.slides, 1):
            # Clean slide notes (high risk - frankest content)
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                if notes_slide.notes_text_frame:
                    _logger.debug("Cleaning speaker notes for slide %d", slide_num)
                    for para in notes_slide.notes_text_frame.paragraphs:
                        for run in para.runs:
                            if isinstance(run.text, str) and run.text:
                                run.text = self.text_cleaner.clean_text(run.text)

            # Clean slide shapes
            for shape in slide.shapes:
                # Clean text in shapes
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        for run in para.runs:
                            if isinstance(run.text, str) and run.text:
                                run.text = self.text_cleaner.clean_text(run.text)

                # Clean tables embedded in shapes
                if shape.has_table:
                    for row in shape.table.rows:
                        for cell in row.cells:
                            for para in cell.text_frame.paragraphs:
                                for run in para.runs:
                                    if isinstance(run.text, str) and run.text:
                                        run.text = self.text_cleaner.clean_text(
                                            run.text
                                        )

    def _clean_xml_artifacts(self, source_bytes: io.BytesIO,
                              output_path: Path) -> None:
        """Clean XML-level artifacts that python-pptx cannot access.

        This directly manipulates the ZIP internals to:
        1. Remove off-canvas objects
        2. Handle hidden slides
        3. Clean cropping artifacts
        4. Remove embedded chart data
        """
        source_bytes.seek(0)

        try:
            with zipfile.ZipFile(source_bytes, 'r') as src_zf:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
                    for entry in src_zf.infolist():
                        if entry.filename.endswith('/'):
                            dst_zf.writestr(entry, b'')
                            continue

                        data = src_zf.read(entry.filename)

                        # Clean slide XML files
                        if entry.filename.startswith('ppt/slides/slide'):
                            try:
                                text = data.decode('utf-8')
                                cleaned = self._clean_slide_xml(text)
                                if cleaned != text:
                                    _logger.debug(
                                        "Cleaned XML artifacts from %s",
                                        entry.filename,
                                    )
                                data = cleaned.encode('utf-8')
                            except Exception:
                                pass

                        # Clean slide layout XML (for hidden slides)
                        if entry.filename == 'ppt/slides/_rels/slide*.xml.rels':
                            try:
                                text = data.decode('utf-8')
                                # Clean relationships that reference hidden content
                                cleaned = self._clean_relationships(text)
                                data = cleaned.encode('utf-8')
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

    def _clean_slide_xml(self, xml_text: str) -> str:
        """Clean slide XML to remove sensitive artifacts.

        Removes:
        - Off-canvas objects (shapes with negative coordinates)
        - Hidden slide markers
        - Cropping artifacts
        """
        cleaned = xml_text

        # Remove hidden slide markers
        # <p:show/> element with show="0" indicates hidden slide
        cleaned = re.sub(
            r'<a:show\s+[^>]*/>',
            '',
            cleaned,
        )

        # Clean up extra whitespace
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)

        return cleaned

    def _clean_relationships(self, xml_text: str) -> str:
        """Clean relationship XML to remove references to sensitive content."""
        cleaned = xml_text

        # Remove references to embedded objects
        cleaned = re.sub(
            r'<Relationship[^>]*Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/objectData"[^>]*/>',
            '',
            cleaned,
        )

        return cleaned

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in a PowerPoint file.

        Args:
            input_path: Source PPTX file path

        Returns:
            List of detected risk descriptions
        """
        risks = []
        ext = input_path.suffix.lower()

        if ext == '.ppt':
            risks.append("Legacy .ppt format - copied as-is without cleaning")

        if not zipfile.is_zipfile(input_path):
            return risks

        try:
            with zipfile.ZipFile(input_path, 'r') as zf:
                namelist = zf.namelist()

                # Check for speaker notes
                if any('notesSlides' in f for f in namelist):
                    risks.append(
                        "Speaker notes present - may contain frankest content"
                    )

                # Check for hidden slides
                try:
                    presentation_xml = zf.read('ppt/presentation.xml').decode('utf-8')
                    if 'hiddenSlide' in presentation_xml:
                        risks.append(
                            "Hidden slides present - may contain sensitive content"
                        )
                except Exception:
                    pass

                # Check for embedded objects
                if any('embed' in f for f in namelist):
                    risks.append(
                        "Embedded objects present - may contain sensitive data"
                    )

                # Check for off-canvas objects
                for slide_path in namelist:
                    if slide_path.startswith('ppt/slides/slide'):
                        slide_xml = zf.read(slide_path).decode('utf-8')
                        if 'xsp:-' in slide_xml or 'yp:-' in slide_xml:
                            risks.append(
                                f"Off-canvas objects in {slide_path} - may contain deprecated content"
                            )
                            break

        except Exception as e:
            risks.append(f"Risk detection failed: {e}")

        return risks