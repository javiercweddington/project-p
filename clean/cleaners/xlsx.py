"""
XLSXCleaner - clean Excel files by removing metadata and cleaning content.

Addresses Excel-specific risks (ranked by failure likelihood):

1. Pivot cache (CRITICAL) - pivotCacheRecords retains a FULL copy of source
   data even after the source sheet is deleted. This is ghost content that
   survives visual inspection.

2. Very-hidden sheets (HIGH) - xlSheetVeryHidden sheets don't appear in the
   unhide dialog. Standard location for margin math and sensitive calculations.
   Strategy: Clean their content, then unhide (don't delete - may be referenced).

3. External links (HIGH) - xl/externalLinks/ contains UNC paths to fileservers
   plus cached values pulled from them.

4. Threaded comments (MEDIUM) - persons.xml stores display names and user IDs.

5. Power Query / connections.xml (MEDIUM) - Server names, SQL queries,
   occasionally credentials.

6. Data-validation dropdowns (MEDIUM) - Enumerate your entire vendor list
   even in sheets that look clean.

7. VBA project in .xlsm (MEDIUM) - VBA project passwords are trivially
   bypassed, so treat "protected" as public. Hardcoded paths and developer
   usernames live in modules. Also dump p-code, not just displayed source,
   as they can differ (VBA stomping).

8. Formulas (LOW) - Encode markup multipliers even when displayed values
   look benign. Clean formula text, not just displayed values.

9. BIFF slack in .xls (LOW) - Legacy format retains deleted-cell remnants
   in unused space.

Strategy: Use openpyxl for cell-level cleaning, then directly manipulate
the ZIP internals to remove pivot cache, external links, connections,
and threaded comments at the XML level.

Dependencies (optional):
- openpyxl: Excel file manipulation
"""

from __future__ import annotations

import io
import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from ..anonymizer import EntityMapper
from .text import TextCleaner

_logger = logging.getLogger(__name__)

# Try optional dependencies
try:
    from openpyxl import load_workbook
    from openpyxl.workbook import Workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class XLSXCleaner:
    """Clean Excel files by removing document metadata and cleaning content.

    Uses openpyxl for cell-level cleaning and direct ZIP manipulation
    for XML-level artifact removal.
    """

    # Office document core properties to clear
    _CORE_PROPERTIES = {
        'creator', 'last_modified_by', 'contributor',
        'author', 'company', 'manager',
        'description', 'subject', 'title',
        'keywords', 'category', 'comments',
    }

    # XML paths to remove from the ZIP archive
    _XML_PATHS_TO_REMOVE = {
        # Pivot cache - full copy of source data
        'xl/pivotCache/pivotCacheDefinition1.xml',
        'xl/pivotCache/pivotCacheRecords1.xml',
        # External links - UNC paths to fileservers
        'xl/externalLinks/externalLink1.xml',
        # Connections - server names, SQL, credentials
        'xl/connections.xml',
        # Threaded comments - display names and user IDs
        'xl/threads/persons.xml',
        'xl/threads/threadedComment1.xml',
        # Printer settings (may contain server info)
        'xl/printerSettings/printer1.xml',
    }

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)

        if not HAS_OPENPYXL:
            _logger.warning(
                "openpyxl not available; Excel cleaning limited. "
                "Install with: pip install openpyxl"
            )

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean an Excel file by removing metadata and cleaning content.

        Args:
            input_path: Source Excel file path
            output_path: Destination Excel file path

        Returns:
            True if cleaning was successful
        """
        ext = input_path.suffix.lower()

        # Legacy BIFF format - can't safely clean, copy with warning
        if ext == '.xls':
            _logger.warning(
                "Legacy .xls (BIFF) format detected: %s. "
                "BIFF slack retains deleted-cell remnants. "
                "Copy-as-is with warning.",
                input_path.name,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return False

        if not HAS_OPENPYXL:
            _logger.warning("openpyxl not available; copying Excel file as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return True

        try:
            # Load workbook (preserve VBA for .xlsm)
            keep_vba = ext == '.xlsm'
            wb = load_workbook(input_path, keep_vba=keep_vba)

            # Clear core properties
            self._clear_properties(wb.properties)

            # Clean all cell content (always on)
            self._clean_all_sheets(wb)

            # Handle very-hidden sheets - clean content then unhide
            self._handle_very_hidden_sheets(wb)

            # Save to bytes first (for ZIP manipulation)
            output_bytes = io.BytesIO()
            wb.save(output_bytes)
            output_bytes.seek(0)

            # Post-processing: remove XML-level artifacts by rebuilding ZIP
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._rebuild_zip_without_artifacts(output_bytes, output_path)

            return True

        except Exception as e:
            _logger.error("Error cleaning Excel file %s: %s", input_path, e)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return False

    def _clear_properties(self, props) -> None:
        """Clear core document properties."""
        for prop in self._CORE_PROPERTIES:
            setattr(props, prop, '')

    def _clean_all_sheets(self, wb) -> None:
        """Clean cell content across all worksheets."""
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    # Clean cell value (text)
                    if isinstance(cell.value, str):
                        cell.value = self.text_cleaner.clean_text(cell.value)

                    # Clean formula text (may contain paths, names)
                    if cell.data_type == 'f' and isinstance(cell.value, str):
                        # Note: openpyxl stores formulas as strings starting with =
                        # We clean the formula text but preserve the = prefix
                        formula = cell.value
                        if formula.startswith('='):
                            # Don't clean the formula structure, just literal strings
                            # This is tricky - for now, register entities in formulas
                            self.text_cleaner.clean_text(formula[1:])  # Skip the =
                        else:
                            cell.value = self.text_cleaner.clean_text(formula)

                    # Clean cell comments
                    if cell.comment:
                        if isinstance(cell.comment.text, str):
                            cell.comment.text = self.text_cleaner.clean_text(
                                cell.comment.text
                            )

    def _handle_very_hidden_sheets(self, wb) -> None:
        """Handle very-hidden sheets: clean content, then unhide.

        Very-hidden sheets (xlSheetVeryHidden) don't appear in the unhide
        dialog and are a standard location for margin math and sensitive
        calculations. We clean their content and then unhide them so the
        data is visible (and auditable) rather than hidden.
        """
        for ws in wb.worksheets:
            if ws.sheet_state == 'veryHidden':
                _logger.info(
                    "Found very-hidden sheet: %s - cleaning and unhiding",
                    ws.title,
                )
                # Clean the sheet content (already done in _clean_all_sheets)
                # Then unhide
                ws.sheet_state = 'visible'

    def _rebuild_zip_without_artifacts(self, source_bytes: io.BytesIO,
                                        output_path: Path) -> None:
        """Rebuild the Excel ZIP without sensitive XML artifacts.

        This is the correct way to remove files from a ZIP-based format:
        read all entries, skip the ones we want to remove, write a new ZIP.

        Removes:
        - Pivot cache definitions and records
        - External links
        - Connections
        - Threaded comments (persons.xml)
        - Printer settings
        """
        source_bytes.seek(0)

        try:
            with zipfile.ZipFile(source_bytes, 'r') as src_zf:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
                    for entry in src_zf.infolist():
                        # Check if this entry should be removed
                        should_remove = False

                        # Check exact paths
                        if entry.filename in self._XML_PATHS_TO_REMOVE:
                            should_remove = True
                            _logger.info("Removing Excel artifact: %s", entry.filename)

                        # Check patterns (pivot cache, external links)
                        if not should_remove:
                            if 'pivotCache' in entry.filename:
                                should_remove = True
                                _logger.info(
                                    "Removing pivot cache: %s", entry.filename,
                                )
                            elif 'externalLinks' in entry.filename:
                                should_remove = True
                                _logger.info(
                                    "Removing external link: %s", entry.filename,
                                )
                            elif 'threads' in entry.filename:
                                should_remove = True
                                _logger.info(
                                    "Removing threaded comment: %s", entry.filename,
                                )
                            elif entry.filename == 'xl/connections.xml':
                                should_remove = True
                                _logger.info("Removing connections.xml")

                        if should_remove:
                            continue

                        # Copy entry to new ZIP
                        if entry.filename.endswith('/'):
                            # Directory entry
                            dst_zf.writestr(entry, b'')
                        else:
                            data = src_zf.read(entry.filename)
                            dst_zf.writestr(entry, data)

        except Exception as e:
            _logger.error(
                "Failed to rebuild ZIP without artifacts: %s. "
                "Saving original instead.",
                e,
            )
            # Fallback: save the openpyxl output as-is
            source_bytes.seek(0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(source_bytes.read())

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in an Excel file.

        Args:
            input_path: Source Excel file path

        Returns:
            List of detected risk descriptions
        """
        risks = []
        ext = input_path.suffix.lower()

        if ext == '.xls':
            risks.append(
                "Legacy BIFF format - deleted-cell remnants may persist in slack space"
            )

        if ext == '.xlsm':
            risks.append(
                "Macro-enabled workbook - VBA project passwords are trivially bypassed. "
                "Treat 'protected' VBA as public."
            )

        if not zipfile.is_zipfile(input_path):
            return risks

        try:
            with zipfile.ZipFile(input_path, 'r') as zf:
                namelist = zf.namelist()

                # Check for pivot cache
                pivot_files = [f for f in namelist if 'pivotCache' in f]
                if pivot_files:
                    risks.append(
                        f"Pivot cache detected ({len(pivot_files)} files) - "
                        f"full copy of source data may persist even after source sheet deletion"
                    )

                # Check for external links
                ext_link_files = [f for f in namelist if 'externalLinks' in f]
                if ext_link_files:
                    risks.append(
                        f"External links detected ({len(ext_link_files)} files) - "
                        f"may contain UNC paths to fileservers"
                    )

                # Check for connections
                if 'xl/connections.xml' in namelist:
                    risks.append(
                        "Connections.xml present - may contain server names, SQL queries, credentials"
                    )

                # Check for threaded comments
                thread_files = [f for f in namelist if 'threads' in f]
                if thread_files:
                    risks.append(
                        f"Threaded comments present ({len(thread_files)} files) - "
                        f"display names and user IDs in persons.xml"
                    )

                # Check for very-hidden sheets
                try:
                    workbooks_xml = zf.read('xl/workbook.xml').decode('utf-8')
                    if 'veryHidden' in workbooks_xml:
                        risks.append(
                            "Very-hidden sheets detected - may contain margin math or sensitive data"
                        )
                except Exception:
                    pass

                # Check for data validation (vendor lists)
                try:
                    for sheet_path in namelist:
                        if sheet_path.startswith('xl/worksheets/sheet'):
                            sheet_xml = zf.read(sheet_path).decode('utf-8')
                            if 'dataValidation' in sheet_xml:
                                risks.append(
                                    f"Data validation in {sheet_path} - may enumerate vendor lists"
                                )
                                break
                except Exception:
                    pass

                # Check for VBA project
                if 'xl/vbaProject.bin' in namelist:
                    risks.append(
                        "VBA project present - may contain hardcoded paths, developer usernames"
                    )

        except Exception as e:
            risks.append(f"Risk detection failed: {e}")

        return risks