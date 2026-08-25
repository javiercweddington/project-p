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
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
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


# Fixed ZIP timestamp (MS-DOS epoch: 1980-01-01 00:00:00 UTC)
# Normalizes all zip member dates so the run date doesn't leak.
_FIXED_ZIP_DATE_TIME = (2024, 1, 1, 0, 0, 0)


class XLSXCleaner:
    """Clean Excel files by removing document metadata and cleaning content.

    Uses openpyxl for cell-level cleaning and direct ZIP manipulation
    for XML-level artifact removal.
    """

    # openpyxl core properties that contain identifier-like values
    # and should go through the mapper instead of being blanked.
    # Keys map to (property_name, entity_type) tuples.
    _IDENTIFIER_PROPERTIES = {
        'creator': ('creator', 'person'),
        'last_modified_by': ('last_modified_by', 'person'),
        'contributor': ('contributor', 'person'),
        'company': ('company', 'company'),
    }

    # openpyxl core properties that should be blanked (non-identifiers)
    _BLANK_PROPERTIES = {
        'description', 'subject', 'keywords', 'category', 'comments',
        'title',
    }

    # Fixed timestamp for embedded document timestamps
    _FIXED_TIMESTAMP = "2024-01-01T00:00:00Z"

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
        # VBA project - hardcoded paths, developer usernames
        'xl/vbaProject.bin',
    }

    # OPC namespaces
    CT_NS = '{http://schemas.openxmlformats.org/package/2006/content-types}'
    REL_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

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

        # Legacy BIFF format - can't safely clean, remove staged file
        if ext == '.xls':
            _logger.warning(
                "Legacy .xls (BIFF) format detected: %s. "
                "BIFF slack retains deleted-cell remnants. "
                "Removing staged file (fail-closed).",
                input_path.name,
            )
            if output_path.exists():
                os.remove(output_path)
            return False

        if not HAS_OPENPYXL:
            _logger.warning("openpyxl not available; removing Excel file (fail-closed)")
            if output_path.exists():
                os.remove(output_path)
            return False

        try:
            # Load workbook (do NOT preserve VBA - we drop vbaProject.bin)
            wb = load_workbook(input_path, keep_vba=False)

            # Clear core properties
            self._clear_properties(wb.properties)

            # Clean sheet names (may contain company names)
            self._clean_sheet_names(wb)

            # Clean all cell content (bounded iteration)
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
            if output_path.exists():
                os.remove(output_path)
            return False

    def _clear_properties(self, props) -> None:
        """Anonymize core document properties.

        Identifier properties (creator, company, etc.) go through the mapper
        to generate consistent placeholders like [PERSON_002].
        Non-identifier properties are blanked.
        Embedded timestamps are normalized to a fixed epoch.
        """
        # Anonymize identifier properties through the mapper
        for prop_name, (_, entity_type) in self._IDENTIFIER_PROPERTIES.items():
            try:
                value = getattr(props, prop_name, None)
                if value and str(value).strip():
                    placeholder = self.mapper.get_or_create(
                        entity_type=entity_type,
                        value=str(value).strip(),
                        source='xlsx_core_properties',
                    )
                    setattr(props, prop_name, placeholder)
                else:
                    setattr(props, prop_name, '')
            except (AttributeError, TypeError):
                pass

        # Blank non-identifier properties
        for prop_name in self._BLANK_PROPERTIES:
            try:
                setattr(props, prop_name, '')
            except (AttributeError, TypeError):
                pass

        # Normalize embedded timestamps to fixed epoch
        for prop_name in ('created', 'modified', 'lastPrinted'):
            try:
                setattr(props, prop_name, self._FIXED_TIMESTAMP)
            except (AttributeError, TypeError):
                pass

    def _clean_sheet_names(self, wb) -> None:
        """Clean sheet names and defined names to remove identifiers.

        Sheet names, defined names, and table names can contain company names
        or other identifiers that survive cell-level cleaning.
        """
        # Clean sheet names
        for ws in wb.worksheets:
            if ws.title:
                cleaned = self.text_cleaner.clean_text(ws.title)
                # Excel sheet names have max 31 chars and can't contain :\\/?*[]
                cleaned = re.sub(r'[:\\/?*\[\]]', '', cleaned)
                cleaned = cleaned[:31]
                if cleaned != ws.title:
                    ws.title = cleaned or ws.title  # Don't allow empty

        # Clean defined names (named ranges that may contain paths/names)
        if wb.defined_names:
            for name in list(wb.defined_names.definedNameFinder.keys()):
                # Clean the defined name itself
                cleaned_name = self.text_cleaner.clean_text(name)
                if cleaned_name != name:
                    # Get the value and re-register under cleaned name
                    old_val = wb.defined_names[name]
                    del wb.defined_names[name]
                    wb.defined_names[cleaned_name] = old_val

    def _clean_all_sheets(self, wb) -> None:
        """Clean cell content across all worksheets with bounded iteration.

        Uses iter_rows with explicit min_row/max_row/max_col bounds to avoid
        materializing millions of empty cells. Unbounded iter_rows() on a sheet
        with a large max_row/max_col will allocate cells for every intersection,
        inflating file size dramatically (e.g. 1.58→4.38MB, sheet5 23MB).
        """
        for ws in wb.worksheets:
            if ws.max_row is None or ws.max_column is None:
                continue

            # Bound iteration to actual data dimensions
            for row in ws.iter_rows(
                min_row=1,
                max_row=ws.max_row,
                min_col=1,
                max_col=ws.max_column,
            ):
                for cell in row:
                    # Skip cells that have never been written to
                    if cell.value is None:
                        continue

                    # Clean cell value (text)
                    if isinstance(cell.value, str):
                        cell.value = self.text_cleaner.clean_text(cell.value)

                    # Clean formula text (may contain paths, names)
                    if cell.data_type == 'f' and isinstance(cell.value, str):
                        formula = cell.value
                        if formula.startswith('='):
                            # Clean literal string arguments within formulas
                            # This is a best-effort; full formula parsing is complex
                            cleaned_formula = self._clean_formula_text(formula)
                            cell.value = cleaned_formula
                        else:
                            cell.value = self.text_cleaner.clean_text(formula)

                    # Clean cell comments
                    if cell.comment:
                        if isinstance(cell.comment.text, str):
                            cell.comment.text = self.text_cleaner.clean_text(
                                cell.comment.text
                            )

    def _clean_formula_text(self, formula: str) -> str:
        """Clean text within a formula, preserving formula structure.

        Attempts to clean string literals and sheet references within formulas
        while preserving the formula syntax.
        """
        if not formula.startswith('='):
            return self.text_cleaner.clean_text(formula)

        # Clean the part after the = sign, being careful with quoted strings
        result = '='
        rest = formula[1:]

        # Clean sheet name references (e.g., 'Sheet Name!A1')
        rest = re.sub(
            r"'([^']*?)'",
            lambda m: "'" + self.text_cleaner.clean_text(m.group(1)) + "'",
            rest,
        )

        # Clean string literals in formulas (e.g., VLOOKUP("text", ...))
        rest = re.sub(
            r'"([^"]*?)"',
            lambda m: '"' + self.text_cleaner.clean_text(m.group(1)) + '"',
            rest,
        )

        return result + rest

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

        Also patches [Content_Types].xml and workbook.xml.rels to maintain
        OPC validity after removing external links and other artifacts.

        Removes:
        - Pivot cache definitions and records
        - External links (with rels patching)
        - Connections
        - Threaded comments (persons.xml)
        - Printer settings
        - VBA project (vbaProject.bin)
        - Normalizes zip member timestamps
        """
        source_bytes.seek(0)

        try:
            # Track which artifact types we remove so we can patch manifests
            removed_external_link_ids = set()
            removed_pivot_cache_defs = set()
            removed_vba = False

            with zipfile.ZipFile(source_bytes, 'r') as src_zf:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
                    for entry in src_zf.infolist():
                        # Check if this entry should be removed
                        should_remove = False

                        # Check exact paths
                        if entry.filename in self._XML_PATHS_TO_REMOVE:
                            should_remove = True
                            _logger.info("Removing Excel artifact: %s", entry.filename)
                            if entry.filename == 'xl/vbaProject.bin':
                                removed_vba = True

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

                        # Copy entry to new ZIP with normalized timestamp
                        if entry.filename.endswith('/'):
                            # Directory entry
                            new_entry = zipfile.ZipInfo(filename=entry.filename)
                            new_entry.date_time = _FIXED_ZIP_DATE_TIME
                            new_entry.compress_type = zipfile.ZIP_DEFLATED
                            dst_zf.writestr(new_entry, b'')
                        else:
                            data = src_zf.read(entry.filename)

                            # Patch [Content_Types].xml to remove orphaned entries
                            if entry.filename == '[Content_Types].xml':
                                data = self._patch_content_types(
                                    data, removed_vba,
                                )

                            # Patch workbook.xml.rels to remove external link refs
                            if entry.filename == 'xl/_rels/workbook.xml.rels':
                                data = self._patch_workbook_rels(data)

                            # Normalize zip timestamp
                            new_entry = zipfile.ZipInfo(filename=entry.filename)
                            new_entry.date_time = _FIXED_ZIP_DATE_TIME
                            new_entry.compress_type = zipfile.ZIP_DEFLATED
                            new_entry.external_attr = entry.external_attr
                            dst_zf.writestr(new_entry, data)

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

    def _patch_content_types(self, data: bytes, removed_vba: bool) -> bytes:
        """Patch [Content_Types].xml to remove orphaned content type entries.

        When we remove vbaProject.bin or other artifacts, we must also remove
        the corresponding <Override> entries from [Content_Types].xml, otherwise
        the output is OPC-invalid.
        """
        try:
            tree = ET.fromstring(data)

            if removed_vba:
                # Remove VBA content type overrides
                for override in list(tree.findall(f'{self.CT_NS}:Override')):
                    part_name = override.get('PartName')
                    content_type = override.get('ContentType', '')
                    if (part_name and 'vbaProject' in part_name) or \
                       'MacroEnabled' in content_type:
                        tree.remove(override)
                        _logger.info(
                            "Patched [Content_Types].xml: removed %s", part_name,
                        )

            return ET.tostring(tree, encoding='utf-8', xml_declaration=True)
        except ET.ParseError as e:
            _logger.warning("Failed to parse [Content_Types].xml: %s", e)
            return data

    def _patch_workbook_rels(self, data: bytes) -> bytes:
        """Patch workbook.xml.rels to remove external link relationships.

        Removing externalLinks XML without patching workbook.xml.rels makes
        the output OPC-invalid (Excel repair prompt).
        """
        try:
            tree = ET.fromstring(data)
            removed_count = 0

            for rel in list(tree.findall(f'{self.REL_NS}:Relationship')):
                target = rel.get('Target', '')
                type_ = rel.get('Type', '')
                if ('externalLink' in type_.lower()) or \
                   ('externalLink' in target.lower()):
                    tree.remove(rel)
                    removed_count += 1
                    _logger.info(
                        "Patched workbook.xml.rels: removed external link rel",
                    )

            if removed_count:
                return ET.tostring(tree, encoding='utf-8', xml_declaration=True)
            return data
        except ET.ParseError as e:
            _logger.warning("Failed to parse workbook.xml.rels: %s", e)
            return data

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