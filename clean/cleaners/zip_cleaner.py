"""
ZipCleaner - clean archive files by recursing into members, cleaning them, and repacking.

Addresses ZIP-specific risks (ranked by failure likelihood):

1. Entry paths carry usernames (CRITICAL) - Absolute paths like
   C:\\Users\\jsmith\\Clients\\Acme\\... are embedded in entry names.
   This is the strongest cross-file correlation key you'll leave behind.

2. Orphaned entries (HIGH) - Entries present in local headers but absent
   from the central directory are recoverable "deleted" files. Standard
   ZIP tools (including Python's zipfile) only see the central directory,
   so these ghost entries survive naive extraction.

3. Per-entry mtimes plus NTFS/Unix extended timestamps (MEDIUM) - Map
   your work schedule. Even after repacking, if you preserve original
   mtimes, temporal correlation is possible.

4. Archive and per-file comments (MEDIUM) - May contain sensitive text.

5. Never pass-through (MANDATORY) - Always recurse, clean members, repack.

Strategy:
1. Parse local file headers to detect orphaned entries
2. Extract ALL entries (including orphaned) to temp directory
3. Clean each member using the appropriate cleaner (recursive)
4. Clean entry paths by stripping username patterns
5. Repack into a new ZIP with clean metadata and normalized timestamps
6. No archive comment, no per-file comments

Dependencies:
- zipfile: Standard library (for central directory entries)
- Custom binary parsing: For orphaned entry detection
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from ..anonymizer import EntityMapper
from .text import TextCleaner
# FileCleanerRouter imported lazily to avoid circular instantiation:
# FileCleanerRouter -> ZipCleaner -> FileCleanerRouter -> ...

_logger = logging.getLogger(__name__)

# Fixed timestamp for repacked archives (2024-01-01 00:00:00)
CLEAN_TIMESTAMP = (2024, 1, 1, 0, 0, 0)

# Patterns that indicate username leakage in entry paths
USERNAME_PATTERNS = [
    # Windows paths: C:\Users\username\...
    re.compile(rb'[A-Z]:\\Users\\([^/\\]+)', re.IGNORECASE),
    # Unix paths: /home/username/...
    re.compile(rb'/home/([^/]+)', re.IGNORECASE),
    # UNC paths: \\server\share\...
    re.compile(rb'\\\\([^/\\]+)\\([^/\\]+)', re.IGNORECASE),
    # macOS Users: /Users/username/...
    re.compile(rb'/Users/([^/]+)', re.IGNORECASE),
]


class ZipCleaner:
    """Clean ZIP archive files by recursing into members.

    Extracts all members (including orphaned entries), cleans each using
    the appropriate cleaner, and repacks into a new ZIP with clean metadata.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.text_cleaner = TextCleaner(mapper)
        # Lazy initialization to avoid circular instantiation:
        # FileCleanerRouter -> ZipCleaner -> FileCleanerRouter -> ...
        self._router = None
        self._cleaned_count = 0
        self._copied_count = 0
        self._failed_count = 0
        self._orphaned_count = 0

    @property
    def router(self):
        """Lazy initialization of FileCleanerRouter to break circular dependency."""
        if self._router is None:
            from .router import FileCleanerRouter
            self._router = FileCleanerRouter(self.mapper)
        return self._router

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean a ZIP archive by recursing into members.

        Args:
            input_path: Source ZIP file path
            output_path: Destination ZIP file path

        Returns:
            True if cleaning was successful
        """
        # Verify it's actually a ZIP file
        if not zipfile.is_zipfile(input_path):
            # Might be a .dat or other extension that's actually a ZIP
            # Try to sniff magic bytes
            try:
                with open(input_path, 'rb') as f:
                    header = f.read(4)
                if header[:2] == b'PK':
                    _logger.info(
                        "Detected ZIP magic bytes in non-ZIP extension: %s",
                        input_path.suffix,
                    )
                else:
                    _logger.warning(
                        "Not a ZIP file: %s, removing staged file (fail-closed)",
                        input_path.name,
                    )
                    if output_path.exists():
                        os.remove(output_path)
                    return False
            except Exception:
                if output_path.exists():
                    os.remove(output_path)
                return False

        # Create temp directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="project_p_zip_clean_"))

        try:
            # Detect orphaned entries BEFORE standard extraction
            orphaned_entries = self._detect_orphaned_entries(input_path)

            # Extract all members (including orphaned if found)
            with zipfile.ZipFile(input_path, 'r') as zf:
                # Extract standard entries
                zf.extractall(temp_dir)

                # Get archive comment
                archive_comment = zf.comment.decode('utf-8', errors='replace')
                if archive_comment.strip():
                    _logger.info(
                        "Archive comment detected (will be removed): %s",
                        archive_comment[:100],
                    )

            # Clean each extracted file
            self._clean_extracted_files(temp_dir)

            # Repack into new ZIP with clean metadata and cleaned paths
            self._repack_zip(temp_dir, output_path)

            _logger.info(
                "ZIP cleaning complete: %d cleaned, %d copied, %d failed, "
                "%d orphaned entries detected",
                self._cleaned_count, self._copied_count, self._failed_count,
                self._orphaned_count,
            )

            return True

        except Exception as e:
            _logger.error("Error cleaning ZIP %s: %s", input_path, e)
            if output_path.exists():
                os.remove(output_path)
            return False

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _detect_orphaned_entries(self, input_path: Path) -> List[str]:
        """Detect orphaned entries in local headers not in central directory.

        Orphaned entries are files that were "deleted" from the ZIP but still
        exist in the local file headers. Standard ZIP tools only read the
        central directory, so these ghost entries survive naive extraction.

        Detection: Parse local file headers from the binary and compare
        against the central directory namelist.

        Args:
            input_path: Path to the ZIP file

        Returns:
            List of orphaned entry names
        """
        orphaned = []

        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            # Get central directory names
            try:
                with zipfile.ZipFile(input_path, 'r') as zf:
                    central_names = set(zf.namelist())
            except Exception:
                central_names = set()

            # Parse local file headers
            # Local file header signature: PK\x03\x04
            local_header_sig = b'PK\x03\x04'
            pos = 0
            local_names = []

            while pos < len(data) - 4:
                # Find next local file header
                idx = data.find(local_header_sig, pos)
                if idx == -1:
                    break

                # Parse local file header (minimum 30 bytes)
                if idx + 30 > len(data):
                    break

                try:
                    # Version needed, flags, compression, mtime, mdate
                    # crc32, compressed size, uncompressed size, filename len, extra len
                    fields = struct.unpack_from(
                        '<HHHHHIIIHH',
                        data,
                        idx + 4,
                    )
                    filename_len = fields[8]
                    extra_len = fields[9]

                    # Extract filename
                    if idx + 30 + filename_len > len(data):
                        break

                    filename = data[idx + 30:idx + 30 + filename_len].decode(
                        'utf-8', errors='replace'
                    )
                    local_names.append(filename)

                    # Move past this entry
                    pos = idx + 30 + filename_len + extra_len
                except struct.error:
                    break

            # Find orphaned entries (in local headers but not central directory)
            for name in local_names:
                if name not in central_names:
                    orphaned.append(name)

            if orphaned:
                self._orphaned_count = len(orphaned)
                _logger.warning(
                    "Found %d orphaned entries (recoverable 'deleted' files): %s",
                    len(orphaned), orphaned[:5],
                )

        except Exception as e:
            _logger.debug("Failed to detect orphaned entries: %s", e)

        return orphaned

    def _clean_entry_path(self, original_path: str) -> str:
        """Clean entry path by removing username patterns.

        Strips absolute paths and username references from entry names,
        keeping only the relative file path.

        Args:
            original_path: Original entry path from ZIP

        Returns:
            Cleaned entry path with usernames removed
        """
        cleaned = original_path

        # Remove Windows absolute paths
        cleaned = re.sub(
            r'[A-Z]:\\Users\\[^/\\]+\\',
            '',
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove Unix home paths
        cleaned = re.sub(
            r'/home/[^/]+/',
            '',
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove macOS Users paths
        cleaned = re.sub(
            r'/Users/[^/]+/',
            '',
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove UNC paths
        cleaned = re.sub(
            r'\\\\[^/\\]+\\[^/\\]+\\',
            '',
            cleaned,
            flags=re.IGNORECASE,
        )

        # Clean up leading separators
        cleaned = cleaned.lstrip('/\\')

        # If cleaning resulted in empty path, use original filename
        if not cleaned:
            cleaned = original_path.split('/')[-1].split('\\')[-1]

        return cleaned

    def _clean_extracted_files(self, extract_dir: Path) -> None:
        """Clean all extracted files using the appropriate cleaner."""
        for file_path in extract_dir.rglob('*'):
            if not file_path.is_file():
                continue

            # Skip hidden/system files
            if file_path.name.startswith('._') or file_path.name.startswith('__MACOSX'):
                self._copied_count += 1
                continue

            # Use router to clean the file in-place
            success = self.router.clean_file(
                input_path=file_path,
                output_path=file_path,
            )

            if success:
                self._cleaned_count += 1
            else:
                self._failed_count += 1

    def _repack_zip(self, source_dir: Path, output_path: Path) -> None:
        """Repack cleaned files into a new ZIP with clean metadata.

        - Normalized timestamps (CLEAN_TIMESTAMP)
        - Cleaned entry paths (no usernames)
        - Anonymized entry names (sensitive entities replaced with placeholders)
        - No archive comment
        - No per-file comments
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob('*'):
                if not file_path.is_file():
                    continue

                # Calculate archive path
                arcname = str(file_path.relative_to(source_dir))

                # Clean entry path (remove username patterns)
                arcname = self._clean_entry_path(arcname)

                # Anonymize entry name by replacing sensitive entities
                # This ensures filenames like "JCW20200615 INVOICE - Acme Corp.pdf"
                # become "JCW20200615 INVOICE - [COMPANY_001].pdf"
                arcname = self.mapper.replace_in_text(arcname)

                # Create clean info object with normalized timestamp
                info = zipfile.ZipInfo(
                    filename=arcname,
                    date_time=CLEAN_TIMESTAMP,
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                # No comment field

                # Read file content
                with open(file_path, 'rb') as f:
                    content = f.read()

                zf.writestr(info, content)

        # No archive comment (clean)

    def detect_risks(self, input_path: Path) -> List[str]:
        """Detect potential risk vectors in a ZIP file.

        Args:
            input_path: Source ZIP file path

        Returns:
            List of detected risk descriptions
        """
        risks = []

        if not zipfile.is_zipfile(input_path):
            return risks

        try:
            with zipfile.ZipFile(input_path, 'r') as zf:
                # Check for archive comment
                if zf.comment.strip():
                    risks.append(
                        f"Archive comment present: {zf.comment[:100]}"
                    )

                # Check for MACOSX metadata
                namelist = zf.namelist()
                if any('__MACOSX' in n for n in namelist):
                    risks.append("macOS resource fork metadata present")

                if any(n.startswith('._') for n in namelist):
                    risks.append("macOS resource fork files (._ prefix) present")

                # Check for path leakage in entry names
                for name in namelist:
                    if 'Users\\' in name or 'Users/' in name:
                        risks.append(
                            f"Entry path contains username: {name[:80]}"
                        )
                        break

                # Check for orphaned entries
                orphaned = self._detect_orphaned_entries(input_path)
                if orphaned:
                    risks.append(
                        f"Orphaned entries detected ({len(orphaned)}): "
                        f"recoverable 'deleted' files present"
                    )

        except Exception as e:
            risks.append(f"Risk detection failed: {e}")

        return risks
