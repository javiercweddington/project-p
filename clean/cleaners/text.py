"""
TextCleaner - clean plain text files using entity spans from acquisition.

Uses character offsets from GLiNER/entity detection to perform
precise, surgical replacements. Falls back to regex-based replacement
when spans are not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

from ..anonymizer import EntityMapper, SpanBasedReplacer

_logger = logging.getLogger(__name__)

# Encodings to try in order when reading text files.
# UTF-8 is the most common, GBK/GB18030 cover Chinese-encoded files,
# and latin-1 is a safe fallback that never fails (maps bytes 0-255 directly).
_FALLBACK_ENCODINGS: Iterator[str] | None = None


def _encoding_fallback() -> Iterator[str]:
    """Yield encodings to try in order for fail-open text reading."""
    yield 'utf-8'
    yield 'gbk'
    yield 'gb18030'
    yield 'utf-16'
    yield 'latin-1'  # Never fails; maps every byte


def _read_text_file(input_path: Path) -> tuple[str, str]:
    """Read a text file, trying multiple encodings until one succeeds.

    Returns (text, encoding_used) or raises UnicodeDecodeError if all fail.
    """
    for enc in _encoding_fallback():
        try:
            with open(input_path, 'r', encoding=enc, newline='') as f:
                text = f.read()
            return text, enc
        except (UnicodeDecodeError, UnicodeDecodeError):
            continue
        except LookupError:
            # Unknown encoding name on this platform; skip
            continue
    raise UnicodeDecodeError('open', b'', 0, 1, 'all fallback encodings exhausted')


def _write_text_file(output_path: Path, text: str, encoding: str = 'utf-8') -> None:
    """Write text to a file, preserving line endings via newline=''."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding=encoding, newline='') as f:
        f.write(text)


class TextCleaner:
    """Clean plain text files using entity spans from acquisition.

    Uses character offsets from GLiNER/entity detection to perform
    precise, surgical replacements.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper
        self.replacer = SpanBasedReplacer(mapper)

    def clean_file(self, input_path: Path, output_path: Path,
                   entity_spans: Optional[List[Tuple[int, int, str, str]]] = None,
                   encoding: str | None = None) -> bool:
        """Clean a text file.

        Args:
            input_path: Source file path
            output_path: Destination file path (in /tmp/)
            entity_spans: List of (start, end, entity_type, source) from acquisition
            encoding: File encoding. If None, tries multiple encodings (utf-8, gbk,
                      gb18030, utf-16, latin-1) until one succeeds (fail-open).

        Returns:
            True if cleaning was successful
        """
        try:
            if encoding is not None:
                # Explicit encoding requested; use it directly but preserve line endings
                with open(input_path, 'r', encoding=encoding, newline='') as f:
                    text = f.read()
                out_enc = encoding
            else:
                # Fail-open: try multiple encodings
                text, out_enc = _read_text_file(input_path)

            if entity_spans:
                cleaned = self.replacer.replace(text, entity_spans)
            else:
                # Fallback: replace all known entities via regex
                cleaned = self.mapper.replace_in_text(text)

            _write_text_file(output_path, cleaned, encoding=out_enc)

            return True

        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            _logger.warning("Encoding error cleaning %s: %s", input_path, e)
            return False
        except OSError as e:
            _logger.error("OS error cleaning %s: %s", input_path, e)
            return False

    def clean_text(self, text: str,
                   entity_spans: Optional[List[Tuple[int, int, str, str]]] = None) -> str:
        """Clean text content directly (without file I/O)."""
        if entity_spans:
            return self.replacer.replace(text, entity_spans)
        return self.mapper.replace_in_text(text)