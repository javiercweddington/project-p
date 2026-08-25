"""
TextCleaner - clean plain text files using entity spans from acquisition.

Uses character offsets from GLiNER/entity detection to perform
precise, surgical replacements. Falls back to regex-based replacement
when spans are not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..anonymizer import EntityMapper, SpanBasedReplacer

_logger = logging.getLogger(__name__)


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
                   encoding: str = 'utf-8') -> bool:
        """Clean a text file.

        Args:
            input_path: Source file path
            output_path: Destination file path (in /tmp/)
            entity_spans: List of (start, end, entity_type, source) from acquisition
            encoding: File encoding

        Returns:
            True if cleaning was successful
        """
        try:
            with open(input_path, 'r', encoding=encoding) as f:
                text = f.read()

            if entity_spans:
                cleaned = self.replacer.replace(text, entity_spans)
            else:
                # Fallback: replace all known entities via regex
                cleaned = self.mapper.replace_in_text(text)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding=encoding) as f:
                f.write(cleaned)

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