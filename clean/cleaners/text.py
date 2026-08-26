"""
TextCleaner - clean plain text files using entity spans from acquisition.

Uses character offsets from GLiNER/entity detection to perform
precise, surgical replacements. Falls back to regex-based replacement
when spans are not available.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

from ..anonymizer import EntityMapper, SpanBasedReplacer

_logger = logging.getLogger(__name__)

# High-precision email pattern for automatic entity registration.
# Emails are unambiguous PII, so any email seen in any text is registered
# as an entity — this is what catches addresses no acquisition pass mapped.
_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

# Project/invoice code pattern: 2-5 uppercase letters (someone's initials or
# a company prefix) immediately followed by 6+ digits, e.g. JCW20200615A.
# These codes embed identities and dates; 6-digit minimum keeps standards
# designators like ISO9001 (4 digits) out.
_DOC_CODE_RE = re.compile(r'\b[A-Z]{2,5}\d{6,}[A-Z0-9]*\b')

# Phone numbers, high-precision forms only:
# - international: +<cc> then separator-grouped digits (+1-610-930-1800,
#   +86-131-4690-7122)
# - CN landline with extension: 0769-88685007-608
# - CN mobile: bare 11 digits starting 1[3-9]
_PHONE_RES = (
    re.compile(r'(?<![\dA-Za-z])\+\d{1,3}(?:[-. ]\d{2,5}){2,5}(?![\dA-Za-z])'),
    re.compile(r'(?<![\dA-Za-z])0\d{2,3}-\d{7,8}(?:-\d{2,5})?(?![\dA-Za-z])'),
    re.compile(r'(?<![\dA-Za-z])1[3-9]\d{9}(?![\dA-Za-z])'),
)

# Web domains: require the www. prefix (bare foo.com risks matching
# filenames); the email detector already covers user@domain forms.
_DOMAIN_RE = re.compile(
    r'\b(?:www\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b', re.IGNORECASE,
)

# Registration / ID numbers: long digit run + 2 or more dash groups,
# e.g. 60567723-000-11-12-A, 848-375366-838.
_ID_CODE_RE = re.compile(r'\b\d{6,}(?:-[\dA-Z]{1,4}){2,}\b')

# Account-number shapes: 3+ groups of 3-6 digits separated by space/dash,
# e.g. "848 375 366 838". Dates/amounts don't match (dots, commas, 1-2
# digit groups).
_ACCOUNT_RE = re.compile(r'\b\d{3,6}(?:[ -]\d{3,6}){2,}\b')

# Context-anchored account / SWIFT detection for forms the generic shape
# misses ("Account Number: 848375366838", "SWIFT Address: HSBCHKHHHKH").
_ACCOUNT_CTX_RE = re.compile(
    r'(?:account\s*(?:number|no\.?)|a/c|账号|iban)[^0-9A-Za-z]{0,12}'
    r'(\d[\d\- ]{6,}\d)', re.IGNORECASE,
)
_SWIFT_CTX_RE = re.compile(
    r'(?:swift|bic)(?:\s*(?:address|code|no\.?|number))?'
    r'[^A-Za-z0-9]{0,20}([A-Z0-9]{8}(?:[A-Z0-9]{3})?)(?![A-Za-z0-9])',
    re.IGNORECASE,
)

# Dotted part numbers tied to client projects, e.g. 6203.4000.0186
_PART_NO_RE = re.compile(r'\b\d{3,4}\.\d{3,4}\.\d{3,4}\b')

# US-style street addresses: number + TitleCase words + street suffix,
# e.g. "300 Griffin Brook Drive", "1 Queen's Road Central".
_STREET_RE = re.compile(
    r"\b\d{1,5}(?:\s+[A-Z][A-Za-z'’]+){1,5}\s+"
    r"(?:Street|St|Drive|Dr|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|"
    r"Court|Ct|Way|Place|Pl)\b(?:\s+(?:Central|East|West|North|South))?",
)

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
            # BOM-less UTF-16LE ASCII "successfully" decodes as UTF-8 into
            # NUL-interleaved text (J\x00o\x00h\x00n) that no entity pattern
            # can match — a silent total cleaning bypass. Reject any decode
            # that yields NULs and let the utf-16 attempt handle it.
            if enc != 'utf-16' and '\x00' in text:
                continue
            return text, enc
        except UnicodeError:
            # Covers UnicodeDecodeError plus bare UnicodeError raised by
            # codecs like utf-16 on truncated/BOM-less data.
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
                # Auto-register any email addresses, then replace all
                # known entities via regex
                self._register_emails(text, source=str(input_path))
                cleaned = self.mapper.replace_in_text(text, source=str(input_path))

            _write_text_file(output_path, cleaned, encoding=out_enc)

            return True

        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            _logger.warning("Encoding error cleaning %s: %s", input_path, e)
            return False
        except OSError as e:
            _logger.error("OS error cleaning %s: %s", input_path, e)
            return False

    def _register_emails(self, text: str, source: str = "") -> None:
        """Register auto-detectable identifiers found in the text.

        - Emails are unambiguous identifiers; auto-registration means an
          address like master@ttmlens.com is pseudonymized consistently even
          when no acquisition pass mapped it.
        - Project/invoice codes like JCW20200615A embed personal initials
          plus dates and are registered as document references.
        """
        for match in _EMAIL_RE.finditer(text):
            self.mapper.get_or_create(
                'email', match.group(0),
                source=source or 'auto_email_detection',
            )
        for match in _DOC_CODE_RE.finditer(text):
            self.mapper.get_or_create(
                'sensitive_doc', match.group(0),
                source=source or 'auto_doc_code_detection',
            )
        for phone_re in _PHONE_RES:
            for match in phone_re.finditer(text):
                # Guard against engineering dimension/tolerance callouts:
                # a real phone number has 7-15 digits.
                digit_count = len(re.sub(r'\D', '', match.group(0)))
                if not 7 <= digit_count <= 15:
                    continue
                self.mapper.get_or_create(
                    'phone', match.group(0),
                    source=source or 'auto_phone_detection',
                )
        for match in _DOMAIN_RE.finditer(text):
            self.mapper.get_or_create(
                'company', match.group(0),
                source=source or 'auto_domain_detection',
            )
        for pattern, etype in ((_ID_CODE_RE, 'account'),
                               (_ACCOUNT_RE, 'account'),
                               (_PART_NO_RE, 'product'),
                               (_STREET_RE, 'address')):
            for match in pattern.finditer(text):
                if etype == 'account':
                    # Require >= 9 digits so dimension triplets in tables
                    # ("100 200 300" is exactly 9 — require 10 to be safe)
                    if len(re.sub(r'\D', '', match.group(0))) < 10:
                        continue
                self.mapper.get_or_create(
                    etype, match.group(0),
                    source=source or 'auto_identifier_detection',
                )
        for pattern in (_ACCOUNT_CTX_RE, _SWIFT_CTX_RE):
            for match in pattern.finditer(text):
                self.mapper.get_or_create(
                    'account', match.group(1),
                    source=source or 'auto_identifier_detection',
                )

    def clean_text(self, text: str,
                   entity_spans: Optional[List[Tuple[int, int, str, str]]] = None,
                   source: str = "") -> str:
        """Clean text content directly (without file I/O)."""
        if entity_spans:
            return self.replacer.replace(text, entity_spans)
        self._register_emails(text, source=source)
        return self.mapper.replace_in_text(text, source=source)