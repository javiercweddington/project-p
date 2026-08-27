"""
LLM-backed entity detection and cleanliness verification.

Talks to a LOCAL OpenAI-compatible endpoint (e.g. Qwen served by vLLM at
http://localhost:8000/v1) — no document content ever leaves the machine.

Two roles:

1. LLMEntityDetector — reads the text of already-cleaned files, asks the
   model for any remaining identifying entities (people, companies, emails,
   phones, addresses, product names — English AND Chinese), and registers
   them in the EntityMapper. The pipeline then runs another in-place
   cleaning pass so the new entities are replaced everywhere. Iterated
   until the model finds nothing new.

2. LLMCleanlinessJudge — the final verification gate: after all cleaning,
   any identifying information the model can still find is reported as a
   FAILING verification hit.

Configuration (environment variables):
    PROJECT_P_LLM_BASE    endpoint base   (default http://localhost:8000/v1)
    PROJECT_P_LLM_MODEL   model name      (default qwen27b — set to the
                                           name your server actually serves,
                                           e.g. qwen26b)
    PROJECT_P_LLM_VERIFY  off | auto | required   (default auto)
        off      — never use the LLM
        auto     — use it when the endpoint answers; otherwise continue
                   with an advisory note
        required — a missing/unreachable endpoint FAILS verification
                   (strictest fail-closed posture)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

DEFAULT_BASE = os.environ.get('PROJECT_P_LLM_BASE', 'http://localhost:8000/v1')
DEFAULT_MODEL = os.environ.get('PROJECT_P_LLM_MODEL', 'qwen27b')
DEFAULT_API_KEY = os.environ.get('PROJECT_P_LLM_API_KEY', 'not-needed')


def llm_verify_mode() -> str:
    """Current LLM verification mode: off | auto | required."""
    mode = os.environ.get('PROJECT_P_LLM_VERIFY', 'auto').strip().lower()
    return mode if mode in ('off', 'auto', 'required') else 'auto'


# Entity types the LLM may register, mapped to mapper types.
_LLM_TYPE_MAP = {
    'person': 'person',
    'company': 'company',
    'organization': 'company',
    'org': 'company',
    'email': 'email',
    'phone': 'phone',
    'address': 'address',
    'product': 'product',
}

# Values that must never be registered (generic words the model sometimes
# emits despite instructions).
_VALUE_STOPLIST = {
    'customer', 'supplier', 'company', 'contact', 'address', 'email',
    'phone', 'name', 'unknown', 'n/a', 'none', 'client', 'vendor',
}

# Prefix-anchored, case-sensitive (see anonymizer.PLACEHOLDER_TOKEN_RE):
# a loose pattern here filtered real findings like IMG_20200615 out of
# the LLM's reports.
from .anonymizer import PLACEHOLDER_VALUE_RE as _PLACEHOLDER_RE

_DETECT_SYSTEM_PROMPT = """You are a data-privacy auditor. The user gives you text extracted from a business document that has been anonymized: placeholders like [COMPANY_001], [PERSON_002], [EMAIL_003], FILE_001 are ALREADY-anonymized content — ignore them completely.

Find every piece of IDENTIFYING information that remains. Look for (in ANY language, including Chinese):
- person: real people's names (人名), including partial names
- company: company/organization names (公司名), brand names, web domains
- email: email addresses
- phone: phone/fax/mobile numbers (电话/手机)
- address: street or building addresses (地址)
- product: specific named products or part designations tied to a client project

Do NOT report: generic role words (customer, supplier, CEO), country/city names alone, currencies, quantities, dates alone, material names (PC, MAKROLON is a material brand — DO report material brand names as product), or anything already in [XXX_nnn] placeholder form.

Answer with ONLY a JSON array, no prose:
[{"type":"person","value":"EXACT text as it appears"}, ...]
Return [] if nothing identifying remains."""

_CHUNK_SIZE = 3500
_CHUNK_OVERLAP = 200


def _llm_concurrency() -> int:
    """Number of in-flight requests against the LLM endpoint.

    vLLM throughput scales nearly linearly with concurrent requests;
    sequential chunk calls leave the GPU idle between round-trips.
    """
    try:
        return max(1, int(os.environ.get('PROJECT_P_LLM_CONCURRENCY', '8')))
    except ValueError:
        return 8


class LocalLLM:
    """Minimal OpenAI-compatible chat client over urllib (no SDK needed)."""

    def __init__(self, base_url: str = DEFAULT_BASE,
                 model: str = DEFAULT_MODEL, timeout: int = 180,
                 api_key: str = DEFAULT_API_KEY):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self._available: Optional[bool] = None

    def available(self) -> bool:
        """Probe the endpoint once (GET /models); cached."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                f'{self.base_url}/models',
                headers={'Authorization': f'Bearer {self.api_key}'},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._available = resp.status == 200
        except Exception as e:
            _logger.info("LLM endpoint %s not reachable: %s", self.base_url, e)
            self._available = False
        return self._available

    def chat(self, system: str, user: str, max_tokens: int = 2000) -> str:
        payload = json.dumps({
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'temperature': 0.0,
            'max_tokens': max_tokens,
        }).encode('utf-8')
        req = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data=payload,
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.api_key}'},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        return body['choices'][0]['message']['content'] or ''


def _parse_entity_json(raw: str) -> Optional[List[Dict[str, str]]]:
    """Parse the model's reply; tolerate reasoning text around the JSON.

    Takes the LAST parseable JSON array of {"type","value"} objects
    (reasoning models often emit thinking before the final answer).

    Returns None when the reply is UNPARSEABLE — callers must treat that
    as a failure, not as "no entities found" (a truncated reply silently
    ending discovery would be a fail-open).
    """
    candidates = re.findall(r'\[[^\[\]]*\]', raw, re.DOTALL)
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and all(
                isinstance(e, dict) and 'type' in e and 'value' in e
                for e in data):
            return data
    if raw.strip() in ('[]', ''):
        return []
    # Whole-message parse as a fallback
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [e for e in data
                    if isinstance(e, dict) and 'type' in e and 'value' in e]
    except json.JSONDecodeError:
        pass
    return None


def _iter_chunks(text: str):
    step = _CHUNK_SIZE - _CHUNK_OVERLAP
    for start in range(0, len(text), step):
        chunk = text[start:start + _CHUNK_SIZE]
        if chunk.strip():
            yield chunk


def _detect_chunk(llm: LocalLLM, chunk: str) -> List[Dict[str, str]]:
    """One chunk -> parsed entity dicts. Raises on any failure
    (fail-closed: an unparseable/failed reply is NOT 'no entities')."""
    try:
        reply = llm.chat(_DETECT_SYSTEM_PROMPT, chunk)
    except Exception as e:
        _logger.warning("LLM detection call failed: %s", e)
        raise
    parsed = _parse_entity_json(reply)
    if parsed is None:
        raise ValueError(
            f"LLM reply unparseable (first 120 chars: {reply[:120]!r})")
    return parsed


def _filter_parsed(parsed_lists) -> List[Tuple[str, str]]:
    """Merge parsed chunk replies into deduped (mapper_type, value) pairs."""
    found: List[Tuple[str, str]] = []
    seen = set()
    for parsed in parsed_lists:
        for entity in parsed:
            raw_type = str(entity.get('type', '')).strip().lower()
            value = str(entity.get('value', '')).strip()
            mapper_type = _LLM_TYPE_MAP.get(raw_type)
            if mapper_type is None:
                continue
            # CJK names are routinely 2 characters (朱生); Latin values
            # shorter than 3 are too ambiguous to register.
            has_cjk = re.search(r'[぀-ヿ㐀-䶿一-鿿가-힯]', value)
            if len(value) < (2 if has_cjk else 3) or len(value) > 120:
                continue
            if value.lower() in _VALUE_STOPLIST:
                continue
            if _PLACEHOLDER_RE.fullmatch(value):
                continue
            key = (mapper_type, value.lower())
            if key in seen:
                continue
            seen.add(key)
            found.append((mapper_type, value))
    return found


def detect_entities(llm: LocalLLM, text: str) -> List[Tuple[str, str]]:
    """Ask the LLM for identifying entities in text.

    Chunk requests run concurrently (PROJECT_P_LLM_CONCURRENCY, default 8).
    Returns a list of (mapper_entity_type, value) pairs, filtered for
    junk (stoplist, placeholders, too-short values). Raises if ANY chunk
    fails — a partial scan must not read as a clean scan.
    """
    chunks = list(_iter_chunks(text))
    if not chunks:
        return []
    workers = min(_llm_concurrency(), len(chunks))
    if workers <= 1:
        parsed_lists = [_detect_chunk(llm, c) for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # pool.map preserves chunk order and re-raises the first error
            parsed_lists = list(pool.map(
                lambda c: _detect_chunk(llm, c), chunks))
    return _filter_parsed(parsed_lists)


def detect_entities_batch(
        llm: LocalLLM, texts: Dict[str, str],
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Exception]]:
    """Scan many texts with ONE shared request pool.

    Flattens every (key, chunk) pair into a single pool so small files
    ride along with big ones — per-file sequential scanning leaves the
    endpoint idle whenever a file has fewer chunks than the concurrency.

    Returns (results, errors): per-key entity pairs, and per-key first
    exception for keys whose scan failed (callers decide whether a failed
    key aborts the run or becomes an 'unverifiable' hit).
    """
    tasks: List[Tuple[str, str]] = []
    for key, text in texts.items():
        for chunk in _iter_chunks(text):
            tasks.append((key, chunk))

    results: Dict[str, List[Tuple[str, str]]] = {k: [] for k in texts}
    errors: Dict[str, Exception] = {}
    if not tasks:
        return results, errors

    parsed_by_key: Dict[str, list] = {k: [] for k in texts}
    workers = min(_llm_concurrency(), len(tasks))

    def run(task):
        key, chunk = task
        try:
            return key, _detect_chunk(llm, chunk), None
        except Exception as e:  # collected per-key, never swallowed
            return key, None, e

    if workers <= 1:
        outcomes = [run(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(run, tasks))

    for key, parsed, err in outcomes:
        if err is not None:
            errors.setdefault(key, err)
        elif parsed:
            parsed_by_key[key].append(parsed)

    for key, parsed_lists in parsed_by_key.items():
        if key not in errors:
            results[key] = _filter_parsed(parsed_lists)
    return results, errors


# ---------------------------------------------------------------------------
# Text extraction for scanning cleaned files
# ---------------------------------------------------------------------------

_OFFICE_ZIP_EXTS = {'.docx', '.xlsx', '.pptx', '.xlsm', '.docm', '.pptm',
                    '.odt', '.ods', '.odp'}
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
               '.webp'}
_SKIP_EXTS = {'.mp3', '.wav', '.mp4', '.avi', '.mov', '.stl', '.sldprt',
              '.sldasm'}
_XML_TAG_RE = re.compile(r'<[^>]+>')


def _ocr_image_text(path: Path) -> Optional[str]:
    """OCR an image for LLM scanning (None when tesseract is unavailable)."""
    try:
        from acquire.metadata import ImageOCR
        ocr = ImageOCR()
        if not getattr(ocr, 'available', False):
            return None
        return ocr.extract_text(path)
    except Exception:
        return None


# Extraction cache: (absolute path) -> (content digest, extracted text).
# OCR and PDF parsing are the expensive extractors and files are usually
# unchanged between the discovery scan and the cleanliness judge; keying
# by content digest means a re-cleaned file re-extracts automatically.
_EXTRACT_CACHE: Dict[str, Tuple[str, Optional[str]]] = {}


def _file_digest(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def clear_extract_cache() -> None:
    _EXTRACT_CACHE.clear()


def extract_scannable_text(path: Path, max_chars: int = 200_000) -> Optional[str]:
    """Best-effort text extraction, cached by (path, content digest)."""
    key = str(path)
    digest = _file_digest(path)
    if digest is not None:
        hit = _EXTRACT_CACHE.get(key)
        if hit is not None and hit[0] == digest:
            return hit[1]
    text = _extract_scannable_text_uncached(path, max_chars)
    if digest is not None:
        _EXTRACT_CACHE[key] = (digest, text)
    return text


def _extract_scannable_text_uncached(
        path: Path, max_chars: int = 200_000) -> Optional[str]:
    """Best-effort text extraction from a cleaned file for LLM scanning."""
    suffix = path.suffix.lower()
    if suffix in _SKIP_EXTS:
        return None
    if suffix in _IMAGE_EXTS:
        # Images join the loop via OCR: pixel text the model flags gets
        # registered, and the image re-clean then redacts those pixels.
        return _ocr_image_text(path)

    if suffix == '.pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return '\n'.join(
                (page.extract_text() or '') for page in reader.pages
            )[:max_chars]
        except Exception:
            return None

    if suffix in _OFFICE_ZIP_EXTS:
        try:
            parts = []
            with zipfile.ZipFile(path, 'r') as zf:
                for member in zf.namelist():
                    if member.lower().endswith(('.xml', '.rels')):
                        try:
                            xml_text = zf.read(member).decode(
                                'utf-8', errors='replace')
                            # Strip tags with NO separator: names are often
                            # split across adjacent runs (<w:t>朱</w:t>
                            # <w:t>’R</w:t>) and inserting spaces would hide
                            # them from detection.
                            parts.append(_XML_TAG_RE.sub('', xml_text))
                        except Exception:
                            pass
            return '\n'.join(parts)[:max_chars]
        except zipfile.BadZipFile:
            return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read(max_chars)
    except (UnicodeDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Detector (feeds the mapper) and Judge (verification gate)
# ---------------------------------------------------------------------------

class LLMEntityDetector:
    """Scan cleaned files with the LLM and register findings in the mapper."""

    def __init__(self, mapper, llm: Optional[LocalLLM] = None):
        self.mapper = mapper
        self.llm = llm or LocalLLM()

    def scan_directory(self, staging_dir: Path) -> int:
        """Scan every extractable file; register new entities.

        Extraction is cached; all files' chunks share one concurrent
        request pool. Returns the number of NEWLY registered entities.
        Raises if ANY file's scan failed (fail-closed).
        """
        before = self.mapper.mapping_count
        texts: Dict[str, str] = {}
        for file_path in sorted(staging_dir.rglob('*')):
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
            text = extract_scannable_text(file_path)
            if text and text.strip():
                texts[str(file_path)] = text

        results, errors = detect_entities_batch(self.llm, texts)
        if errors:
            # Endpoint died mid-scan — surface via the caller's mode logic
            key, err = next(iter(errors.items()))
            raise RuntimeError(
                f"LLM scan failed for {len(errors)} file(s), "
                f"first: {Path(key).name}: {err}") from err

        for key, entities in results.items():
            name = Path(key).name
            for entity_type, value in entities:
                placeholder = self.mapper.get_or_create(
                    entity_type, value,
                    source=f'llm_detection:{name}',
                )
                _logger.info(
                    "LLM detected entity %r (%s) in %s -> %s",
                    value, entity_type, name, placeholder,
                )
        return self.mapper.mapping_count - before


class LLMCleanlinessJudge:
    """Final gate: FAIL verification on any identifying info the LLM finds."""

    def __init__(self, mapper, llm: Optional[LocalLLM] = None):
        self.mapper = mapper
        self.llm = llm or LocalLLM()

    def run_check(self, cleaned_dir: Path):
        # Local import to avoid a circular import at module load
        from .verifier import VerificationResult, LeakageHit

        mode = llm_verify_mode()
        if mode == 'off':
            return VerificationResult(
                check_name="LLM Cleanliness Check",
                passed=True,
                details="Disabled (PROJECT_P_LLM_VERIFY=off)",
            )

        if not self.llm.available():
            unavailable_msg = (
                f"LLM endpoint {self.llm.base_url} unreachable — "
                f"cleanliness NOT verified by LLM."
            )
            if mode == 'required':
                return VerificationResult(
                    check_name="LLM Cleanliness Check",
                    passed=False,
                    details=unavailable_msg + " (PROJECT_P_LLM_VERIFY=required)",
                )
            return VerificationResult(
                check_name="LLM Cleanliness Check",
                passed=True,
                details="ADVISORY: " + unavailable_msg,
            )

        hits: List[LeakageHit] = []
        texts: Dict[str, str] = {}
        rels: Dict[str, str] = {}
        for file_path in sorted(cleaned_dir.rglob('*')):
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
            text = extract_scannable_text(file_path)
            if not text or not text.strip():
                continue
            key = str(file_path)
            texts[key] = text
            rels[key] = str(file_path.relative_to(cleaned_dir))
        scanned = len(texts)

        results, errors = detect_entities_batch(self.llm, texts)
        for key, err in errors.items():
            hits.append(LeakageHit(
                file_path=rels[key],
                entity_type='unverifiable',
                original=f'LLM scan failed: {err}',
            ))
        for key, entities in results.items():
            for entity_type, value in entities:
                hits.append(LeakageHit(
                    file_path=rels[key],
                    entity_type=f'llm_{entity_type}',
                    original=value,
                    context='Identifying information found by LLM after cleaning',
                ))

        return VerificationResult(
            check_name="LLM Cleanliness Check",
            passed=len(hits) == 0,
            details=f"LLM-scanned {scanned} files via {self.llm.model} "
                    f"at {self.llm.base_url}",
            hits=hits,
        )
