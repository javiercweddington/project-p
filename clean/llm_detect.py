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
    """Current LLM verification mode: off | auto | required | judge | sample.

    judge = LLM is used ONLY for the final cleanliness check (no
    discovery): deterministic + CV detection carry cleaning, and the
    model audits the finished output once. Unreachable endpoint fails
    the check (you asked for an audit; a skipped audit is not a pass).

    sample = LLM audits ONE cleaned file per extension, its findings are
    registered and the affected files re-cleaned, then the samples (and
    re-cleaned files) are audited again. Two bounded LLM rounds over a
    small subset instead of judging every file — O(types), not O(files).
    Residual round-2 findings FAIL the run.
    """
    mode = os.environ.get('PROJECT_P_LLM_VERIFY', 'auto').strip().lower()
    return mode if mode in ('off', 'auto', 'required', 'judge',
                            'sample') else 'auto'


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
    # Location granularity all maps to 'address' (cities/towns/countries
    # and postal codes are identifying for a supply chain).
    'location': 'address',
    'city': 'address',
    'town': 'address',
    'country': 'address',
    'zip': 'address',
    'zipcode': 'address',
    'zip code': 'address',
    'zip_code': 'address',
    'postal code': 'address',
    'postal_code': 'address',
}

# Values that must never be registered (generic words the model sometimes
# emits despite instructions). Shared by the LLM paths AND GLiNER
# auto-registration — 'client' slipped in through GLiNER (which had no
# stoplist) and its boundary pattern then flagged/replaced the ordinary
# word everywhere.
_VALUE_STOPLIST = {
    'customer', 'supplier', 'company', 'contact', 'address', 'email',
    'phone', 'name', 'unknown', 'n/a', 'none', 'client', 'vendor',
    # Software/producer strings (docProps <Application>, PDF Producer):
    # tool fingerprints, not client-identifying — registering them turns
    # every Office file's metadata into a permanent "leak".
    'microsoft excel', 'microsoft word', 'microsoft powerpoint',
    'microsoft office', 'microsoft', 'excel', 'word', 'powerpoint',
    'openpyxl', 'libreoffice', 'openoffice', 'wps office', 'wps',
    'adobe', 'acrobat', 'adobe acrobat', 'pdf', 'windows',
    # Generic system account names (registered as 'person' live)
    'user', 'admin', 'administrator', 'guest', 'owner',
    # Generic hardware/packaging vocabulary: cannot identify a client,
    # and registering them blacks out every engineering drawing table
    # ('HOLE SAW PULL-OFF TEST' died to a 'HOLE SAW' product entry).
    'hole saw', 'hole saw arbor', 'hole saw kit', 'pilot bit',
    'ball bearing', 'bearing ball', 'arbor', 'arbor shank', 'collar',
    'shank', 'forged shank', 'color bands', 'shrink film',
    # Generic web/software strings seen auto-registered as companies
    'www.google.com', 'google', 'sap',
    # SharePoint/Office internals reported as products live
    'documentlibraryform', 'document id generator',
    # Generic mechanical components (reported as llm_product live —
    # 'WASHER', 'COMPRESSION SPRING' identify nothing)
    'washer', 'spring', 'compression spring', 'shaft', 'bearing',
    'screw', 'nut', 'bolt', 'gasket', 'o-ring',
}

# Value SHAPES that are extraction junk, not entities. These all showed
# up registered live (jacky run): spreadsheet cell refs tagged as
# phone/address by GLiNER, formula fragments as doc refs, XML runs
# concatenated with ISO timestamps as person names, and dimension
# triples as products.
_CELL_REF_RE = re.compile(r'^[A-Za-z]{1,3}\d{1,4}$')
_CELL_REF_CHAIN_RE = re.compile(
    r'^[A-Za-z]{1,3}\d{1,4}([-:][A-Za-z]{1,3}\d{1,4})+$')
_FORMULA_TOKEN_RE = re.compile(
    r'(?i)(IFERROR|VLOOKUP|HLOOKUP|SUMIFS?|COUNTIFS?|CONCATENATE'
    r'|ISBLANK|INDIRECT|falsefalse|truetrue)')
_TIMESTAMP_GLUE_RE = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}')
_NUMERIC_ONLY_RE = re.compile(r'^[\d.,\s-]+$')


def _junk_shaped(value: str, entity_type: Optional[str] = None) -> bool:
    """True when a candidate value is extraction junk by SHAPE.

    Applied at every registration choke point (GLiNER discovery, LLM
    detection, deterministic identifier scan). Deliberately narrow:
    cell refs cap at 4 digits so real doc ids (MM00032628) still
    register.
    """
    v = value.strip()
    if _CELL_REF_RE.match(v) or _CELL_REF_CHAIN_RE.match(v):
        return True
    # Dotted code namespaces ('Microsoft.Office.DocumentManagement'):
    # software fingerprints, not client entities.
    if re.match(r'(?i)^(microsoft|com|org|net|system|office)\.\w', v):
        return True
    if _FORMULA_TOKEN_RE.search(v):
        return True
    if _TIMESTAMP_GLUE_RE.search(v):
        return True
    if entity_type in ('product', 'company', 'person') \
            and _NUMERIC_ONLY_RE.match(v):
        return True
    return False

_VERSION_SUFFIX_RE = re.compile(r'^[\sv.\d()+-]+$')


def _stoplisted(value: str) -> bool:
    """True when a value must never be registered.

    Exact stoplist hit, or a stoplisted software name followed by a
    version-like suffix ('Openpyxl 3.1.53.1', 'Microsoft Excel 2016' —
    seen registered live). The suffix must look like a version so real
    names such as 'Word Industries Ltd' are NOT blocked.
    """
    v = value.lower().strip()
    if v in _VALUE_STOPLIST:
        return True
    words = v.split()
    for n in (1, 2):
        if len(words) > n and ' '.join(words[:n]) in _VALUE_STOPLIST:
            suffix = ' '.join(words[n:])
            if _VERSION_SUFFIX_RE.match(suffix):
                return True
    return False

# Prefix-anchored, case-sensitive (see anonymizer.PLACEHOLDER_TOKEN_RE):
# a loose pattern here filtered real findings like IMG_20200615 out of
# the LLM's reports.
from .anonymizer import PLACEHOLDER_VALUE_RE as _PLACEHOLDER_RE

_DETECT_SYSTEM_PROMPT = """You are a data-privacy auditor. The user gives you text extracted from a business document that has been anonymized: placeholders like [COMPANY_001], [PERSON_002], [EMAIL_003], FILE_001 are ALREADY-anonymized content — ignore them completely.

Find every piece of IDENTIFYING information that remains. Look for (in ANY language, including Chinese):
- person: real people's names (人名), including partial names AND names appearing in signature blocks / signature lines (签名)
- company: company/organization names (公司名), brand names, web domains
- email: email addresses
- phone: phone/fax/mobile numbers (电话/手机)
- address: street or building addresses (地址), AND standalone city/town/country names (城市/国家) AND postal/zip codes — report each as type "address"
- product: BRANDED or model-designated products ONLY — trademark names, product-line names, SKU/model codes tied to a client project

Do NOT report: generic role words (customer, supplier, CEO), software/application names (Microsoft Excel, Adobe), currencies, quantities, dates alone, or anything already in [XXX_nnn] placeholder form. NEVER report generic component/engineering vocabulary as product: part descriptions (washer, compression spring, arbor, shank, collar, bearing, end cap), colors ("Red Plastic"), materials or material+part combos ("6150 Shank", "Plastic (PC) Insert Molded Steel"), or design-option labels ("COLLAR OPTION 1", "Arbor (Bearing Design)") — these identify nothing. Material BRAND names (MAKROLON) are the exception: DO report those as product.

Answer with ONLY a JSON array, no prose:
[{"type":"person","value":"EXACT text as it appears"}, ...]
Return [] if nothing identifying remains."""

_CHUNK_SIZE = 3500
_CHUNK_OVERLAP = 200


def _llm_concurrency() -> int:
    """Number of in-flight requests against the LLM endpoint.

    vLLM throughput scales nearly linearly with concurrent requests;
    sequential chunk calls leave the GPU idle between round-trips.
    (Measured: 8 workers gave 7.6x overlap on a 4-GPU TP server — the
    server was not the bottleneck, so the default is 16.)
    """
    try:
        return max(1, int(os.environ.get('PROJECT_P_LLM_CONCURRENCY', '16')))
    except ValueError:
        return 16


def _llm_scan_cap() -> int:
    """Per-file char cap for LLM scanning (PROJECT_P_LLM_MAX_CHARS,
    default 60000; <=0 disables). The deterministic Entity Leakage Check
    scans every file in FULL — the LLM pass is a semantic auditor, and
    capping its input trades no hard guarantee."""
    try:
        return int(os.environ.get('PROJECT_P_LLM_MAX_CHARS', '60000'))
    except ValueError:
        return 60000


def _llm_max_tokens() -> int:
    """Reply-token budget (PROJECT_P_LLM_MAX_TOKENS, default 4096).
    A truncated reply is a FAILED scan, not an empty one — and
    product-dense spreadsheets overflowed 2000 tokens live (structured
    JSON is verbose: ~25 tokens per entity)."""
    try:
        return max(64, int(os.environ.get('PROJECT_P_LLM_MAX_TOKENS', '4096')))
    except ValueError:
        return 4096


def _detect_system_prompt() -> str:
    """System prompt, with Qwen3-style thinking disabled by default.

    Reasoning models emit hundreds of hidden thinking tokens before the
    JSON (measured 31s mean per call); '/no_think' turns that off for
    Qwen3-family models and is inert noise for others. Disable with
    PROJECT_P_LLM_NOTHINK=0 if your model needs its reasoning budget.
    """
    if os.environ.get('PROJECT_P_LLM_NOTHINK', '1') != '0':
        return _DETECT_SYSTEM_PROMPT + '\n/no_think'
    return _DETECT_SYSTEM_PROMPT


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

    def chat(self, system: str, user: str, max_tokens: int = 2000,
             extra: Optional[Dict] = None) -> str:
        body_fields = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'temperature': 0.0,
            'max_tokens': max_tokens,
        }
        if extra:
            body_fields.update(extra)
        payload = json.dumps(body_fields).encode('utf-8')
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
    # Structured-output replies: bare array, or the object-root wrapper
    # {"entities": [...]} required by strict json_schema backends.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get('entities'), list):
        return [e for e in data['entities']
                if isinstance(e, dict) and 'type' in e and 'value' in e]

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

    # Truncation salvage: max_tokens cuts dense replies mid-array
    # ('{"entities": [{"type": ...' then silence — seen live on
    # product-heavy xlsx). Every COMPLETE {"type","value"} object is
    # still recoverable; only the object the cut landed in is lost.
    # Salvage keeps the audit useful instead of failing the whole file.
    if re.match(r'\s*[\[{]', raw):
        objs = re.findall(
            r'\{\s*"type"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'"value"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', raw)
        if objs:
            _logger.warning(
                "LLM reply truncated mid-JSON; salvaged %d complete "
                "entities (raise PROJECT_P_LLM_MAX_TOKENS to avoid).",
                len(objs))
            return [{'type': json.loads(f'"{t}"'),
                     'value': json.loads(f'"{v}"')} for t, v in objs]
    return None


def _iter_chunks(text: str):
    step = _CHUNK_SIZE - _CHUNK_OVERLAP
    for start in range(0, len(text), step):
        chunk = text[start:start + _CHUNK_SIZE]
        if chunk.strip():
            yield chunk


# Guided decoding (vLLM `guided_json`): a grammar constraint on the output
# tokens — the model CANNOT emit a thinking preamble or prose, only JSON
# matching this schema, from the first token. Kills the measured
# hundreds-of-CoT-tokens-per-reply cost AND guarantees parseability.
# PROJECT_P_LLM_GUIDED=0 disables; a server that rejects the parameter
# (HTTP 4xx) automatically falls back to unguided for the rest of the run.
# Object-root schema: strict backends (OpenAI json_schema spec, xgrammar
# strict mode) require the ROOT to be an object, not an array. The parser
# accepts both the bare array and this {"entities": [...]} wrapper.
_ENTITY_ARRAY_SCHEMA = {
    'type': 'array',
    'items': {
        'type': 'object',
        'properties': {
            'type': {'type': 'string'},
            'value': {'type': 'string'},
        },
        'required': ['type', 'value'],
        'additionalProperties': False,
    },
}
_GUIDED_SCHEMA = {
    'type': 'object',
    'properties': {'entities': _ENTITY_ARRAY_SCHEMA},
    'required': ['entities'],
    'additionalProperties': False,
}

# Structured-output negotiation. vLLM generations want different request
# fields — 'structured_outputs' on current builds (0.10+; guided_json is
# deprecated/removed there), 'guided_json' historically, OpenAI-style
# 'response_format' in between — and servers silently IGNORE fields they
# don't know (measured live on vLLM 0.27: guided_json AND response_format
# both HTTP 200 with the constraint ignored). The mode self-adapts:
# advance on HTTP 4xx OR when a "constrained" reply doesn't start with
# JSON. PROJECT_P_LLM_GUIDED=0 disables entirely.
_STRUCTURED_MODES = ('structured_outputs', 'guided_json',
                     'response_format', 'off')
_structured_idx = 0


def _structured_extra() -> Optional[Dict]:
    if os.environ.get('PROJECT_P_LLM_GUIDED', '1') == '0':
        return None
    mode = _STRUCTURED_MODES[min(_structured_idx, len(_STRUCTURED_MODES) - 1)]
    if mode == 'structured_outputs':
        return {'structured_outputs': {'json': _GUIDED_SCHEMA}}
    if mode == 'guided_json':
        return {'guided_json': _GUIDED_SCHEMA}
    if mode == 'response_format':
        return {'response_format': {
            'type': 'json_schema',
            'json_schema': {'name': 'entity_list',
                            'strict': True,
                            'schema': _GUIDED_SCHEMA}}}
    return None


def _advance_structured_mode(reason: str) -> None:
    global _structured_idx
    if _structured_idx < len(_STRUCTURED_MODES) - 1:
        _structured_idx += 1
        _logger.warning(
            "Structured output mode %r failed (%s); switching to %r.",
            _STRUCTURED_MODES[_structured_idx - 1], reason,
            _STRUCTURED_MODES[_structured_idx])


def _detect_chunk(llm: LocalLLM, chunk: str) -> List[Dict[str, str]]:
    """One chunk -> parsed entity dicts. Raises on any failure
    (fail-closed: an unparseable/failed reply is NOT 'no entities')."""
    while True:
        extra = _structured_extra()
        try:
            reply = llm.chat(_detect_system_prompt(), chunk,
                             max_tokens=_llm_max_tokens(), extra=extra)
        except urllib.error.HTTPError as e:
            if extra is not None and 400 <= e.code < 500:
                _advance_structured_mode(f'rejected with HTTP {e.code}')
                continue
            _logger.warning("LLM detection call failed: %s", e)
            raise
        except Exception as e:
            _logger.warning("LLM detection call failed: %s", e)
            raise
        break
    if extra is not None and not reply.lstrip().startswith(('[', '{')):
        # Server accepted the field but ignored the constraint.
        _advance_structured_mode('constraint ignored — reply began with prose')
    parsed = _parse_entity_json(reply)
    if parsed is None:
        raise ValueError(
            f"LLM reply unparseable (first 120 chars: {reply[:120]!r})")
    return parsed


def _filter_parsed(parsed_lists) -> List[Tuple[str, str]]:
    """Merge parsed chunk replies into deduped (mapper_type, value) pairs."""
    from .anonymizer import targeted_types
    allowed = targeted_types()
    found: List[Tuple[str, str]] = []
    seen = set()
    for parsed in parsed_lists:
        for entity in parsed:
            raw_type = str(entity.get('type', '')).strip().lower()
            value = str(entity.get('value', '')).strip()
            mapper_type = _LLM_TYPE_MAP.get(raw_type)
            if mapper_type is None:
                continue
            # Targeting policy: this single gate covers the LLM detector,
            # the sample audit AND the cleanliness judge — a names-only
            # run must neither register nor FAIL over llm_address /
            # llm_product / llm_phone content that ships by design.
            if allowed is not None and mapper_type not in allowed:
                continue
            # CJK names are routinely 2 characters (朱生); Latin values
            # shorter than 3 are too ambiguous to register.
            has_cjk = re.search(r'[぀-ヿ㐀-䶿一-鿿가-힯]', value)
            if len(value) < (2 if has_cjk else 3) or len(value) > 120:
                continue
            if _stoplisted(value) or _junk_shaped(value, mapper_type):
                continue
            if _PLACEHOLDER_RE.fullmatch(value):
                continue
            # Placeholder combos ('[ENTITY_002]™ [COMPANY_025]'): already
            # pseudonymized text the judge echoes back — registering it
            # would mint placeholder-of-placeholder entries.
            if not re.search(r'[A-Za-z0-9]',
                             re.sub(r'\[[A-Z]+_\d{3}\]', '', value)):
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
              '.sldasm',
              # CAD ASCII: machine-generated coordinate streams — one STEP
              # file cost 58 of 92 judge calls for zero detection value.
              # The Entity Leakage Check still regex-scans these in FULL.
              '.step', '.stp', '.igs', '.iges'}
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
        cap = _llm_scan_cap()
        texts: Dict[str, str] = {}
        for file_path in sorted(staging_dir.rglob('*')):
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
            text = extract_scannable_text(file_path)
            if text and text.strip():
                texts[str(file_path)] = text[:cap] if cap > 0 else text

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
        if mode == 'sample':
            return VerificationResult(
                check_name="LLM Cleanliness Check",
                passed=True,
                details="Delegated to the LLM Sample Audit check "
                        "(PROJECT_P_LLM_VERIFY=sample)",
            )

        if not self.llm.available():
            unavailable_msg = (
                f"LLM endpoint {self.llm.base_url} unreachable — "
                f"cleanliness NOT verified by LLM."
            )
            if mode in ('required', 'judge'):
                return VerificationResult(
                    check_name="LLM Cleanliness Check",
                    passed=False,
                    details=unavailable_msg
                            + f" (PROJECT_P_LLM_VERIFY={mode})",
                )
            return VerificationResult(
                check_name="LLM Cleanliness Check",
                passed=True,
                details="ADVISORY: " + unavailable_msg,
            )

        hits: List[LeakageHit] = []
        cap = _llm_scan_cap()
        texts: Dict[str, str] = {}
        rels: Dict[str, str] = {}
        for file_path in sorted(cleaned_dir.rglob('*')):
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
            text = extract_scannable_text(file_path)
            if not text or not text.strip():
                continue
            key = str(file_path)
            texts[key] = text[:cap] if cap > 0 else text
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
