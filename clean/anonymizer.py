"""
Anonymizer module - consistent entity-to-placeholder mapping.

Maintains a global, deterministic mapping of sensitive entities to
synthetic placeholders. The same entity text always maps to the same
placeholder across all documents and file types within a cleaning session.

Placeholder format:
    [PERSON_001], [COMPANY_001], [LOCATION_001], [EMAIL_001], [PHONE_001]
"""

from __future__ import annotations

import os
import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

_logger = logging.getLogger(__name__)


# Entity type → placeholder prefix mapping
ENTITY_PREFIX_MAP = {
    'person': 'PERSON',
    'company': 'COMPANY',
    'location': 'LOCATION',
    'email': 'EMAIL',
    'phone': 'PHONE',
    'address': 'ADDRESS',
    'sensitive_doc': 'DOCREF',
    'account': 'ACCOUNT',
    'filename': 'FILE',
    'directory': 'DIR',
}

# Entity types recorded for the audit trail only — NEVER used for text
# replacement. Filename stems can be short/generic ("in", "report"), so
# letting them into replace_in_text would corrupt ordinary document text.
NON_TEXT_ENTITY_TYPES = frozenset({'filename', 'directory'})

# ---- Targeting policy -------------------------------------------------
# PROJECT_P_ENTITY_TYPES limits which entity types DETECTORS may
# auto-register. Human seeds (--seed / --seed-file) are exempt: whatever
# a person deliberately enrolls is honored at any type. Everything a
# detector finds outside the policy is DEMOTED to a review suggestion
# instead of registered — it ships untouched, and the human can promote
# it to a seed on the next run. Default is names-only: doc-ref codes,
# phones, addresses, part numbers, dimensions and products are exactly
# the blueprint content that must be left alone.
_TARGETS_ENV = 'PROJECT_P_ENTITY_TYPES'
_DEFAULT_TARGETS = 'person,company,email'


def targeted_types() -> Optional[frozenset]:
    """Entity types detectors may auto-register; None means all (legacy)."""
    raw = os.environ.get(_TARGETS_ENV, _DEFAULT_TARGETS).strip().lower()
    if raw in ('all', '*'):
        return None
    return frozenset(t.strip() for t in raw.split(',') if t.strip())

# (original, is_person) -> substring needles for the replacement/verify
# prefilter (see EntityMapper.prefilter_needles). Module-wide: originals
# repeat across mapper instances within a run.
_PREFILTER_NEEDLE_CACHE: Dict[Tuple[str, bool], tuple] = {}

# Default prefix for unknown entity types
_DEFAULT_PREFIX = 'ENTITY'

# The ONLY placeholder shapes this pipeline mints, anchored to the actual
# prefixes and CASE-SENSITIVE. A loose [A-Z]+_\d+ pattern let real names
# like IMG_20200615 or ACME_2020 masquerade as placeholders and bypass the
# strict filename policy entirely.
_ALL_PREFIXES = sorted(set(ENTITY_PREFIX_MAP.values()) | {_DEFAULT_PREFIX})
PLACEHOLDER_TOKEN_RE = re.compile(
    r'\[?(?:' + '|'.join(_ALL_PREFIXES) + r')_\d{3,}\]?'
)
PLACEHOLDER_VALUE_RE = re.compile(
    r'^\[?(?:' + '|'.join(_ALL_PREFIXES) + r')_\d{3,}\]?$'
)

# Person-name tokens that are also common English words; matching these
# standalone would corrupt ordinary prose ('Will Smith' -> every "will").
PERSON_TOKEN_STOPWORDS = frozenset({
    'will', 'bill', 'mark', 'grant', 'frank', 'rose', 'may', 'june',
    'jack', 'art', 'gene', 'ray', 'rich', 'sunny', 'young', 'long',
    'white', 'black', 'brown', 'green', 'stone', 'wood', 'hill', 'park',
    'price', 'love', 'joy', 'guy', 'norm', 'dean', 'earl', 'king',
})


# Separators that may remain in a fully anonymized path component
NAME_SEPARATOR_RE = re.compile(r'[\s\-_.,;()+&#@!~\[\]{}]+')


def anonymize_path_component(mapper: 'EntityMapper', name: str,
                             entity_type: str) -> str:
    """Fully anonymize one file/directory/zip-member name (STRICT policy).

    Entity-replace first; if the stem still carries ANY residue beyond
    placeholders and separators (dates, initials, app names, unmapped CJK),
    the whole stem becomes a FILE_nnn/DIR_nnn pseudonym registered in the
    audit mapper. The extension is preserved.
    """
    import os as _os
    if entity_type == 'filename':
        stem, ext = _os.path.splitext(name)
    else:
        stem, ext = name, ''

    replaced = mapper.replace_in_text(stem, source='path_anonymization')

    residue = PLACEHOLDER_TOKEN_RE.sub('', replaced)
    residue = NAME_SEPARATOR_RE.sub('', residue)
    if residue:
        placeholder = mapper.get_or_create(
            entity_type, stem, source='path_anonymization',
        )
        replaced = placeholder.strip('[]')

    return replaced + ext


_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]+$")


def _person_name_variants(name: str) -> List[str]:
    """Derived writings of a two-part Latin personal name.

    'Ward, Bryan' also appears as 'Bryan Ward', 'B. Ward' and 'B.WARD'
    in drawing title blocks (all four seen live; only the registered
    form was caught). Matching is case-insensitive downstream, so one
    dotted variant covers the caps form. The dotless glued 'BWard' is
    also derived: OCR drops the dot ('B.WARD' read as 'BWARD' shipped
    a drafter name on live sheets) and it is the standard Windows
    username shape (C:\\Users\\bward). Only clean two-token names
    qualify — usernames, digits, initials, CJK and glued XML runs
    derive nothing.
    """
    name = name.strip()
    if ',' in name:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) != 2 or not all(parts):
            return []
        last, first = parts
    else:
        parts = name.split()
        if len(parts) != 2:
            return []
        first, last = parts
    if not (_NAME_TOKEN_RE.match(first) and _NAME_TOKEN_RE.match(last)):
        return []
    variants = [f'{first} {last}', f'{last}, {first}',
                f'{first[0]}. {last}', f'{first[0]}.{last}']
    # Glued initial+surname ('BWard') covers OCR dot-drops ('B.WARD'
    # reads as 'BWARD') and username shapes — but initial+surname is
    # often an ordinary WORD ('Carol Ash' -> 'cash', 'Brian Rand' ->
    # 'brand'), and a word-variant would replace that word everywhere
    # in prose (live-reproduced in review). Register the glued form
    # only when it is not in the system dictionary; with no dictionary
    # available, skip it — content corruption outranks the residual.
    glued = f'{first[0]}{last}'
    if not _is_dictionary_word(glued):
        variants.append(glued)
    return variants


_DICT_WORDS: Optional[frozenset] = None


def _is_dictionary_word(token: str) -> bool:
    """True when token is an ordinary English word (or no dict exists —
    fail toward NOT minting risky variants)."""
    global _DICT_WORDS
    if _DICT_WORDS is None:
        words: Set[str] = set()
        for path in ('/usr/share/dict/words',
                     '/usr/share/dict/american-english'):
            try:
                with open(path, encoding='utf-8', errors='ignore') as fh:
                    words = {w.strip().lower() for w in fh
                             if 2 < len(w.strip()) <= 12}
                break
            except OSError:
                continue
        _DICT_WORDS = frozenset(words)
    if not _DICT_WORDS:
        return True
    return token.lower() in _DICT_WORDS


@dataclass
class EntityMapping:
    """A single mapping from original entity text to a placeholder."""
    original: str
    placeholder: str
    entity_type: str
    sources: List[str] = field(default_factory=list)
    occurrence_count: int = 0


class EntityMapper:
    """Maintains deterministic entity-to-placeholder mappings.

    The same entity text always produces the same placeholder.
    Mappings are case-insensitive for normalization but preserve
    the original casing for the first-seen entity.

    Optionally accepts a tracker callback that is invoked for every
    substitution so downstream components (e.g., the verifier's
    ChangeTracker) can record what was actually replaced.
    """

    def __init__(self, tracker_callback=None):
        # Normalized key → EntityMapping
        self._mappings: Dict[str, EntityMapping] = {}
        # Counter per placeholder prefix (not per entity_type) so that unknown
        # types sharing the [ENTITY_nnn] prefix don't collide.
        self._counters: Dict[str, int] = defaultdict(int)
        # Reverse mapping: placeholder → original
        self._reverse: Dict[str, str] = {}
        # Optional callback: (original, placeholder, source) -> None
        self._tracker_callback = tracker_callback
        # Detector findings the targeting policy refused to register,
        # deduped by (type, value): review-file suggestions the human can
        # promote to seeds.
        self._demoted: Dict[Tuple[str, str], Dict] = {}

    @property
    def mappings(self) -> List[EntityMapping]:
        """Return all entity mappings."""
        return list(self._mappings.values())

    def _demote(self, entity_type: str, value: str,
                source: Optional[str]) -> None:
        """Record a detector finding the targeting policy refused."""
        key = (entity_type, value.strip().lower())
        entry = self._demoted.get(key)
        if entry is None:
            entry = {'type': entity_type, 'value': value.strip(),
                     'count': 0, 'sources': []}
            self._demoted[key] = entry
        entry['count'] += 1
        # Keep the detector family, drop the per-file suffix.
        src = (source or '').split(':', 1)[0]
        if src and src not in entry['sources']:
            entry['sources'].append(src)

    @property
    def demoted_suggestions(self) -> List[Dict]:
        """Findings blocked by the targeting policy, most frequent first.

        Written to review_candidates.json so a human can promote real
        targets to seeds (--seed / --seed-file) on the next run.
        """
        return sorted(self._demoted.values(), key=lambda e: -e['count'])

    @property
    def mapping_count(self) -> int:
        return len(self._mappings)

    def _generate_variants(self, original: str) -> List[str]:
        """Generate normalized variants of an entity value for matching.

        Addresses cases where "Globus Medical" appears as "globusmedical",
        "globus_medical", "globus-medical", etc. in emails, URLs, or
        concatenated text. Trailing punctuation is stripped from variants
        so "NOA Labs Ltd." also matches "NOA Labs Ltd" and "noalabsltd".

        Returns a list of unique variants (deduplicated, lowercase).
        """
        variants = set()
        base = original.lower().strip()
        variants.add(base)

        # Variant without trailing punctuation ("Ltd." -> "Ltd")
        base_nopunct = base.rstrip('.,;:')
        if len(base_nopunct) >= 2:
            variants.add(base_nopunct)

        for stem in (base, base_nopunct):
            if not stem:
                continue
            # Remove spaces, hyphens, underscores for concatenated forms.
            # Strip trailing punctuation so 'noalabsltd.' doesn't consume
            # the dot in 'noalabsltd.com'.
            collapsed = re.sub(r'[\s\-_]+', '', stem).rstrip('.,;:')
            if len(collapsed) >= 2:
                variants.add(collapsed)
            # Replace spaces with hyphens / underscores
            variants.add(re.sub(r'\s+', '-', stem))
            variants.add(re.sub(r'\s+', '_', stem))

        # XML-escaped spellings — both classes shipped inside cleaned
        # Office members (seen live):
        # 1. numeric character references: 'Gürses' stored as
        #    'G&#252;rses' ('Ü' -> &#220;, 'ü' -> &#252;);
        # 2. predefined entities: 'Package & Content Development' stored
        #    as 'Package &amp; Content Development'.
        for source_form in (original.strip(), base, base_nopunct):
            if not source_form:
                continue
            if any(ord(ch) > 127 for ch in source_form):
                escaped = ''.join(
                    ch if ord(ch) < 128 else f'&#{ord(ch)};'
                    for ch in source_form)
                variants.add(escaped.lower())
            if any(ch in '&<>' for ch in source_form):
                escaped = (source_form
                           .replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;'))
                variants.add(escaped.lower())

        return [v for v in variants if len(v) >= 2]

    # CJK ranges (Han, Hiragana/Katakana, Hangul): \b is meaningless between
    # CJK characters because Python's re treats them all as word characters,
    # so boundary-anchored patterns can NEVER match a CJK entity inside
    # continuous CJK text.
    _CJK_RE = re.compile(
        r'[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]'
    )

    def _build_pattern_cached(self, original: str,
                              entity_type: Optional[str] = None) -> Optional[re.Pattern]:
        """Cached wrapper: patterns depend only on (original, entity_type),
        so compile each exactly once. Uncached, replace_in_text recompiled
        every pattern per call and thrashed re's 512-entry cache on large
        mappers (quadratic slowdown on big workbooks)."""
        cache = getattr(self, '_pattern_cache', None)
        if cache is None:
            cache = self._pattern_cache = {}
        key = (original, entity_type)
        if key not in cache:
            cache[key] = self._build_pattern(original, entity_type)
        return cache[key]

    def _build_pattern(self, original: str,
                       entity_type: Optional[str] = None) -> Optional[re.Pattern]:
        """Build a boundary-aware, case-insensitive regex pattern for an entity.

        Strategy:
        - CJK-containing variants: matched with NO boundaries (CJK scripts
          have no word delimiters; boundary anchors would prevent any match
          inside continuous CJK text).
        - All other variants: lookaround boundaries applied only on sides
          that end in a word character, so entities with leading/trailing
          punctuation ("NOA Labs Ltd.") still match, while short entities
          ("SA") never match inside larger words ("USA").
        - PERSON entities additionally match their individual name tokens
          ("Josh Woodard" also matches a bare "Woodard"), because surname-
          only mentions identify the person just as well.

        All variant patterns are joined with | (longest first) so the regex
        engine tries the most specific match first.
        """
        variants = set(self._generate_variants(original))
        if entity_type == 'person':
            for token in re.split(r'[\s\-_]+', original.lower().strip()):
                token = token.strip('.,;:')
                # Skip tokens that are common English words — 'Will Smith'
                # must not redact every occurrence of "will".
                if len(token) >= 3 and token not in PERSON_TOKEN_STOPWORDS:
                    variants.add(token)

        parts = []
        for variant in sorted(variants, key=len, reverse=True):
            if len(variant) < 2:
                continue
            escaped = re.escape(variant)
            # Multi-word entities must match across line wraps: PDF/OCR
            # extraction routinely breaks 'Lech Alexander Murawski' onto
            # separate lines, which a literal-space pattern never matches.
            escaped = re.sub(r'(?:\\\s|\s)+', r'\\s+', escaped)
            if self._CJK_RE.search(variant):
                parts.append(escaped)
                continue
            # Underscore counts as a SEPARATOR (filenames like
            # report_globusmedical_2020.pdf), so boundaries exclude it:
            # only letters/digits block a match at the edge.
            left = r'(?<![A-Za-z0-9])' if variant[0].isalnum() else ''
            right = r'(?![A-Za-z0-9])' if variant[-1].isalnum() else ''
            parts.append(left + escaped + right)

        if not parts:
            # A value too short/degenerate to match safely: no pattern.
            # (The old fallback compiled a boundary-FREE single-char pattern
            # that replaced every occurrence of that letter in a document.)
            return None

        pattern_str = '(?:' + '|'.join(parts) + ')'
        return re.compile(pattern_str, re.IGNORECASE)

    def get_or_create(self, entity_type: str, value: str,
                      source: Optional[str] = None,
                      derive_variants: bool = True) -> str:
        """Get existing placeholder or create a new one.

        Args:
            entity_type: Type of entity (person, company, etc.)
            value: Original entity text
            source: Source file/path for attribution
            derive_variants: person entities also register their other
                common writings ('Ward, Bryan' -> 'Bryan Ward',
                'B. Ward', 'B.Ward') so title-block initials are caught.
                Internal recursion guard — leave True.

        Returns:
            The placeholder string (e.g., "[PERSON_001]")
        """
        key = value.strip().lower()
        if not key:
            return value

        # Never map a placeholder itself: on re-clean passes, metadata
        # fields already hold '[PERSON_002]'-style values — mapping those
        # would mint chains of placeholder-for-placeholder entries.
        # Anchored to real prefixes and case-sensitive so genuine values
        # like 'IMG_20200615' or 'acme_2020' are still mappable.
        if PLACEHOLDER_VALUE_RE.match(value.strip()):
            return value.strip()

        # CENTRAL gates. Human seeds keep the last word (both bypasses
        # use the same startswith('seed') predicate, covering --seed and
        # --seed-file). Path pseudonyms (filename/directory) bypass BOTH
        # gates: they are minted BY policy, not detected — the stoplist
        # blocking 'Bearing'/'Collar' stems shipped six drawings under
        # their ORIGINAL names in the HoleSaw run.
        if (key not in self._mappings
                and not (source or '').startswith('seed')
                and entity_type not in NON_TEXT_ENTITY_TYPES):
            # Stoplist gate: junk values ('User', 'Microsoft Excel')
            # minted by paths that bypassed per-detector filters.
            try:
                from .llm_detect import _stoplisted
                if _stoplisted(value):
                    return value.strip()
            except ImportError:
                pass
            # TARGETING gate: detectors may only register policy types
            # (default names-only). Everything else becomes a review
            # suggestion and the content ships untouched.
            allowed = targeted_types()
            if allowed is not None and entity_type not in allowed:
                self._demote(entity_type, value, source)
                return value.strip()

        if key in self._mappings:
            mapping = self._mappings[key]
            mapping.occurrence_count += 1
            if source and source not in mapping.sources:
                mapping.sources.append(source)
            return mapping.placeholder

        # Create new mapping
        prefix = ENTITY_PREFIX_MAP.get(entity_type, _DEFAULT_PREFIX)
        # Key counter by prefix (not entity_type) so two unknown types that
        # both map to "ENTITY" don't mint [ENTITY_001] twice each.
        self._counters[prefix] += 1
        counter = self._counters[prefix]
        placeholder = f"[{prefix}_{counter:03d}]"

        mapping = EntityMapping(
            original=value.strip(),
            placeholder=placeholder,
            entity_type=entity_type,
            sources=[source] if source else [],
            occurrence_count=1,
        )
        self._mappings[key] = mapping
        self._reverse[placeholder] = value.strip()

        # NOTE: the tracker callback is deliberately NOT fired here —
        # registration is not a substitution. Firing it on registration
        # produced phantom "entities replaced" counts for mere seeding.
        # replace_in_text / replace_spans fire it per actual replacement.

        if entity_type == 'person' and derive_variants:
            # Variants inherit the parent's seed-ness: under a custom
            # --targets that excludes 'person', a SEEDED person's
            # variants must not be demoted by the targeting gate
            # ('seed_variant' passes the startswith('seed') predicate).
            variant_src = ('seed_variant' if (source or '').startswith(
                'seed') else 'name_variant')
            for variant in _person_name_variants(value):
                if variant.strip().lower() != key:
                    self.get_or_create(
                        'person', variant,
                        source=f'{variant_src}:{value.strip()}',
                        derive_variants=False)

        return placeholder

    def resolve(self, placeholder: str) -> Optional[str]:
        """Look up original entity text from a placeholder."""
        return self._reverse.get(placeholder)

    def has_entity(self, value: str) -> bool:
        """Check if an entity value has been mapped."""
        return value.strip().lower() in self._mappings

    def get_mapping(self, value: str) -> Optional[EntityMapping]:
        """Get the mapping for an entity value."""
        return self._mappings.get(value.strip().lower())

    def prefilter_needles(self, original: str,
                          entity_type: Optional[str] = None) -> tuple:
        """Cheap substring needles gating this entity's pattern.

        Same soundness argument as the verifier's prefilter: variant
        generation only rewrites SEPARATORS, so every variant match
        contains all original tokens — non-person entities gate on the
        single longest token; person patterns also match lone name
        tokens, so persons gate on any-of-all-tokens.

        Tokens containing non-ASCII letters additionally contribute
        their XML numeric-reference spelling ('gürses' AND 'g&#252;rses')
        so escaped-form variants are never prefiltered away.

        Returns () when nothing is selective enough — callers must then
        run the pattern unconditionally.
        """
        cache_key = (original, entity_type == 'person')
        cached = _PREFILTER_NEEDLE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        tokens = tuple(
            t for t in re.findall(r'\w+', original.lower())
            if len(t) >= (2 if re.search(r'[^\W\da-z_]', t) else 3)
        )
        if tokens and entity_type != 'person':
            tokens = (max(tokens, key=len),)
        needles = []
        for token in tokens:
            needles.append(token)
            if any(ord(ch) > 127 for ch in token):
                needles.append(''.join(
                    ch if ord(ch) < 128 else f'&#{ord(ch)};'
                    for ch in token))
        needles = tuple(needles)
        _PREFILTER_NEEDLE_CACHE[cache_key] = needles
        return needles

    def replace_in_text(self, text: str, source: str = "") -> str:
        """Replace all known entities in text with their placeholders.

        Uses boundary-aware regex replacement with variant generation
        to handle cases where entities appear in different formats:
        - "Globus Medical" matches "globusmedical" (in emails)
        - "Globus Medical" matches "globus-medical", "globus_medical"
        - Short entities use word boundaries to avoid partial matches

        Longer entities are replaced first to prevent conflicts.

        When a tracker callback is configured, each successful substitution
        is recorded via the callback.
        """
        if not self._mappings:
            return text

        # Sort mappings by original length (descending) to handle overlaps.
        # Audit-only types (filename/directory stems) are excluded — they
        # are often short/generic and would corrupt normal text.
        sorted_mappings = sorted(
            (m for m in self._mappings.values()
             if m.entity_type not in NON_TEXT_ENTITY_TYPES),
            key=lambda m: len(m.original),
            reverse=True,
        )

        placeholder_tail = re.compile(r'_\d{3,}\]')

        result = text
        # Prefilter against the ORIGINAL text, computed once: replacement
        # only removes entity text and inserts ASCII placeholders (which
        # the substitute-guard below refuses to rewrite), so a pattern
        # whose needles are absent from the input cannot gain a match
        # mid-loop. With 300+ entities this gate is the difference
        # between seconds and minutes on multi-MB XML members.
        text_lower = text.lower()
        for mapping in sorted_mappings:
            needles = self.prefilter_needles(
                mapping.original, mapping.entity_type)
            if needles and not any(n in text_lower for n in needles):
                continue

            pattern = self._build_pattern_cached(
                mapping.original, mapping.entity_type)
            if pattern is None:
                continue

            def _substitute(match: re.Match, _mapping=mapping) -> str:
                # Guard against corrupting an already-inserted placeholder:
                # an entity literally named e.g. "Company" would otherwise
                # match inside "[COMPANY_001]" (case-insensitive).
                start, end = match.start(), match.end()
                subject = match.string
                if (start > 0 and subject[start - 1] == '['
                        and placeholder_tail.match(subject, end)):
                    return match.group(0)

                _mapping.occurrence_count += 1
                if source and source not in _mapping.sources:
                    _mapping.sources.append(source)
                if self._tracker_callback is not None:
                    self._tracker_callback(
                        _mapping.original, _mapping.placeholder, source)
                return _mapping.placeholder

            result = pattern.sub(_substitute, result)

        return result

    def replace_spans(self, text: str,
                      spans: List[Tuple[int, int, str]],
                      source: str = "") -> str:
        """Replace entity spans in text using character offsets.

        Args:
            text: Original text
            spans: List of (start, end, entity_type) tuples.
                   The text[start:end] is the entity value.
            source: Optional source path for the tracker callback.

        Returns:
            Text with entities replaced by placeholders.
        """
        if not spans:
            return text

        # Sort spans by start position (descending) to preserve offsets
        sorted_spans = sorted(spans, key=lambda s: s[0], reverse=True)

        result = text
        for start, end, entity_type in sorted_spans:
            entity_value = result[start:end]
            placeholder = self.get_or_create(entity_type, entity_value, source)
            # Demoted/stoplisted values come back unchanged — no
            # replacement happened, so no tracker record.
            if (self._tracker_callback is not None
                    and placeholder != entity_value):
                self._tracker_callback(entity_value, placeholder, source)
            result = result[:start] + placeholder + result[end:]

        return result

    def summary(self) -> str:
        """Get a human-readable summary of all mappings."""
        lines = [f"Entity Mapper Summary: {self.mapping_count} mappings"]
        lines.append("=" * 60)

        # Group by entity type
        by_type: Dict[str, List[EntityMapping]] = defaultdict(list)
        for m in self._mappings.values():
            by_type[m.entity_type].append(m)

        for etype in sorted(by_type.keys()):
            mappings = by_type[etype]
            lines.append(f"\n{etype.upper()} ({len(mappings)}):")
            for m in sorted(mappings, key=lambda x: x.placeholder):
                lines.append(
                    f"  {m.placeholder} = {m.original!r} "
                    f"(x{m.occurrence_count})"
                )

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Dict]:
        """Export mappings as a serializable dictionary."""
        result = {}
        for m in self._mappings.values():
            result[m.placeholder] = {
                'original': m.original,
                'entity_type': m.entity_type,
                'occurrence_count': m.occurrence_count,
                'sources': m.sources,
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> 'EntityMapper':
        """Load mappings from a dictionary."""
        mapper = cls()
        for placeholder, info in data.items():
            entity_type = info['entity_type']
            original = info['original']
            key = original.lower()
            mapping = EntityMapping(
                original=original,
                placeholder=placeholder,
                entity_type=entity_type,
                sources=info.get('sources', []),
                occurrence_count=info.get('occurrence_count', 0),
            )
            mapper._mappings[key] = mapping
            mapper._reverse[placeholder] = original
            # Restore counter (keyed by prefix, not entity_type)
            prefix = ENTITY_PREFIX_MAP.get(entity_type, _DEFAULT_PREFIX)
            match = re.match(rf'\[{prefix}_(\d+)\]', placeholder)
            if match:
                num = int(match.group(1))
                mapper._counters[prefix] = max(
                    mapper._counters[prefix], num
                )
        return mapper


class SpanBasedReplacer:
    """Replace entities using character offsets from acquisition pass.

    This is the primary replacement engine for text cleaning.
    It uses the exact character offsets from GLiNER/entity detection
    to perform surgical replacements without regex ambiguity.
    """

    def __init__(self, mapper: EntityMapper):
        self.mapper = mapper

    def replace(self, text: str, entity_spans: List[Tuple[int, int, str, str]]) -> str:
        """Replace entities in text using precise character offsets.

        Args:
            text: Original text
            entity_spans: List of (start, end, entity_type, source) tuples

        Returns:
            Text with entities replaced by placeholders.
        """
        if not entity_spans:
            return text

        # Sort by start position descending to preserve offsets during replacement
        sorted_spans = sorted(entity_spans, key=lambda s: s[0], reverse=True)

        result = text
        for start, end, entity_type, source in sorted_spans:
            if start >= len(result) or end > len(result):
                _logger.warning("Span (%d, %d) out of bounds for text length %d",
                               start, end, len(result))
                continue

            entity_value = result[start:end].strip()
            if not entity_value:
                continue

            placeholder = self.mapper.get_or_create(entity_type, entity_value, source)
            if (self.mapper._tracker_callback is not None
                    and placeholder != entity_value):
                self.mapper._tracker_callback(entity_value, placeholder, source)
            result = result[:start] + placeholder + result[end:]

        return result

    def replace_with_original_boundaries(self, text: str,
                                          entity_spans: List[Tuple[int, int, str, str]]) -> str:
        """Replace entities, preserving original whitespace boundaries.

        Handles cases where the entity span includes leading/trailing whitespace.
        """
        if not entity_spans:
            return text

        sorted_spans = sorted(entity_spans, key=lambda s: s[0], reverse=True)

        result = text
        for start, end, entity_type, source in sorted_spans:
            if start >= len(result) or end > len(result):
                continue

            entity_value = result[start:end]
            stripped = entity_value.strip()
            if not stripped:
                continue

            # Preserve surrounding whitespace
            left_ws = entity_value[:len(entity_value) - len(entity_value.lstrip())]
            right_ws = entity_value[len(entity_value.rstrip()):]

            placeholder = self.mapper.get_or_create(entity_type, stripped, source)
            result = result[:start] + left_ws + placeholder + right_ws + result[end:]

        return result