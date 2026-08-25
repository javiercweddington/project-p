"""
Anonymizer module - consistent entity-to-placeholder mapping.

Maintains a global, deterministic mapping of sensitive entities to
synthetic placeholders. The same entity text always maps to the same
placeholder across all documents and file types within a cleaning session.

Placeholder format:
    [PERSON_001], [COMPANY_001], [LOCATION_001], [EMAIL_001], [PHONE_001]
"""

from __future__ import annotations

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
}

# Default prefix for unknown entity types
_DEFAULT_PREFIX = 'ENTITY'


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

    @property
    def mappings(self) -> List[EntityMapping]:
        """Return all entity mappings."""
        return list(self._mappings.values())

    @property
    def mapping_count(self) -> int:
        return len(self._mappings)

    def _generate_variants(self, original: str) -> List[str]:
        """Generate normalized variants of an entity value for matching.

        Addresses cases where "Globus Medical" appears as "globusmedical",
        "globus_medical", "globus-medical", etc. in emails, URLs, or
        concatenated text.

        Returns a list of unique variants (deduplicated, lowercase).
        """
        variants = set()
        variants.add(original.lower().strip())

        # Remove spaces, hyphens, underscores for concatenated forms
        collapsed = re.sub(r'[\s\-_]+', '', original).lower().strip()
        if collapsed and len(collapsed) >= 2:
            variants.add(collapsed)

        # Replace spaces with hyphens
        hyphenated = re.sub(r'\s+', '-', original).lower().strip()
        if hyphenated != original.lower():
            variants.add(hyphenated)

        # Replace spaces with underscores
        underscored = re.sub(r'\s+', '_', original).lower().strip()
        if underscored != original.lower():
            variants.add(underscored)

        return list(variants)

    def _build_pattern(self, original: str) -> re.Pattern:
        """Build a boundary-aware, case-insensitive regex pattern for an entity.

        Strategy:
        - Variants containing spaces (e.g. "globus medical"): use \\b word
          boundaries so they do not match inside larger words.
        - Variants WITHOUT spaces (e.g. "globusmedical" or "sa"): use BOTH
          \\b word boundaries (for standalone words like "SA") AND context-aware
          boundaries (for email domains like @globusmedical.com).
        - Pure-numeric variants: always use \\b word boundaries.

        All variant patterns are joined with | (longest first) into a single
        alternation so the regex engine tries the most specific match first.
        """
        variants = self._generate_variants(original)

        # Separate into groups
        spaced = []
        no_space = []
        pure_numeric = []

        for v in variants:
            if v.isdigit():
                pure_numeric.append(v)
            elif ' ' in v:
                spaced.append(v)
            else:
                no_space.append(v)

        parts = []

        # Spaced variants: standard word boundary
        for v in sorted(spaced, key=len, reverse=True):
            parts.append(r'\b' + re.escape(v) + r'\b')

        # No-space alphabetic variants: use BOTH word boundary AND context-aware
        for v in sorted(no_space, key=len, reverse=True):
            escaped = re.escape(v)
            # Word boundary pattern (for standalone words like "SA")
            parts.append(r'\b' + escaped + r'\b')
            # Context-aware pattern (for email domains like @globusmedical.com)
            # Use (?<!\w) lookbehind to prevent matching inside larger words
            # (e.g., "sa" should not match inside "USA")
            parts.append(
                r'(?<!\w)' + escaped + r'(?:\.|@|(?=[<>,;:\s_\-/])|$)'
            )

        # Pure numeric: word boundary
        for v in sorted(pure_numeric, key=len, reverse=True):
            parts.append(r'\b' + re.escape(v) + r'\b')

        if not parts:
            # Fallback: should not happen for valid entities
            return re.compile(re.escape(original.lower()), re.IGNORECASE)

        pattern_str = '(?:' + '|'.join(parts) + ')'
        return re.compile(pattern_str, re.IGNORECASE)

    def get_or_create(self, entity_type: str, value: str,
                      source: Optional[str] = None) -> str:
        """Get existing placeholder or create a new one.

        Args:
            entity_type: Type of entity (person, company, etc.)
            value: Original entity text
            source: Source file/path for attribution

        Returns:
            The placeholder string (e.g., "[PERSON_001]")
        """
        key = value.strip().lower()
        if not key:
            return value

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

        # Notify tracker callback of the new mapping
        if self._tracker_callback is not None:
            self._tracker_callback(value.strip(), placeholder, source or "")

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

        # Sort mappings by original length (descending) to handle overlaps
        sorted_mappings = sorted(
            self._mappings.values(),
            key=lambda m: len(m.original),
            reverse=True,
        )

        result = text
        for mapping in sorted_mappings:
            pattern = self._build_pattern(mapping.original)

            # Count and record replacements for tracker
            if self._tracker_callback is not None:
                matches = pattern.findall(result)
                for _ in matches:
                    self._tracker_callback(mapping.original, mapping.placeholder, source)

            result = pattern.sub(mapping.placeholder, result)

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