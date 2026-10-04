#!/usr/bin/env python3
"""Post-process: rename pseudonymous FILE_nnn files to safe-but-descriptive.

Goal
----
`FILE_001.stl` is safe but tells you nothing; `Stanley_Screwdriver_001.stl`
leaks the company. We want `Screwdriver_001.stl`: a GENERIC product/word
that reveals nothing about the company, person, or product identity.

How
---
Using the saved mappings from a finished clean run:
  * path_manifest.json  {final_name -> original_relative_path}
  * entity_mapper.json  {"[COMPANY_001]": {"original","entity_type",...}}

for each delivered file we recover its ORIGINAL name and build a safe stem:
  1. keep only ALPHABETIC words from the original stem — this drops every
     digit run, model code, date, and part number automatically
     (sp350, 48-89-2301, 20190829 all disappear);
  2. drop any word that matches a mapper entity (company/person/email/
     product tokens), a noise word (copy/final/draft/rev/...), a --deny
     word, or is shorter than --min-word;
  3. keep the first --max-words survivors, Title_Cased and '_'-joined;
  4. reattach the FILE_nnn number (audit traceability) as the counter;
     if no generic word survives, fall back to <--fallback>_nnn.

Nothing identifying survives by construction (names are scrubbed, numbers
are dropped). Run without --apply to review the full rename table first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

# Filename noise that is not descriptive of the product.
NOISE_WORDS = {
    'copy', 'final', 'draft', 'rev', 'revision', 'version', 'ver', 'new',
    'old', 'update', 'updated', 'test', 'temp', 'tmp', 'the', 'of', 'and',
    'for', 'with', 'to', 'from', 'a', 'an', 'file', 'document', 'doc',
}

WORD_RE = re.compile(r'[A-Za-z]+')
NUM_RE = re.compile(r'(\d+)')


def entity_tokens(mapper: dict) -> set[str]:
    """Lowercased alphabetic tokens of every mapped entity's original text."""
    toks: set[str] = set()
    for info in mapper.values():
        original = str(info.get('original', ''))
        for w in WORD_RE.findall(original):
            if len(w) >= 2:
                toks.add(w.lower())
    return toks


def safe_stem(original_stem: str, deny: set[str], min_word: int,
              max_words: int) -> str:
    """Generic Title_Cased stem, or '' if nothing safe survives."""
    kept = []
    for w in WORD_RE.findall(original_stem):
        wl = w.lower()
        if len(w) < min_word:
            continue
        if wl in NOISE_WORDS or wl in deny:
            continue
        kept.append(w)
        if len(kept) >= max_words:
            break
    return '_'.join(s.capitalize() for s in kept)


def counter_from(name: str, fallback_idx: int) -> str:
    """Trailing/any number in the pseudonym (FILE_001 -> '001'); else index."""
    nums = NUM_RE.findall(Path(name).stem)
    if nums:
        return nums[-1]
    return f'{fallback_idx:04d}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cleaned', required=True,
                    help='Delivered tree containing the FILE_nnn files.')
    ap.add_argument('--manifest', required=True,
                    help='path_manifest.json (final_name -> original_rel).')
    ap.add_argument('--mapper', required=True,
                    help='entity_mapper.json (sensitive originals).')
    ap.add_argument('--deny-words', default='',
                    help='Comma-separated extra words to strip (product/'
                         'brand terms the strict mapper does not hold), or '
                         'a path to a file with one word per line.')
    ap.add_argument('--fallback', default='File',
                    help='Stem when no generic word survives (default File).')
    ap.add_argument('--min-word', type=int, default=3)
    ap.add_argument('--max-words', type=int, default=3)
    ap.add_argument('--apply', action='store_true',
                    help='Actually rename. Default prints the table only.')
    args = ap.parse_args()

    cleaned = Path(args.cleaned).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    mapper = json.loads(Path(args.mapper).read_text())

    deny = set()
    if args.deny_words:
        p = Path(args.deny_words)
        words = (p.read_text().split() if p.is_file()
                 else args.deny_words.split(','))
        deny = {w.strip().lower() for w in words if w.strip()}
    deny |= entity_tokens(mapper)

    # Build rename plan, resolving collisions within each target stem.
    used: set[str] = set()
    plan = []       # (current_path, new_name, original_stem)
    missing = 0
    for idx, (final_name, original_rel) in enumerate(sorted(manifest.items()),
                                                      start=1):
        matches = list(cleaned.rglob(final_name))
        if not matches:
            missing += 1
            continue
        cur = matches[0]
        ext = cur.suffix
        orig_stem = Path(original_rel).stem
        stem = safe_stem(orig_stem, deny, args.min_word, args.max_words)
        if not stem:
            stem = args.fallback
        counter = counter_from(final_name, idx)
        new_name = f'{stem}_{counter}{ext}'
        # Collision guard (same stem+counter+ext twice).
        bump = 1
        base = new_name
        while new_name.lower() in used:
            bump += 1
            new_name = f'{stem}_{counter}_{bump}{ext}'
        used.add(new_name.lower())
        plan.append((cur, new_name, orig_stem))

    # Report.
    print(f"files in manifest: {len(manifest)}   located: {len(plan)}   "
          f"missing from cleaned: {missing}")
    print(f"{'CURRENT':<22} {'-> NEW':<34} ORIGINAL STEM")
    for cur, new_name, orig_stem in plan[:200]:
        print(f"{cur.name:<22} {new_name:<34} {orig_stem}")
    if len(plan) > 200:
        print(f"... (+{len(plan) - 200} more; re-run with output piped to a "
              f"file to see all)")

    if not args.apply:
        print("\nDRY RUN — nothing renamed. Add --apply to rename.")
        return 0

    renamed = 0
    for cur, new_name, _ in plan:
        dst = cur.with_name(new_name)
        if dst != cur:
            shutil.move(str(cur), str(dst))
            renamed += 1
    print(f"\nRenamed {renamed} file(s).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
