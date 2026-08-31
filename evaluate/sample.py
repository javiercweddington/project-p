#!/usr/bin/env python
"""Stratified samplers for the two human-review gates.

Stage 1 (precheck): a stratified random sample of the SOURCE corpus,
reviewed by a human before cleaning. Findings seed the entity mapper
and the media review. Discovery sample — it informs the pipeline.

Stage 4 (audit): a fresh stratified random sample of the CLEANED
deliverable, disjoint from the stage-1 sample, blind-reviewed for
survivors. Measurement sample — it bounds the residual rate, and the
same files must not serve both purposes or the estimate is circular.

Sample size comes from the rule of three: if n randomly sampled
documents are all clean, the 95% upper bound on the residual rate is
~3/n. So --bound 0.10 needs 30 clean documents, --bound 0.03 needs
100. Pick the bound you are willing to defend.

    # Stage 1, before cleaning
    python -m evaluate.sample precheck --corpus SOURCE_DIR \
        --project NAME --bound 0.10

    # Stage 4, after cleaning (disjointness enforced via the stage-1
    # record + the flat-output path manifest)
    python -m evaluate.sample audit --corpus CLEANED_DIR \
        --project NAME --bound 0.10 \
        --stage1-record AUDIT/NAME/sample_precheck.json \
        --path-manifest AUDIT/NAME/path_manifest.json

Each run writes sample_<stage>.json next to the audit map: the file
list, the seed, the sizing arithmetic, and (for audit) a per-file
verdict skeleton the reviewer fills with clean/leak.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def collect_files(corpus: Path) -> List[Path]:
    return sorted(
        p.relative_to(corpus)
        for p in corpus.rglob('*')
        if p.is_file() and not p.name.startswith('.'))


def rule_of_three_n(bound: float) -> int:
    """Documents that must ALL be clean to claim residual rate < bound
    at 95% confidence."""
    if not 0 < bound < 1:
        raise ValueError('bound must be in (0, 1)')
    return math.ceil(3.0 / bound)


def stratified_sample(files: List[Path], n: int, seed: int,
                      exclude: Optional[set] = None) -> List[Path]:
    """Proportional allocation by extension, at least one per stratum.

    Extension is the stratum because failure modes track file format
    (the OLE leak class has nothing in common with the raster-PDF
    class); a uniform draw over a spreadsheet-heavy corpus would let a
    rare-but-risky format dodge review entirely.
    """
    exclude = exclude or set()
    pool = [f for f in files if str(f) not in exclude]
    if len(pool) <= n:
        return pool

    strata: Dict[str, List[Path]] = defaultdict(list)
    for f in pool:
        strata[f.suffix.lower() or '<none>'].append(f)

    rng = random.Random(seed)
    # One guaranteed slot per stratum, remainder proportional.
    picks: List[Path] = []
    order = sorted(strata)
    for ext in order:
        picks.append(rng.choice(strata[ext]))
    remaining = n - len(picks)
    if remaining > 0:
        rest = [f for f in pool if f not in set(picks)]
        weights_pool: List[Path] = rest
        picks.extend(rng.sample(weights_pool,
                                min(remaining, len(weights_pool))))
    elif remaining < 0:
        picks = rng.sample(picks, n)
    return sorted(picks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Stratified samples for the human-review gates.')
    parser.add_argument('stage', choices=['precheck', 'audit'])
    parser.add_argument('--corpus', required=True,
                        help='precheck: SOURCE dir. audit: CLEANED dir.')
    parser.add_argument('--project', default=None)
    parser.add_argument('--audit-dir', default=None,
                        help='Where the sample record lands '
                             '(default /tmp/clean_audit/<project>)')
    parser.add_argument('--bound', type=float, default=0.10,
                        help='Residual-rate bound to defend (rule of '
                             'three sets n; default 0.10 -> 30 files)')
    parser.add_argument('--n', type=int, default=None,
                        help='Override the computed sample size')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed (default: random, recorded)')
    parser.add_argument('--stage1-record', default=None,
                        help='audit only: sample_precheck.json from '
                             'stage 1, to enforce disjointness')
    parser.add_argument('--path-manifest', default=None,
                        help='audit only: path_manifest.json mapping '
                             'cleaned names to original paths (flat '
                             'layout writes this automatically)')
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f'ERROR: corpus not found: {corpus}', file=sys.stderr)
        return 2
    project = args.project or corpus.name
    audit = Path(args.audit_dir) if args.audit_dir else (
        Path('/tmp/clean_audit') / project)
    audit.mkdir(parents=True, exist_ok=True)

    n = args.n or rule_of_three_n(args.bound)
    seed = args.seed if args.seed is not None else random.SystemRandom(
        ).randrange(2 ** 31)

    # Disjointness (audit stage): the stage-1 files are named in SOURCE
    # terms; the cleaned corpus is FILE_nnn. The path manifest bridges.
    exclude: set = set()
    if args.stage == 'audit':
        if args.stage1_record and args.path_manifest:
            with open(args.stage1_record) as fh:
                stage1 = set(json.load(fh).get('files', []))
            with open(args.path_manifest) as fh:
                manifest = json.load(fh)  # cleaned name -> original rel
            exclude = {clean for clean, orig in manifest.items()
                       if orig in stage1}
            print(f'Excluding {len(exclude)} cleaned file(s) that were '
                  f'in the stage-1 sample.')
        elif args.stage1_record or args.path_manifest:
            print('WARNING: need BOTH --stage1-record and '
                  '--path-manifest to enforce disjointness; got one. '
                  'Proceeding WITHOUT exclusion — do not reuse this '
                  'sample for a defensible bound.', file=sys.stderr)
        else:
            print('WARNING: no --stage1-record given; disjointness from '
                  'the discovery sample is NOT enforced.',
                  file=sys.stderr)

    files = collect_files(corpus)
    picks = stratified_sample(files, n, seed, exclude)

    record = {
        'stage': args.stage,
        'project': project,
        'corpus': str(corpus),
        'generated': datetime.now(timezone.utc).isoformat(
            timespec='seconds'),
        'seed': seed,
        'bound': args.bound,
        'required_clean': n,
        'population': len(files),
        'excluded_stage1_overlap': len(exclude),
        'files': [str(f) for f in picks],
    }
    if args.stage == 'audit':
        record['instructions'] = (
            'Blind review: open each file, record "clean" or "leak" '
            '(with what leaked). ALL files must be clean to claim '
            f'residual rate < {args.bound:.0%} at 95% confidence. '
            'One leak = the gate fails; fix, re-clean, draw a FRESH '
            'sample (a reviewed sample is spent).')
        record['verdicts'] = {str(f): '' for f in picks}

    out = audit / f'sample_{args.stage}.json'
    with open(out, 'w') as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    by_ext: Dict[str, int] = defaultdict(int)
    for f in picks:
        by_ext[f.suffix.lower() or '<none>'] += 1
    print(f'{args.stage}: {len(picks)} of {len(files)} files '
          f'(bound {args.bound:.0%} -> n={n}, seed {seed})')
    for ext, count in sorted(by_ext.items(), key=lambda kv: -kv[1]):
        print(f'  {count:4d}  {ext}')
    if len(picks) < n:
        print(f'WARNING: population only supports {len(picks)} < {n} — '
              f'the defensible bound is ~{3 / max(len(picks), 1):.0%}, '
              f'not {args.bound:.0%}.', file=sys.stderr)
    print(f'Sample record: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
