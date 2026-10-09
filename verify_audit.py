#!/usr/bin/env python3
"""Standalone, READ-ONLY deterministic QC of a cleaned deliverable.

Scans every file in a cleaned directory for any surviving MAPPED entity
(the person/company/email originals in .entity_mapper.json). This is the
real deliverable gate: it catches a file that didn't get its known names
removed. CPU-only — no GPU, no LLM, no vLLM. Read-only: it never moves,
modifies, flattens, or quarantines anything, so it is safe on an
already-flattened deliverable.

(The LLM audit, llm_audit.py, is the complementary net for names that
were never in the mapper; this one covers everything the cleaner knew
about, and needs no GPU.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cleaned', required=True,
                    help='Finished deliverable directory to audit.')
    ap.add_argument('--mapper', required=True,
                    help='.entity_mapper.json for this run.')
    ap.add_argument('--report', default=None,
                    help='Where to write findings (default: '
                         '<cleaned>_verify_audit.txt).')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from clean.verifier import LeakageChecker
    from clean.anonymizer import EntityMapper

    cleaned = Path(args.cleaned).resolve()
    if not cleaned.is_dir():
        print(f"ERROR: {cleaned} is not a directory.", file=sys.stderr)
        return 2

    mapper = EntityMapper.from_dict(
        json.loads(Path(args.mapper).read_text()))
    print(f"READ-ONLY deterministic audit of {cleaned} "
          f"({mapper.mapping_count} mapped entities) — CPU only, no GPU.",
          file=sys.stderr)

    checker = LeakageChecker(mapper)
    # check_file ignores the original path (it only scans the CLEANED text
    # for mapped entities), so a placeholder original dir is fine — and
    # necessary here, because the flattened FILE_nnn names have no match
    # in the real source anyway.
    result = checker.run_check(cleaned, Path('/nonexistent-original'))

    hits = list(getattr(result, 'hits', []) or [])
    report = (Path(args.report) if args.report
              else cleaned.with_name(cleaned.name + '_verify_audit.txt'))

    by_file: dict[str, list] = {}
    for h in hits:
        by_file.setdefault(getattr(h, 'file_path', '?'), []).append(h)

    lines = [
        f"Deterministic leakage audit of {cleaned}",
        f"PASSED: {result.passed}   residual mapped-name hits: {len(hits)}",
        '',
    ]
    for fp in sorted(by_file):
        lines.append(fp)
        for h in by_file[fp]:
            lines.append(
                f"    [{getattr(h, 'entity_type', '?')}] "
                f"{getattr(h, 'original', '?')!r}  "
                f"ctx={getattr(h, 'context', '')!r}")
    report.write_text('\n'.join(lines))

    print(f"\n{'PASS' if result.passed else 'FINDINGS'}: "
          f"{len(hits)} residual mapped name(s) across {len(by_file)} file(s)")
    for fp in sorted(by_file)[:15]:
        vals = ', '.join(
            repr(getattr(h, 'original', '?')) for h in by_file[fp][:5])
        print(f"  {fp}: {vals}")
    if len(by_file) > 15:
        print(f"  ... (+{len(by_file) - 15} more files)")
    print(f"\nFull report: {report}")
    print(f"READ-ONLY — nothing in {cleaned} was modified.")
    return 0 if result.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
