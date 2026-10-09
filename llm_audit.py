#!/usr/bin/env python3
"""Standalone, READ-ONLY LLM audit of a finished cleaned deliverable.

Walks a cleaned directory, extracts text from each file, and asks the
local vLLM judge whether any person/company name survived cleaning. Every
name it finds in an already-cleaned file is a residual leak. Writes the
findings to a report file.

It ONLY reads. Unlike `run_clean`, it never cleans, flattens, moves, or
quarantines anything — so it is safe to run against an already-flattened
deliverable (which `run_clean --resume` cannot be, because it looks up
each file's original in the source by name and FILE_nnn doesn't exist
there). Use this to QC a deliverable after cleaning is done.

Requires vLLM reachable (PROJECT_P_LLM_BASE, default
http://localhost:8000/v1; PROJECT_P_LLM_MODEL for the model name).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cleaned', required=True,
                    help='Finished deliverable directory to audit.')
    ap.add_argument('--mapper', default=None,
                    help='Optional .entity_mapper.json for context (the '
                         'judge works without it, but it can help).')
    ap.add_argument('--report', default=None,
                    help='Where to write findings (default: '
                         '<cleaned>_llm_audit.txt alongside the dir).')
    args = ap.parse_args()

    # Force JUDGE mode so the judge actually audits — in 'off'/'sample'
    # mode run_check early-returns without scanning.
    os.environ['PROJECT_P_LLM_VERIFY'] = 'judge'

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from clean.llm_detect import LocalLLM, LLMCleanlinessJudge
    from clean.anonymizer import EntityMapper

    cleaned = Path(args.cleaned).resolve()
    if not cleaned.is_dir():
        print(f"ERROR: {cleaned} is not a directory.", file=sys.stderr)
        return 2

    mapper = EntityMapper()
    if args.mapper and Path(args.mapper).is_file():
        try:
            mapper = EntityMapper.from_dict(
                json.loads(Path(args.mapper).read_text()))
        except Exception as e:
            print(f"(could not load mapper {args.mapper}: {e} — "
                  f"auditing without it)", file=sys.stderr)

    llm = LocalLLM()
    if not llm.available():
        print(f"ERROR: LLM endpoint {llm.base_url} is unreachable. "
              f"Start vLLM (or set PROJECT_P_LLM_BASE) and retry.",
              file=sys.stderr)
        return 2

    print(f"READ-ONLY LLM audit of {cleaned}", file=sys.stderr)
    print(f"  via {llm.base_url}  model={llm.model}", file=sys.stderr)
    judge = LLMCleanlinessJudge(mapper, llm)
    result = judge.run_check(cleaned)

    hits = list(getattr(result, 'hits', []) or [])
    report_path = (Path(args.report) if args.report
                   else cleaned.with_name(cleaned.name + '_llm_audit.txt'))

    lines = [
        f"LLM audit of {cleaned}",
        result.details or '',
        f"PASSED: {result.passed}   findings: {len(hits)}",
        '',
    ]
    # Group findings by file for a readable report.
    by_file: dict[str, list] = {}
    for h in hits:
        by_file.setdefault(getattr(h, 'file_path', '?'), []).append(h)
    for fp in sorted(by_file):
        lines.append(fp)
        for h in by_file[fp]:
            lines.append(
                f"    [{getattr(h, 'entity_type', '?')}] "
                f"{getattr(h, 'original', '?')!r}")
    report_path.write_text('\n'.join(lines))

    # Console summary.
    print(f"\n{'PASS' if result.passed else 'FINDINGS'}: "
          f"{len(hits)} residual name(s) across {len(by_file)} file(s)")
    for fp in sorted(by_file)[:15]:
        vals = ', '.join(
            repr(getattr(h, 'original', '?')) for h in by_file[fp][:5])
        print(f"  {fp}: {vals}")
    if len(by_file) > 15:
        print(f"  ... (+{len(by_file) - 15} more files)")
    print(f"\nFull report: {report_path}")
    print(f"READ-ONLY — nothing in {cleaned} was modified.")
    return 0 if result.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
