#!/usr/bin/env python
"""
Server entrypoint for the cleaning pipeline.

Default flow (--mode onepass, --llm off): up-front discovery
(deterministic detectors + GLiNER NER + CV/OCR) -> ONE clean pass
(PDFs and legacy .ppt rasterize to image-only PDFs; Office XML gets a
raw-member catch-all incl. cross-run splices) -> strict path
anonymization -> verification suite -> review_candidates.json (auto
entities + GLiNER suggestions) for the human-in-the-loop correction
cycle (edit it, re-run with --seed-file).

LLM modes (local Qwen via vLLM, structured outputs auto-negotiated):
  --llm sample   audit ONE cleaned file per type, register + fix its
                 findings everywhere, re-check (O(types) LLM cost —
                 the recommended accuracy net)
  --llm judge    audit EVERY finished file once (no LLM discovery)
  --llm auto/required   LLM also joins up-front discovery
Chunk requests run concurrently (PROJECT_P_LLM_CONCURRENCY, default 16).

Examples:
    # Clean one project directory with seeded entities (no LLM)
    python run_clean.py \
        --source '/media/sparrows/KINGSTON/Globus Medical - AR Combiner' \
        --project 'AR_Combiner' \
        --seed 'company=Globus Medical' --seed 'product=AR Combiner'

    # Correction pass after human review of the candidates file
    python run_clean.py --source DIR \
        --seed-file /tmp/clean_audit/AR_Combiner/review_candidates.json

    # With the LLM discovery + cleanliness gate enabled
    python run_clean.py --source DIR --llm required --llm-model qwen27b

Exit code 0 only when every file cleaned AND all verification checks
(including the LLM gate) passed. Failed/unverifiable files are moved to
{staging_parent}_quarantine/; the placeholder->original audit map is
written to {staging_parent}_audit/ (keep it private).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pseudo-anonymize a project directory.')
    parser.add_argument('--source', required=True,
                        help='Directory with the original project files')
    parser.add_argument('--project', default=None,
                        help='Project name (default: source directory name)')
    parser.add_argument('--staging', default=None,
                        help='Output directory (default: /tmp/clean/<project>)')
    parser.add_argument('--seed', action='append', default=[],
                        metavar='TYPE=VALUE',
                        help="Seed entity, e.g. --seed 'company=Acme Corp' "
                             "(repeatable)")
    parser.add_argument('--seed-file', default=None, metavar='PATH',
                        help='JSON file of entities to seed (the '
                             'review_candidates.json a previous run wrote, '
                             'after human editing, or any JSON with an '
                             '"entities" list of {"type","value"} objects)')
    parser.add_argument('--mode', choices=['onepass', 'iterative'],
                        default='onepass',
                        help='onepass (default): discover up front, clean '
                             'once, verify once — no retroactive re-cleans. '
                             'iterative: legacy loop (re-clean until the '
                             'mapper stops growing).')
    parser.add_argument('--llm',
                        choices=['off', 'auto', 'required', 'judge',
                                 'sample'],
                        default='off',
                        help='LLM mode (default: off — deterministic + CV + '
                             'human review carry detection). sample: LLM '
                             'audits ONE cleaned file per type, fixes its '
                             'findings everywhere, re-checks (O(types) LLM '
                             'cost); judge: LLM audits EVERY finished file '
                             'once; auto/required: LLM also joins up-front '
                             'discovery.')
    parser.add_argument('--opaque-binary',
                        choices=['quarantine', 'ship-scanned'],
                        default='quarantine',
                        help='Non-OLE binary CAD (e.g. newer SolidWorks): '
                             'quarantine (default, fail-closed) or '
                             'ship-scanned (same-length entity surgery + '
                             'raw-byte verify + embedded-image OCR gate; '
                             'compressed streams are NOT scannable — '
                             'accepts that residual risk).')
    parser.add_argument('--llm-base', default=None,
                        help='OpenAI-compatible base URL '
                             '(default http://localhost:8000/v1)')
    parser.add_argument('--llm-model', default=None,
                        help='Served model name (default qwen27b)')
    parser.add_argument('--llm-key', default=None,
                        help='API key for the endpoint (default not-needed)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Environment wiring BEFORE importing the pipeline (modules read these
    # at import/instantiation time)
    os.environ['PROJECT_P_LLM_VERIFY'] = args.llm
    os.environ['PROJECT_P_OPAQUE_BINARY'] = args.opaque_binary
    if args.llm_base:
        os.environ['PROJECT_P_LLM_BASE'] = args.llm_base
    if args.llm_model:
        os.environ['PROJECT_P_LLM_MODEL'] = args.llm_model
    if args.llm_key:
        os.environ['PROJECT_P_LLM_API_KEY'] = args.llm_key

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname).1s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    for noisy in ('PIL', 'pypdf', 'huggingface_hub', 'gliner'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from clean.pipeline import CleanPipeline
    from clean.anonymizer import EntityMapper
    from clean.llm_detect import LocalLLM

    source = Path(args.source)
    if not source.is_dir():
        print(f'ERROR: source directory not found: {source}', file=sys.stderr)
        return 2
    project = args.project or source.name
    staging = Path(args.staging) if args.staging else (
        Path('/tmp/clean') / project)

    llm = LocalLLM()
    print(f'LLM endpoint: {llm.base_url} (model {llm.model}) — '
          f'{"reachable" if llm.available() else "NOT reachable"} '
          f'[mode={args.llm}]')
    if args.llm in ('required', 'judge', 'sample') and not llm.available():
        print(f'ERROR: --llm {args.llm} but the endpoint is unreachable. '
              f'Start the Qwen server or pass --llm auto/off.',
              file=sys.stderr)
        return 2

    mapper = EntityMapper()
    for seed in args.seed:
        if '=' not in seed:
            print(f'ERROR: bad --seed {seed!r} (expected TYPE=VALUE)',
                  file=sys.stderr)
            return 2
        entity_type, value = seed.split('=', 1)
        placeholder = mapper.get_or_create(
            entity_type.strip(), value.strip(), source='seed')
        print(f'Seeded: {value.strip()!r} -> {placeholder}')

    if args.seed_file:
        import json
        seed_path = Path(args.seed_file)
        if not seed_path.is_file():
            print(f'ERROR: --seed-file not found: {seed_path}',
                  file=sys.stderr)
            return 2
        with open(seed_path) as f:
            payload = json.load(f)
        entries = payload.get('entities', payload) if isinstance(
            payload, dict) else payload
        seeded = 0
        for entry in entries:
            entity_type = str(entry.get('type', '')).strip()
            value = str(entry.get('value', '')).strip()
            if not entity_type or not value:
                continue
            mapper.get_or_create(entity_type, value,
                                 source=f'seed_file:{seed_path.name}')
            seeded += 1
        print(f'Seeded {seeded} entities from {seed_path}')

    pipeline = CleanPipeline(
        project_name=project,
        source_dir=source,
        staging_dir=staging,
        mapper=mapper,
        one_pass=(args.mode == 'onepass'),
    )
    print(f'Mode: {args.mode}')
    result = pipeline.run()

    print()
    print('=' * 64)
    print(result.summary())
    print('=' * 64)
    if result.leakage_report:
        for check in result.leakage_report.results:
            mark = 'PASS' if check.passed else 'FAIL'
            print(f'  [{mark}] {check.check_name}: {check.details[:90]}')
            for hit in check.hits[:5]:
                print(f'        - {hit.original!r} in {hit.file_path}')
    # Human-in-the-loop review file: everything that was replaced plus
    # every verification hit, in a format --seed-file accepts back after
    # editing (delete wrong entries, add missed ones, re-run).
    import json
    review = {
        'project': project,
        # GLiNER hits below the auto-register threshold: human triage —
        # move real ones into 'entities' and re-run with --seed-file.
        'suggested_entities': getattr(pipeline, 'suggested_entities', []),
        'entities': [
            {
                'type': m.entity_type,
                'value': m.original,
                'placeholder': m.placeholder,
                'source': getattr(m, 'source', ''),
            }
            for m in mapper.mappings
        ],
        'verification_failures': [
            {
                'check': check.check_name,
                'details': check.details,
                'hits': [
                    {'file': hit.file_path,
                     'type': hit.entity_type,
                     'value': hit.original}
                    for hit in check.hits
                ],
            }
            for check in (result.leakage_report.failed_checks
                          if result.leakage_report else [])
        ],
    }
    review_path = pipeline._audit_root() / project / 'review_candidates.json'
    try:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with open(review_path, 'w') as f:
            json.dump(review, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f'WARNING: could not write review file: {e}', file=sys.stderr)
        review_path = None

    print()
    print(f'Deliverable: {staging}')
    print(f'Quarantine:  {pipeline._quarantine_root() / project}')
    print(f'Audit map:   {pipeline._audit_root() / project} (KEEP PRIVATE)')
    if review_path:
        print(f'Review file: {review_path}')
        print('  Edit it (drop wrong entries, add missed ones), then '
              're-run with --seed-file to apply corrections.')

    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
