#!/usr/bin/env python
"""
Server entrypoint for the cleaning pipeline.

Wires the local Qwen endpoint (OpenAI-compatible, default
http://localhost:8000/v1, model qwen27b) into the full clean run:
deterministic detectors -> LLM discovery loop -> strict path
anonymization -> verification suite incl. the LLM cleanliness gate.

Examples:
    # Clean one project directory with seeded entities
    python run_clean.py \
        --source '/media/sparrows/KINGSTON/Globus Medical - AR Combiner' \
        --project 'AR_Combiner' \
        --seed 'company=Globus Medical' --seed 'product=AR Combiner'

    # Different endpoint/model/key
    python run_clean.py --source DIR --llm-model qwen27b \
        --llm-base http://localhost:8000/v1 --llm-key not-needed

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
    parser.add_argument('--llm', choices=['off', 'auto', 'required'],
                        default='required',
                        help='LLM discovery/verification mode '
                             '(default: required — endpoint down fails the run)')
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
    if args.llm == 'required' and not llm.available():
        print('ERROR: --llm required but the endpoint is unreachable. '
              'Start the Qwen server or pass --llm auto/off.',
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

    pipeline = CleanPipeline(
        project_name=project,
        source_dir=source,
        staging_dir=staging,
        mapper=mapper,
    )
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
    print()
    print(f'Deliverable: {staging}')
    print(f'Quarantine:  {pipeline._quarantine_root() / project}')
    print(f'Audit map:   {pipeline._audit_root() / project} (KEEP PRIVATE)')

    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
