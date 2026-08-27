#!/usr/bin/env python
"""
Test the cleaning pipeline with ONE representative file per type,
with per-step timing instrumentation to locate choke points.

Selects one file of each major type from the Globus Medical project,
runs the full pipeline (same path as run_clean.py, including the LLM
discovery loop and cleanliness judge), and prints a timing profile:

  - Pipeline stages: copy, clean passes, LLM discovery, path
    anonymization, verification, LLM judge (roughly additive).
  - Hot functions: individual llm.chat calls, per-extension
    clean_file / text extraction, tesseract OCR calls, GLiNER
    load + inference. These NEST inside the stages, so their
    totals overlap with (and explain) the stage totals.

Raw durations are also dumped to /tmp/clean_test_one_each_timing.json
so the numbers can be shared/compared across runs.

Usage (on the server, with the Qwen endpoint up):
    python test_one_each.py
    python test_one_each.py --llm auto          # tolerate endpoint down
    python test_one_each.py --llm-model qwen26b
"""

import argparse
import functools
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SOURCE_BASE = Path('/media/sparrows/KINGSTON/Globus Medical - AR Combiner')

# One representative file per type
SAMPLES = {
    # PDF - a quotation with incremental updates
    'pdf': '5-Quotation For Customer/JCW20191226A QUOTATION - Globus Medical - Ar Combiner.pdf',

    # XLSX - a supplier quotation
    'xlsx': '9-Quotation From Supplier/20210313【六】15点15分06秒---诺尔信息-m1【注塑报价单】.xlsx',

    # XLSM - a macro-enabled invoice (tests pivot cache, external links)
    'xlsm': '6-Invoice For Customer/JCW20200615 INVOICE - Globus Medical - AR Combiner.xlsm',

    # DOCX - a supplier quotation doc
    'docx': '9-Quotation From Supplier/SIMPLIFIED OPTICAL COMBINER -new产品（阶梯）报价单2020-6-8.docx',

    # PPT - legacy format (only one exists)
    'ppt': 'E-Shared with Client/6-Development Phase Deliverables/20201010 Lens Planned Adjustments 镜片调整.ppt',

    # JPG - a WeChat sample photo (has EXIF)
    'jpg': 'E-Shared with Client/6-Development Phase Deliverables/20200428 Early Samples/WeChat Image_20200428100001.jpg',

    # PNG - a payment receipt screenshot
    'png': '6-Invoice For Customer/JCW20200803F DEPOSIT INVOICE - Payment Receipt.png',

    # STEP - a CAD file
    'step': 'E-Shared with Client/0-Requirement/XR Hipbox.STEP',

    # SLDPRT - SolidWorks part
    'sldprt': 'E-Shared with Client/0-Requirement/20200221 Official PO Files/6203.4000.0186 Rev 2 - Optical Combiner, GNS.SLDPRT',
}

TIMING_JSON = Path('/tmp/clean_test_one_each_timing.json')

# label -> list of durations (seconds). Filled by the probe wrappers.
STATS = defaultdict(list)


def _timed(label):
    """Decorator: record wall-clock of every call under `label`."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                STATS[label].append(time.perf_counter() - t0)
        return wrapper
    return deco


def install_probes():
    """Wrap pipeline internals with timers.

    Must be called AFTER the PROJECT_P_LLM_* env vars are set (llm_detect
    reads them at import time) and BEFORE the pipeline is constructed.
    Only this test is affected — production code is untouched.
    """
    from clean import pipeline as pl
    from clean import llm_detect as ld
    from clean.cleaners import router as rt

    # --- Pipeline stages (roughly additive; top-level view) ---
    for name in ('_copy_to_staging', '_clean_all_files',
                 '_llm_discovery_loop', '_inplace_reclean',
                 '_anonymize_paths', '_normalize_all_mtimes',
                 '_save_mapper'):
        setattr(pl.CleanPipeline, name,
                _timed(f'stage:{name.lstrip("_")}')(
                    getattr(pl.CleanPipeline, name)))

    # pipeline.run() resolves these names from its module globals /
    # class attributes at call time, so rebinding here takes effect.
    pl.verify_clean = _timed('stage:verify_clean')(pl.verify_clean)
    ld.LLMCleanlinessJudge.run_check = _timed('stage:llm_judge')(
        ld.LLMCleanlinessJudge.run_check)
    ld.LLMEntityDetector.scan_directory = _timed('stage:llm_scan_directory')(
        ld.LLMEntityDetector.scan_directory)

    # --- Hot functions (nested inside the stages above) ---
    ld.LocalLLM.chat = _timed('llm.chat')(ld.LocalLLM.chat)

    orig_clean_file = rt.FileCleanerRouter.clean_file

    @functools.wraps(orig_clean_file)
    def clean_file(self, input_path, output_path, entity_spans=None):
        label = f'clean_file[{input_path.suffix.lower() or input_path.name}]'
        t0 = time.perf_counter()
        try:
            return orig_clean_file(self, input_path, output_path,
                                   entity_spans)
        finally:
            STATS[label].append(time.perf_counter() - t0)
    rt.FileCleanerRouter.clean_file = clean_file

    orig_extract = ld.extract_scannable_text

    @functools.wraps(orig_extract)
    def extract_scannable_text(path, max_chars=200_000):
        label = f'extract[{path.suffix.lower() or path.name}]'
        t0 = time.perf_counter()
        try:
            return orig_extract(path, max_chars)
        finally:
            STATS[label].append(time.perf_counter() - t0)
    ld.extract_scannable_text = extract_scannable_text

    # Tesseract: acquire.metadata and clean/cleaners/image.py both call
    # through the pytesseract module attributes at call time, so patching
    # the module covers every OCR call in the run.
    try:
        import pytesseract
        pytesseract.image_to_string = _timed('ocr.image_to_string')(
            pytesseract.image_to_string)
        pytesseract.image_to_data = _timed('ocr.image_to_data')(
            pytesseract.image_to_data)
    except ImportError:
        pass

    # GLiNER: separate the one-time model load from per-image inference.
    try:
        from acquire import catalog as cat
        if hasattr(cat, '_get_gliner_model'):
            cat._get_gliner_model = _timed('gliner.get_model')(
                cat._get_gliner_model)
        if hasattr(cat, '_extract_entities_with_gliner'):
            cat._extract_entities_with_gliner = _timed('gliner.extract')(
                cat._extract_entities_with_gliner)
    except ImportError:
        pass


def print_report(wall_time):
    def rows_for(labels):
        rows = []
        for label in labels:
            durations = STATS[label]
            total = sum(durations)
            rows.append((label, len(durations), total,
                         total / len(durations), 100.0 * total / wall_time))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows

    def print_table(title, labels):
        if not labels:
            return
        print(f'\n--- {title} ---')
        print(f'{"label":<34s} {"calls":>5s} {"total":>9s} '
              f'{"mean":>8s} {"%wall":>6s}')
        for label, calls, total, mean, share in rows_for(labels):
            print(f'{label:<34s} {calls:>5d} {total:>8.2f}s '
                  f'{mean:>7.3f}s {share:>5.1f}%')

    stage_labels = [k for k in STATS if k.startswith('stage:')]
    hot_labels = [k for k in STATS if not k.startswith('stage:')]

    print(f'\n{"=" * 64}')
    print(f'TIMING PROFILE (wall: {wall_time:.1f}s)')
    print(f'{"=" * 64}')
    print_table('Pipeline stages (roughly additive)', stage_labels)
    print_table('Hot functions (nested inside stages — totals overlap)',
                hot_labels)

    accounted = sum(sum(STATS[k]) for k in stage_labels)
    print(f'\nStage-accounted: {accounted:.1f}s of {wall_time:.1f}s wall '
          f'({100.0 * accounted / wall_time:.0f}% — remainder is '
          f'imports/model loads outside stages)')

    try:
        with open(TIMING_JSON, 'w') as f:
            json.dump({'wall_time': wall_time,
                       'durations': {k: v for k, v in STATS.items()}},
                      f, indent=2)
        print(f'Raw durations written to {TIMING_JSON}')
    except OSError as e:
        print(f'Could not write timing JSON: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='One-file-per-type pipeline run with timing profile.')
    parser.add_argument('--source', default=str(SOURCE_BASE),
                        help='Project directory to sample from')
    parser.add_argument('--llm', choices=['off', 'auto', 'required'],
                        default='required',
                        help='LLM mode (default required, matching '
                             'run_clean.py — the LLM steps are the prime '
                             'profiling target)')
    parser.add_argument('--llm-base', default=None)
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--llm-key', default=None)
    args = parser.parse_args()

    # Env wiring BEFORE importing clean modules (same reason as
    # run_clean.py: llm_detect reads these at import time).
    os.environ['PROJECT_P_LLM_VERIFY'] = args.llm
    if args.llm_base:
        os.environ['PROJECT_P_LLM_BASE'] = args.llm_base
    if args.llm_model:
        os.environ['PROJECT_P_LLM_MODEL'] = args.llm_model
    if args.llm_key:
        os.environ['PROJECT_P_LLM_API_KEY'] = args.llm_key

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    for noisy in ('PIL', 'pypdf', 'huggingface_hub', 'gliner'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    install_probes()

    from clean.pipeline import CleanPipeline
    from clean.anonymizer import EntityMapper
    from clean.llm_detect import LocalLLM

    llm = LocalLLM()
    print(f'LLM endpoint: {llm.base_url} (model {llm.model}) — '
          f'{"reachable" if llm.available() else "NOT reachable"} '
          f'[mode={args.llm}]')
    if args.llm == 'required' and not llm.available():
        print('ERROR: --llm required but the endpoint is unreachable. '
              'Start the Qwen server or pass --llm auto/off.',
              file=sys.stderr)
        return 2

    # Create a test directory with one of each
    source_base = Path(args.source)
    test_dir = Path('/tmp/clean_test_one_each')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    print(f"\n{'='*60}")
    print(f"Selecting one file per type from {source_base.name}")
    print(f"{'='*60}\n")

    copied = 0
    for ext, rel_path in SAMPLES.items():
        src = source_base / rel_path
        if not src.exists():
            print(f"  SKIP {ext:6s}: {rel_path} (not found)")
            continue

        dst = test_dir / src.name
        shutil.copy2(src, dst)
        size = src.stat().st_size
        print(f"  COPY {ext:6s}: {src.name[:50]:50s} ({size:>10,} bytes)")
        copied += 1

    if copied == 0:
        print("ERROR: No files copied!")
        return 1

    # Create entity mapper
    mapper = EntityMapper()
    for entity_type, value in [('company', 'Globus Medical'), ('product', 'AR Combiner')]:
        mapper.get_or_create(entity_type=entity_type, value=value, source='test')

    # Run pipeline
    project_name = 'OneEach_Test'
    staging_dir = Path('/tmp/clean') / project_name

    pipeline = CleanPipeline(
        project_name=project_name,
        source_dir=test_dir,
        staging_dir=staging_dir,
        mapper=mapper,
    )

    print(f"\n{'='*60}")
    print(f"Running pipeline on {copied} files...")
    print(f"{'='*60}\n")

    start = time.time()
    result = pipeline.run()
    elapsed = time.time() - start

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(result.summary())

    if result.diff_report:
        print(f"\n--- Diff Report ---")
        print(f"  Total files:      {result.diff_report.total_files}")
        print(f"  Files w/ changes: {result.diff_report.files_with_changes}")
        print(f"  Total changes:    {result.diff_report.total_changes}")

    if result.leakage_report:
        print(f"\n--- Leakage Report ---")
        print(f"  Total leakages:  {result.leakage_report.total_leakages}")
        print(f"  Failed checks:   {len(result.leakage_report.failed_checks)}")
        passed = len(result.leakage_report.results) - len(result.leakage_report.failed_checks)
        print(f"  Passed checks:   {passed}")
        if result.leakage_report.failed_checks:
            for check in result.leakage_report.failed_checks[:10]:
                print(f"    - {check.check_name}: {check.details}")
                for hit in check.hits[:5]:
                    print(f"      * {hit.original!r} in {hit.file_path}")

    if result.errors:
        print(f"\n--- Errors ---")
        for error in result.errors:
            print(f"  ! {error}")

    print_report(elapsed)

    print(f"\n{'='*60}\n")
    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
