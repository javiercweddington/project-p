#!/usr/bin/env python
"""
Test the cleaning pipeline with ONE representative file per type.

Selects one file of each major type from the Globus Medical project.
"""

import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clean.pipeline import CleanPipeline
from clean.anonymizer import EntityMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('gliner').setLevel(logging.WARNING)

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

def main():
    # Create a test directory with one of each
    test_dir = Path('/tmp/clean_test_one_each')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    print(f"\n{'='*60}")
    print(f"Selecting one file per type from Globus Medical project")
    print(f"{'='*60}\n")

    copied = 0
    for ext, rel_path in SAMPLES.items():
        src = SOURCE_BASE / rel_path
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

    print(f"\n{'='*60}\n")
    return 0 if result.success else 1

if __name__ == '__main__':
    sys.exit(main())