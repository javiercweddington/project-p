#!/usr/bin/env python
"""
Test the cleaning pipeline on real project data.

Usage:
    python test_real_data.py [source_directory] [project_name]

Defaults to: /media/sparrows/KINGSTON/Globus Medical - AR Combiner/
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from clean.pipeline import CleanPipeline, CleanResult
from clean.anonymizer import EntityMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

# Reduce noise from third-party libraries
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def count_file_types(source_dir: Path) -> dict:
    """Count files by extension in the source directory."""
    counts = {}
    for f in source_dir.rglob('*'):
        if f.is_file() and not f.name.startswith('._'):
            ext = f.suffix.lower() or '(no extension)'
            counts[ext] = counts.get(ext, 0) + 1
    return counts

def main():
    # Parse arguments
    if len(sys.argv) > 1:
        source_dir = Path(sys.argv[1])
    else:
        source_dir = Path('/media/sparrows/KINGSTON/Globus Medical - AR Combiner/')

    if len(sys.argv) > 2:
        project_name = sys.argv[2]
    else:
        project_name = 'Globus_AR_Combiner_Test'

    # Validate source
    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Count file types
    print(f"\n{'='*60}")
    print(f"Source: {source_dir}")
    print(f"Project: {project_name}")
    print(f"{'='*60}")

    counts = count_file_types(source_dir)
    print(f"\nFile type distribution ({sum(counts.values())} total):")
    for ext, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ext:12s}: {count:4d}")

    # Create entity mapper with some known entities from this project
    mapper = EntityMapper()

    # Pre-populate with entities visible in file names
    known_entities = [
        ('company', 'Globus Medical'),
        ('product', 'AR Combiner'),
        ('product', 'GHAP'),
        ('product', 'GNS'),
    ]

    for entity_type, value in known_entities:
        mapper.get_or_create(entity_type=entity_type, value=value, source='test')

    print(f"\nEntity mapper: {len(mapper.mappings)} known entities")

    # Create pipeline
    staging_dir = Path('/tmp/clean') / project_name
    pipeline = CleanPipeline(
        project_name=project_name,
        source_dir=source_dir,
        staging_dir=staging_dir,
        mapper=mapper,
    )

    # Run pipeline
    print(f"\nStarting cleaning pipeline...")
    print(f"Staging to: {staging_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = pipeline.run()
    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(result.summary())

    # Print diff report if available
    if result.diff_report:
        print(f"\n--- Diff Report ---")
        print(f"  Files tracked: {result.diff_report.files_tracked}")
        print(f"  Files changed: {result.diff_report.files_changed}")
        print(f"  Files unchanged: {result.diff_report.files_unchanged}")
        print(f"  Total changes: {result.diff_report.total_changes}")

    # Print leakage report if available
    if result.leakage_report:
        print(f"\n--- Leakage Report ---")
        print(f"  Total leakages: {result.leakage_report.total_leakages}")
        print(f"  Failed checks: {len(result.leakage_report.failed_checks)}")
        print(f"  Passed checks: {len(result.leakage_report.passed_checks)}")
        if result.leakage_report.failed_checks:
            print(f"  Failures:")
            for check in result.leakage_report.failed_checks[:10]:
                print(f"    - {check}")

    # Print errors
    if result.errors:
        print(f"\n--- Errors ---")
        for error in result.errors:
            print(f"  ! {error}")

    print(f"\n{'='*60}\n")
    return 0 if result.success else 1

if __name__ == '__main__':
    sys.exit(main())