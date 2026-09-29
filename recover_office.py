#!/usr/bin/env python3
"""Recover Office files broken by the pre-fix XML catch-all pass.

Background
----------
`clean/cleaners/xml_pass.py` used to run a blunt text-replace over EVERY
XML member of an Office zip, including OPC plumbing (`*.rels`,
`[Content_Types].xml`). When a discovered entity string collided with a
reserved attribute NAME there (e.g. a company named "Target" vs the
`Target="..."` relationship attribute), the replacement produced
malformed XML — `[COMPANY_510]="..."` — and the whole workbook became
unopenable (blank in Excel / VS Code).

That produced TWO kinds of broken Office files:
  1. corrupt-but-shipped  — sitting in the cleaned tree, unreadable,
     recorded in the resume checkpoint as "done".
  2. quarantined          — a later fail-closed guard moved the original
     OUT of the cleaned tree into the quarantine dir; it is therefore
     MISSING from cleaned and NOT in the checkpoint.

The code fix (skip plumbing + revert only the broken member) prevents
both going forward. This tool repairs what the buggy run already
produced so `--resume` can finish the job.

What this does
--------------
For every Office file (.xlsx/.xlsm/.docx/.pptx) in --source, it checks
the matching path under --cleaned and, if that copy is MISSING or fails
to XML-parse (corrupt), restores the pristine original into the cleaned/
staging tree and drops its line from the resume checkpoint. Files that
cleaned fine are left untouched.

Originals in --source are read-only inputs; this only writes into the
cleaned/staging tree and the checkpoint (backed up first). Run without
--apply to see the scope; add --apply to act.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

OFFICE_EXTS = ('.xlsx', '.xlsm', '.docx', '.pptx')
XML_SUFFIXES = ('.xml', '.rels', '.vml')


def is_corrupt(path: Path) -> bool:
    """True if any XML member of the Office zip fails to parse."""
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith(XML_SUFFIXES):
                    ET.fromstring(z.read(name))
        return False
    except Exception:
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cleaned', required=True,
                    help='Root of the cleaned/staging output tree.')
    ap.add_argument('--source', required=True,
                    help="Root of the pristine originals (the run's "
                         "--source dir).")
    ap.add_argument('--checkpoint', required=True,
                    help='Path to .clean_checkpoint.tsv for this run.')
    ap.add_argument('--apply', action='store_true',
                    help='Actually restore originals and edit the '
                         'checkpoint. Default is a dry run.')
    args = ap.parse_args()

    cleaned = Path(args.cleaned).resolve()
    source = Path(args.source).resolve()
    checkpoint = Path(args.checkpoint)

    total = 0
    missing: list[str] = []   # rel paths absent from cleaned (quarantined)
    corrupt: list[str] = []   # rel paths present but unreadable
    fine = 0
    for src in source.rglob('*'):
        if not (src.is_file() and src.suffix.lower() in OFFICE_EXTS):
            continue
        total += 1
        rel = os.path.relpath(src, source)
        dst = cleaned / rel
        if not dst.exists():
            missing.append(rel)
        elif is_corrupt(dst):
            corrupt.append(rel)
        else:
            fine += 1

    to_restore = missing + corrupt
    print(f"Office files in source: {total}")
    print(f"  cleaned OK (leave alone): {fine}")
    print(f"  MISSING from cleaned (quarantined): {len(missing)}")
    print(f"  CORRUPT in cleaned:                 {len(corrupt)}")
    print(f"  -> to restore + re-clean:           {len(to_restore)}")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to recover.")
        return 0

    # Restore pristine originals into the staging/cleaned tree.
    for rel in to_restore:
        dst = cleaned / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
    print(f"\nRestored {len(to_restore)} original(s) into the cleaned tree.")

    # Drop restored files from the checkpoint so --resume re-cleans them.
    restore_set = set(to_restore)
    if checkpoint.is_file() and restore_set:
        backup = checkpoint.with_name(
            f'{checkpoint.name}.bak.{int(time.time())}')
        shutil.copy2(checkpoint, backup)
        kept, dropped = [], 0
        with open(checkpoint) as fh:
            for line in fh:
                rel = line.rstrip('\n')
                if rel in restore_set:
                    dropped += 1
                else:
                    kept.append(line if line.endswith('\n') else line + '\n')
        with open(checkpoint, 'w') as fh:
            fh.writelines(kept)
        print(f"Checkpoint: dropped {dropped} line(s) "
              f"(backup at {backup.name}).")

    print("\nDone. Relaunch with --resume to re-clean the restored files "
          "and finish the backlog.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
