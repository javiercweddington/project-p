#!/usr/bin/env python3
"""Recover Office files corrupted by the pre-fix XML catch-all pass.

Background
----------
`clean/cleaners/xml_pass.py` used to run a blunt text-replace over EVERY
XML member of an Office zip, including OPC plumbing (`*.rels`,
`[Content_Types].xml`). When a discovered entity string collided with a
reserved attribute NAME there (e.g. a company named "Target" vs the
`Target="..."` relationship attribute), the replacement produced
malformed XML — `[COMPANY_510]="..."` — and the whole workbook became
unopenable (blank in Excel / VS Code).

The fix (skip plumbing + validate well-formedness) prevents new
corruption, but files already produced by the buggy run are broken AND
checkpointed as done, so a plain `--resume` would skip them.

What this does
--------------
1. Scans the cleaned/output tree for zip-based Office files
   (.xlsx/.xlsm/.docx/.pptx).
2. Flags a file as corrupt if ANY of its XML members fails to parse
   (definitive test for this bug class — not a heuristic).
3. For each corrupt file: restores the pristine original from --source
   back into the staging/output tree, and removes its line from the
   resume checkpoint so `--resume` re-cleans it (now with the fixed code).

Originals are read-only inputs; this only overwrites already-broken
outputs. Run without --apply first to see the scope; add --apply to act.
A timestamped checkpoint backup is written before any edit.
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
                    help='Root of the pristine originals (the run\'s '
                         '--source dir).')
    ap.add_argument('--checkpoint', required=True,
                    help='Path to .clean_checkpoint.tsv for this run.')
    ap.add_argument('--apply', action='store_true',
                    help='Actually restore originals and edit the '
                         'checkpoint. Default is a dry run.')
    args = ap.parse_args()

    cleaned = Path(args.cleaned).resolve()
    source = Path(args.source).resolve()
    checkpoint = Path(args.checkpoint)

    corrupt: list[Path] = []
    total = 0
    for f in cleaned.rglob('*'):
        if f.is_file() and f.suffix.lower() in OFFICE_EXTS:
            total += 1
            if is_corrupt(f):
                corrupt.append(f)

    print(f"Office files scanned: {total}")
    print(f"Corrupt (need recovery): {len(corrupt)}")

    restorable: list[tuple[Path, Path, str]] = []  # (dst, src, rel)
    missing_src: list[Path] = []
    for f in corrupt:
        rel = os.path.relpath(f, cleaned)
        src = source / rel
        if src.is_file():
            restorable.append((f, src, rel))
        else:
            missing_src.append(f)

    print(f"  - originals found (recoverable): {len(restorable)}")
    print(f"  - originals MISSING (manual):    {len(missing_src)}")
    for f in missing_src[:20]:
        print(f"      MISSING SRC: {f}")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to recover.")
        return 0

    # Restore pristine originals over the corrupt outputs.
    restored_rels = set()
    for dst, src, rel in restorable:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored_rels.add(rel)
    print(f"\nRestored {len(restored_rels)} original(s) into the output tree.")

    # Drop restored files from the checkpoint so --resume re-cleans them.
    if checkpoint.is_file() and restored_rels:
        backup = checkpoint.with_name(
            f'{checkpoint.name}.bak.{int(time.time())}')
        shutil.copy2(checkpoint, backup)
        kept = []
        dropped = 0
        with open(checkpoint) as fh:
            for line in fh:
                rel = line.rstrip('\n')
                if rel in restored_rels:
                    dropped += 1
                else:
                    kept.append(line if line.endswith('\n') else line + '\n')
        with open(checkpoint, 'w') as fh:
            fh.writelines(kept)
        print(f"Checkpoint: dropped {dropped} line(s) "
              f"(backup at {backup.name}).")
        if dropped != len(restored_rels):
            print(f"  NOTE: {len(restored_rels) - dropped} restored file(s) "
                  f"were not found in the checkpoint by rel-path — verify "
                  f"the checkpoint rel-path convention matches the output "
                  f"tree before resuming (otherwise --resume will skip "
                  f"them and they stay unrecovered).")

    print("\nDone. Now relaunch with --resume to re-clean the restored "
          "files and finish the backlog.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
