#!/usr/bin/env python
"""Clean every project directory on a drive in one supervised sweep.

Each immediate subdirectory of --source-root (or grandchild with
--depth 2) becomes one cleaning UNIT: its own EntityMapper (placeholders
consistent within the unit, unlinkable across units), its own staging /
quarantine / audit trees, and a PROJECT_nnn pseudonym — the source
directory names themselves ('70198 - Impact Shockwave Hole Saw') are
identifying, so only pseudonyms appear in the deliverable. The
name -> pseudonym map and per-unit outcomes land in
<output>_audit/batch_manifest.json (keep private, like every audit dir).

Media decisions scale drive-wide: run run_media_review.py ONCE on the
whole drive (clusters dedupe by content hash, so each distinct picture
is reviewed once, 'logo' enrolls a matcher template), then pass the
resulting media_review.json here — each unit is rewritten against those
hash decisions into a temp dir before cleaning, and the temp copy is
removed afterwards (disk cost: one unit at a time, never the drive).

Typical drive run:
    python run_batch.py \
        --source-root '/media/sparrows/slvrsrfr/samples unfinished' \
        --output ~/full_slvrsrfr_data_cleaned \
        --media-review ~/full_slvrsrfr_data_cleaned_audit/_media/media_review.json \
        --logo-templates ~/full_slvrsrfr_data_cleaned_audit/_media/logo_templates

Re-running resumes: units recorded as cleanly finished are skipped
(--no-resume to redo everything). Exit 0 = every unit finished with all
checks passing; 1 = some unit finished with failed checks or
quarantined files (triage its log + review_candidates.json); 2 = a unit
could not run at all.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_logger = logging.getLogger('run_batch')

_SKIP_NAMES = {'$RECYCLE.BIN', 'System Volume Information', 'lost+found'}


def _units(root: Path, depth: int):
    """Yield unit directories; warn on loose files that get no unit."""
    def _children(d: Path):
        out = []
        for p in sorted(d.iterdir()):
            if p.name.startswith('.') or p.name.startswith('._'):
                continue
            if p.name in _SKIP_NAMES or p.name.endswith('_media_clean'):
                continue
            out.append(p)
        return out

    loose = []
    if depth == 1:
        for p in _children(root):
            if p.is_dir():
                yield p
            else:
                loose.append(p)
    else:
        for child in _children(root):
            if not child.is_dir():
                loose.append(child)
                continue
            for p in _children(child):
                if p.is_dir():
                    yield p
                else:
                    loose.append(p)
    if loose:
        _logger.warning(
            "%d loose file(s) at unit level get NO cleaning unit and "
            "are NOT processed — move them into a directory first: %s",
            len(loose), ', '.join(str(p) for p in loose[:10]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Batch-clean every project directory on a drive.')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True,
                        help='Deliverable root; each unit lands in '
                             '<output>/PROJECT_nnn')
    parser.add_argument('--depth', type=int, choices=(1, 2), default=1,
                        help='1 (default): each immediate subdirectory '
                             'is a unit. 2: each grandchild directory '
                             '(client/project layouts).')
    parser.add_argument('--media-review', default=None, metavar='JSON',
                        help='Drive-wide media_review.json; each unit is '
                             'rewritten against its hash decisions before '
                             'cleaning. Omit = no media pre-pass (embedded '
                             'office media ships as-is — not recommended).')
    parser.add_argument('--logo-templates', default=None, metavar='DIR')
    parser.add_argument('--seed', action='append', default=[],
                        metavar='TYPE=VALUE',
                        help='Passed to every unit (repeatable)')
    parser.add_argument('--seed-file', default=None,
                        help='Passed to every unit')
    parser.add_argument('--targets', default='names',
                        help="Passed through to run_clean (default "
                             "'names' = person,company,email)")
    parser.add_argument('--only', action='append', default=[],
                        metavar='NAME',
                        help='Process only units whose directory name '
                             'matches (repeatable)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Redo units already recorded as finished')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname).1s] %(message)s',
        datefmt='%H:%M:%S')

    from run_clean import parse_targets
    try:
        parse_targets(args.targets)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    root = Path(args.source_root)
    if not root.is_dir():
        print(f'ERROR: source root not found: {root}', file=sys.stderr)
        return 2
    output = Path(args.output).expanduser()
    audit_root = output.parent / (output.name + '_audit')
    work_root = output.parent / (output.name + '_work')
    audit_root.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_root / 'batch_manifest.json'
    manifest = {}
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)

    review = None
    if args.media_review:
        review_path = Path(args.media_review).expanduser()
        if not review_path.is_file():
            print(f'ERROR: --media-review not found: {review_path}',
                  file=sys.stderr)
            return 2
        with open(review_path) as f:
            review = json.load(f)
        undecided = [c['id'] for c in review.get('clusters', [])
                     if c.get('action', 'review') not in
                     ('keep', 'redact', 'logo')]
        if undecided:
            print(f'ERROR: {len(undecided)} media cluster(s) still '
                  f"undecided ('review') — finish the review first: "
                  + ', '.join(undecided[:8]), file=sys.stderr)
            return 2

    units = [u for u in _units(root, args.depth)
             if not args.only or u.name in args.only]
    if not units:
        print('No units found.', file=sys.stderr)
        return 2

    # Stable pseudonyms: reuse manifest assignments, mint for new units.
    taken = {m['pseudonym'] for m in manifest.values()
             if isinstance(m, dict) and 'pseudonym' in m}
    counter = 0
    for unit in units:
        key = str(unit)
        if key not in manifest:
            counter += 1
            while (p := f'PROJECT_{counter:03d}') in taken:
                counter += 1
            taken.add(p)
            manifest[key] = {'pseudonym': p, 'status': 'pending'}

    worst = 0
    for i, unit in enumerate(units, 1):
        entry = manifest[str(unit)]
        pseudonym = entry['pseudonym']
        staging = output / pseudonym
        if (not args.no_resume and entry.get('status') == 'done'
                and staging.is_dir()):
            _logger.info('[%d/%d] %s -> %s already done — skipped',
                         i, len(units), unit.name, pseudonym)
            continue

        _logger.info('[%d/%d] %s -> %s', i, len(units), unit.name,
                     pseudonym)
        source_for_clean = unit
        media_work = None
        media_quarantined = []
        try:
            if review is not None:
                from run_media_review import apply_decisions
                media_work = work_root / f'{pseudonym}_media'
                if media_work.exists():
                    shutil.rmtree(media_work)
                stats = apply_decisions(unit, review, media_work)
                if stats.get('unresolved'):
                    raise RuntimeError(
                        'media clusters unresolved: '
                        + ', '.join(stats['unresolved'][:5]))
                media_quarantined = list(stats.get('quarantined', []))
                _logger.info(
                    '  media pre-pass: %d container(s) rewritten, '
                    '%d part(s) replaced, %d quarantined',
                    stats.get('files_rewritten', 0),
                    stats.get('parts_replaced', 0),
                    len(media_quarantined))
                # apply_decisions quarantines unprocessable ORIGINALS
                # into <media_work>_quarantine. Relocate them to the
                # real quarantine root (the one the summary prints) so
                # nothing raw is stranded in the _work tree, and count
                # them against the exit contract.
                media_q = Path(str(media_work) + '_quarantine')
                if media_q.is_dir():
                    dest = (output.parent
                            / (output.name + '_quarantine')
                            / pseudonym / '_media')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(media_q), str(dest))
                    _logger.warning(
                        '  %d media-pass file(s) quarantined -> %s',
                        len(media_quarantined), dest)
                source_for_clean = media_work

            cmd = [sys.executable,
                   str(Path(__file__).parent / 'run_clean.py'),
                   '--source', str(source_for_clean),
                   '--project', pseudonym,
                   '--staging', str(staging),
                   '--targets', args.targets,
                   '--clobber']
            for seed in args.seed:
                cmd += ['--seed', seed]
            if args.seed_file:
                cmd += ['--seed-file', args.seed_file]
            if args.logo_templates:
                cmd += ['--logo-templates',
                        str(Path(args.logo_templates).expanduser())]

            log_path = audit_root / f'{pseudonym}.log'
            with open(log_path, 'w') as log:
                rc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc >= 2:
                # run_clean could not run at all (bad args, missing
                # seed file, unreachable required LLM) — not a partial
                # result, so it must not read as 'done-with-failures'.
                entry['status'] = 'error'
                worst = max(worst, 2)
            elif rc == 1 or media_quarantined:
                entry['status'] = 'done-with-failures'
                worst = max(worst, 1)
            else:
                entry['status'] = 'done'
            entry['rc'] = rc
            if media_quarantined:
                entry['media_quarantined'] = media_quarantined
            entry['source'] = str(unit)
            entry['log'] = str(log_path)
            entry['finished'] = datetime.now(
                timezone.utc).isoformat(timespec='seconds')
            _logger.info('  %s (rc=%d, log: %s)',
                         entry['status'], rc, log_path.name)
        except Exception as e:
            entry['status'] = 'error'
            entry['error'] = str(e)
            worst = max(worst, 2)
            _logger.error('  FAILED to run: %s', e)
        finally:
            if media_work is not None and media_work.exists():
                shutil.rmtree(media_work, ignore_errors=True)
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 64)
    for unit in units:
        entry = manifest[str(unit)]
        print(f"  {entry['pseudonym']}  {entry.get('status'):20s}"
              f"  {unit.name}")
    print('=' * 64)
    print(f'Deliverable: {output}')
    print(f'Audit (KEEP PRIVATE): {audit_root} '
          f'(batch_manifest.json = name->pseudonym map)')
    print(f'Quarantine: {output.parent / (output.name + "_quarantine")}')
    return worst


if __name__ == '__main__':
    sys.exit(main())
