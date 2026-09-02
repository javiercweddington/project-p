#!/usr/bin/env python
"""Build the embedded-media review sheet for a project directory.

Runs ahead of run_clean.py. Extracts every embedded image in the
corpus, collapses duplicates, and writes a review sheet plus contact
sheets so a human can mark each distinct picture keep/redact.

    # 1. inventory (no decisions made)
    python run_media_review.py --source DIR --project NAME

    # 2. open media_contact_XX.png, edit media_review.json:
    #    set "action" to "redact" or "keep" on each cluster
    #    (anything left as "review" is treated as unresolved)

    # 3. apply, then clean as usual
    python run_media_review.py --source DIR --project NAME \
        --apply --review-file /tmp/clean_audit/NAME/media_review.json
    python run_clean.py --source DIR_media_clean --project NAME ...

Why this exists: clean/cleaners/zip_cleaner.py does not route embedded
media through ImageCleaner, so a logo at word/media/image1.png in a
letterhead currently passes through untouched. And ImageCleaner is
OCR-driven, so a purely graphical mark has nothing to trigger on even
where it is reached. Hash grouping sidesteps both: a human confirms a
few dozen pictures, and removal is by exact hash match.

Exit code 0 when the run completes; 3 when --apply finds clusters
still marked "review" (fail closed rather than shipping undecided
media).
"""

import argparse
import json
import logging
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CONTACT_COLS = 6
CONTACT_CELL = 150
CONTACT_PER_SHEET = 36


def _decode_preview(blob: bytes):
    """Decode image bytes to a PIL image, trying hard.

    PIL first; then ImageMagick, then ffmpeg (first frame — covers
    .wdp/JPEG-XR, EMF/WMF metafiles, and video members like .MOV).
    A reviewer who cannot SEE a picture cannot veto a logo — live, the
    undecodable clusters were exactly where marks slipped review.
    Returns None only when every decoder fails.
    """
    import io
    import shutil as _sh
    import subprocess
    import tempfile
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
        return img
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        tf.write(blob)
        src = tf.name
    try:
        for tool, cmd in (
                ('magick', ['magick', src, 'png:-']),
                ('convert', ['convert', src, 'png:-']),
                ('ffmpeg', ['ffmpeg', '-v', 'error', '-i', src,
                            '-frames:v', '1', '-f', 'image2pipe',
                            '-vcodec', 'png', 'pipe:1'])):
            if not _sh.which(tool):
                continue
            try:
                out = subprocess.run(cmd, capture_output=True, timeout=30)
                if out.returncode == 0 and out.stdout:
                    img = Image.open(io.BytesIO(out.stdout))
                    img.load()
                    return img
            except Exception:
                continue
    finally:
        Path(src).unlink(missing_ok=True)
    return None


def write_contact_sheets(clusters, out_dir: Path) -> list:
    """Render numbered thumbnail grids of every distinct picture.

    The review step is only as good as the reviewer's ability to
    actually see what they are deciding about.
    """
    from PIL import Image, ImageDraw

    written = []
    renderable = [c for c in clusters if c.sample_bytes]
    for start in range(0, len(renderable), CONTACT_PER_SHEET):
        chunk = renderable[start:start + CONTACT_PER_SHEET]
        rows = (len(chunk) + CONTACT_COLS - 1) // CONTACT_COLS
        sheet = Image.new(
            'RGB',
            (CONTACT_COLS * CONTACT_CELL, rows * (CONTACT_CELL + 22)),
            (255, 255, 255))
        draw = ImageDraw.Draw(sheet)

        for offset, cluster in enumerate(chunk):
            col, row = offset % CONTACT_COLS, offset // CONTACT_COLS
            x, y = col * CONTACT_CELL, row * (CONTACT_CELL + 22)
            box = (x + 4, y + 4, x + CONTACT_CELL - 4, y + CONTACT_CELL - 4)
            draw.rectangle(box, outline=(200, 200, 200))
            img = _decode_preview(cluster.sample_bytes)
            if img is not None:
                try:
                    thumb = img.convert('RGB')
                    thumb.thumbnail((CONTACT_CELL - 16, CONTACT_CELL - 16))
                    sheet.paste(
                        thumb,
                        (x + (CONTACT_CELL - thumb.width) // 2,
                         y + (CONTACT_CELL - thumb.height) // 2))
                except Exception:
                    img = None
            if img is None:
                draw.text((x + 12, y + CONTACT_CELL // 2),
                          'undecodable', fill=(180, 0, 0))
            label = (f'{cluster.cluster_id.replace("media_", "#")}  '
                     f'x{cluster.occurrence_count}')
            draw.text((x + 6, y + CONTACT_CELL - 2), label, fill=(0, 0, 0))

        path = out_dir / f'media_contact_{start // CONTACT_PER_SHEET:02d}.png'
        sheet.save(path)
        written.append(path)
    return written


def write_thumbnails(clusters, out_dir: Path) -> dict:
    """One PNG per distinct picture, for the review page.

    These are copies of the extracted art, so they live in the audit
    directory alongside the placeholder->original map and inherit the
    same KEEP PRIVATE handling.
    """
    import io
    from PIL import Image

    thumbs = out_dir / 'thumbs'
    thumbs.mkdir(parents=True, exist_ok=True)
    out = {}
    for cluster in clusters:
        if not cluster.sample_bytes:
            continue
        target = thumbs / f'{cluster.cluster_id}.png'
        img = _decode_preview(cluster.sample_bytes)
        if img is None:
            continue
        try:
            thumb = img.convert('RGBA')
            thumb.thumbnail((320, 320))
            thumb.save(target)
        except Exception:
            continue
        out[cluster.cluster_id] = f'thumbs/{target.name}'
    return out


def build_review(clusters, corpus_size: int, source: Path,
                 project: str, thumbs: dict = None) -> dict:
    thumbs = thumbs or {}
    return {
        'project': project,
        'source': str(source),
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'documents_scanned': corpus_size,
        'distinct_images': len(clusters),
        'total_occurrences': sum(c.occurrence_count for c in clusters),
        'instructions': (
            'Set "action" on each cluster to "redact" or "keep". '
            'Clusters left as "review" block --apply. Ranking is a '
            'sorting hint only -- review every cluster.'),
        'clusters': [
            {
                'id': c.cluster_id,
                'action': 'review',
                'score': c.score(corpus_size),
                'signals': c.signals(corpus_size),
                'occurrences': c.occurrence_count,
                'documents': c.document_count,
                'in_chrome': c.chrome_count,
                'width': c.width,
                'height': c.height,
                'format': c.image_format,
                'sha256': c.sha256,
                'decode_error': c.decode_error,
                'thumbnail': thumbs.get(c.cluster_id),
                'examples': [o.as_dict(source) for o in c.occurrences[:5]],
            }
            for c in clusters
        ],
    }


def _placeholder_png(width: int, height: int) -> bytes:
    """Neutral fill of identical dimensions, so layout does not reflow."""
    import io
    from PIL import Image
    img = Image.new('RGB', (max(width or 1, 1), max(height or 1, 1)),
                    (229, 229, 229))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _apply_pdf(path: Path, target: Path, flagged: dict) -> int:
    """Replace flagged image XObjects inside a PDF, in place.

    Hashes are computed with the same extract_image() call the scan
    used, so digests agree across the two passes. The save uses full
    garbage collection: without it the original stream survives as an
    orphaned object and the logo is still recoverable from the file.

    Raises if PyMuPDF is unavailable -- the caller quarantines rather
    than copying an unprocessed PDF into the deliverable.
    """
    import hashlib
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # older installs expose only the fitz name

    replaced = 0
    doc = pymupdf.open(path)
    try:
        for page_no in range(doc.page_count):
            page = doc[page_no]
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    info = doc.extract_image(xref)
                except Exception:
                    continue
                blob = info.get('image')
                if not blob:
                    continue
                if hashlib.sha256(blob).hexdigest() not in flagged:
                    continue
                page.replace_image(
                    xref,
                    stream=_placeholder_png(info.get('width', 1),
                                            info.get('height', 1)))
                replaced += 1
        doc.save(target, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()
    return replaced


def apply_decisions(source: Path, review: dict, out_dir: Path) -> dict:
    """Rewrite the corpus with flagged media parts replaced.

    OOXML containers are rewritten member by member; PDFs have their
    image XObjects swapped. Anything that cannot be processed is
    quarantined rather than copied through, so an unprocessable file
    never reaches the deliverable by default.
    """
    flagged = {}
    unresolved = []
    for entry in review.get('clusters', []):
        action = entry.get('action', 'review')
        # 'logo' = redact here AND enroll as a template (see the export
        # block in main) so the pixel belt hunts the mark corpus-wide.
        if action in ('redact', 'logo'):
            for digest in entry.get('sha256', []):
                flagged[digest] = entry
        elif action != 'keep':
            unresolved.append(entry['id'])

    if unresolved:
        return {'unresolved': unresolved}

    import hashlib
    from clean.media_index import (OOXML_SUFFIXES,
                                   STANDALONE_IMAGE_SUFFIXES)

    out_dir.mkdir(parents=True, exist_ok=True)
    quarantine = Path(str(out_dir) + '_quarantine')
    stats = {'files_rewritten': 0, 'parts_replaced': 0,
             'pdfs_rewritten': 0, 'pdf_images_replaced': 0,
             'files_copied': 0, 'images_replaced': 0,
             'quarantined': [], 'unresolved': []}

    def _quarantine_file(path: Path, rel: Path) -> None:
        # Preserve the relative path: two 'image.png' in different
        # folders must not overwrite each other in quarantine.
        dest = quarantine / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        stats['quarantined'].append(str(rel))

    # Walk EVERYTHING (skipping dotfiles). iter_source_files() only
    # yields media-capable formats — iterating it here silently dropped
    # every .doc/.xls/.txt/.sldprt from the rewritten corpus.
    for path in sorted(p for p in source.rglob('*') if p.is_file()):
        if path.name.startswith('.'):
            continue
        rel = path.relative_to(source)
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            try:
                replaced = _apply_pdf(path, target, flagged)
            except Exception as exc:
                logging.error('Could not rewrite PDF %s: %s -- '
                              'quarantining', path, exc)
                target.unlink(missing_ok=True)
                _quarantine_file(path, rel)
                continue
            stats['pdfs_rewritten'] += 1
            stats['pdf_images_replaced'] += replaced
            continue

        if suffix in STANDALONE_IMAGE_SUFFIXES:
            # A flagged standalone image IS the logo file — replace it
            # with a same-size placeholder instead of copying it through.
            try:
                blob = path.read_bytes()
            except OSError as exc:
                logging.error('Could not read %s: %s -- quarantining',
                              path, exc)
                _quarantine_file(path, rel)
                continue
            if hashlib.sha256(blob).hexdigest() in flagged:
                width = height = 1
                try:
                    import io as _io
                    from PIL import Image as _Image
                    with _Image.open(_io.BytesIO(blob)) as probe:
                        width, height = probe.size
                except Exception:
                    pass
                target.with_suffix('.png').write_bytes(
                    _placeholder_png(width, height))
                if target.suffix.lower() != '.png':
                    target.unlink(missing_ok=True)
                stats['images_replaced'] += 1
            else:
                shutil.copy2(path, target)
                stats['files_copied'] += 1
            continue

        if suffix not in OOXML_SUFFIXES:
            shutil.copy2(path, target)
            stats['files_copied'] += 1
            continue

        replaced_here = 0
        try:
            with zipfile.ZipFile(path) as src, \
                    zipfile.ZipFile(target, 'w',
                                    zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    blob = src.read(info)
                    digest = hashlib.sha256(blob).hexdigest()
                    if digest in flagged:
                        entry = flagged[digest]
                        # Size from the part being replaced: a cluster
                        # merges rescaled variants, so the
                        # representative's dimensions are not
                        # necessarily this occurrence's.
                        width = entry.get('width') or 1
                        height = entry.get('height') or 1
                        try:
                            import io as _io
                            from PIL import Image as _Image
                            with _Image.open(_io.BytesIO(blob)) as probe:
                                width, height = probe.size
                        except Exception:
                            pass
                        blob = _placeholder_png(width, height)
                        replaced_here += 1
                    dst.writestr(info, blob)
        except (zipfile.BadZipFile, OSError) as exc:
            logging.error('Could not rewrite %s: %s -- quarantining',
                          path, exc)
            target.unlink(missing_ok=True)
            _quarantine_file(path, rel)
            continue

        stats['files_rewritten'] += 1
        stats['parts_replaced'] += replaced_here

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Inventory and triage embedded media before cleaning.')
    parser.add_argument('--source', required=True)
    parser.add_argument('--project', default=None)
    parser.add_argument('--audit-dir', default=None,
                        help='Where the review sheet lands '
                             '(default /tmp/clean_audit/<project>)')
    parser.add_argument('--apply', action='store_true',
                        help='Rewrite containers using --review-file')
    parser.add_argument('--review-file', default=None)
    parser.add_argument('--output', default=None,
                        help='Rewritten corpus (default <source>_media_clean)')
    parser.add_argument('--serve', action='store_true',
                        help='Open the review sheet in a local server '
                             'that writes decisions straight back')
    parser.add_argument('--port', type=int, default=8765,
                        help='Review server port (default 8765 — NOT '
                             '8000, which the vLLM endpoint owns; if '
                             'busy, the server bumps to the next free '
                             'port)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Bind address (default loopback only -- '
                             'forward the port instead of widening this)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname).1s] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger('PIL').setLevel(logging.WARNING)

    source = Path(args.source)
    if not source.is_dir():
        print(f'ERROR: source not found: {source}', file=sys.stderr)
        return 2
    project = args.project or source.name
    audit = Path(args.audit_dir) if args.audit_dir else (
        Path('/tmp/clean_audit') / project)
    audit.mkdir(parents=True, exist_ok=True)
    review_path = audit / 'media_review.json'

    if args.serve:
        path = Path(args.review_file) if args.review_file else review_path
        if not path.is_file():
            print(f'ERROR: review sheet not found: {path}\n'
                  f'Run the inventory pass first.', file=sys.stderr)
            return 2
        from clean.media_review_server import serve
        serve(path, path.parent, port=args.port, host=args.host)
        return 0

    if args.apply:
        path = Path(args.review_file) if args.review_file else review_path
        if not path.is_file():
            print(f'ERROR: review file not found: {path}', file=sys.stderr)
            return 2
        with open(path) as handle:
            review = json.load(handle)
        # ONLY explicit 'logo' decisions become templates. The old
        # behavior (every redact thumbnail auto-exported) enrolled
        # product photos and whole drawing sheets: matching went ~40x
        # slower and false boxes covered 16% of a live drawing page.
        # Thumbnails are flattened onto WHITE at export — cv2 loads
        # grayscale without alpha, so a transparent-background crop
        # would otherwise be matched on art the reviewer never saw.
        tdir = path.parent / 'logo_templates'
        exported = 0
        for entry in review.get('clusters', []):
            thumb = entry.get('thumbnail')
            if entry.get('action') == 'logo' and thumb:
                src_thumb = path.parent / thumb
                if src_thumb.is_file():
                    tdir.mkdir(parents=True, exist_ok=True)
                    dest = (tdir / src_thumb.name).with_suffix('.png')
                    try:
                        from PIL import Image as _Image
                        with _Image.open(src_thumb) as img:
                            if 'A' in img.getbands():
                                base = _Image.new(
                                    'RGBA', img.size,
                                    (255, 255, 255, 255))
                                base.alpha_composite(
                                    img.convert('RGBA'))
                                base.convert('RGB').save(dest)
                            else:
                                img.convert('RGB').save(dest)
                    except Exception:
                        shutil.copy2(src_thumb, tdir / src_thumb.name)
                    exported += 1
        if exported:
            print(f'Exported {exported} logo enrollment(s) as '
                  f'templates: {tdir}\n'
                  f'  Pass --logo-templates {tdir} to run_clean.py — '
                  f'the pixel belt hunts each mark corpus-wide (any '
                  f'color/scale, mild warp/skew) and blanks it to the '
                  f'local background.')
        out_dir = Path(args.output) if args.output else Path(
            str(source) + '_media_clean')
        stats = apply_decisions(source, review, out_dir)
        if stats.get('unresolved'):
            print('ERROR: clusters still marked "review": '
                  + ', '.join(stats['unresolved'][:10]), file=sys.stderr)
            return 3
        print(f'Rewrote {stats["files_rewritten"]} containers, '
              f'replaced {stats["parts_replaced"]} media parts; '
              f'{stats["pdfs_rewritten"]} PDFs, '
              f'{stats["pdf_images_replaced"]} images replaced; '
              f'{stats["images_replaced"]} standalone images replaced; '
              f'copied {stats["files_copied"]} other files.')
        if stats['quarantined']:
            print(f'QUARANTINED {len(stats["quarantined"])} unprocessable '
                  f'files (not in deliverable): '
                  + ', '.join(stats['quarantined'][:5]))
        print(f'Output: {out_dir}')
        return 0

    from clean.media_index import build_index

    clusters, corpus_size = build_index(source, progress=True)
    if not clusters:
        print('No embedded media found.')
        return 0

    thumbs = write_thumbnails(clusters, audit)
    review = build_review(clusters, corpus_size, source, project, thumbs)
    with open(review_path, 'w') as handle:
        json.dump(review, handle, indent=2, ensure_ascii=False)
    sheets = write_contact_sheets(clusters, audit)

    total = review['total_occurrences']
    print()
    print('=' * 64)
    print(f'{corpus_size} documents -> {total} embedded images '
          f'-> {len(clusters)} distinct pictures to review')
    print('=' * 64)
    for entry in review['clusters'][:12]:
        marks = ','.join(k for k, v in entry['signals'].items() if v)
        print(f'  {entry["id"]}  score {entry["score"]:>2}  '
              f'x{entry["occurrences"]:<5} in {entry["documents"]:<4} docs  '
              f'{entry["width"]}x{entry["height"]}  {marks}')
    if len(review['clusters']) > 12:
        print(f'  ... {len(review["clusters"]) - 12} more')
    print()
    print(f'Review sheet:   {review_path}')
    for sheet in sheets:
        print(f'Contact sheet:  {sheet}')
    print()
    print('Review in the browser:')
    print(f'  python run_media_review.py --source {source} '
          f'--project {project} --serve')
    print('or edit the JSON by hand. Then re-run with --apply.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
