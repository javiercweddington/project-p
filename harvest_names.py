#!/usr/bin/env python3
"""Harvest candidate company/person names from drawing title blocks.

The clean run's GLiNER discovery only sees extractable TEXT; on CAD PDFs
the title-block text is vector art, so names like ZhengYU / Qihao / the
DR.BY/APP.BY people never surface in review_candidates.json. This script
RENDERS + OCRs every source PDF (at 4 rotations — title blocks are often
sideways) and pulls out:

  * company candidates: phrases ending in Co.,LTD / Technologies /
    Electronic(s) / Inc / Corp / Limited
  * person candidates:  the token(s) after DR. BY / APP. BY / CHECKED /
    DESIGNED / DRAWN / CUSTOMER / APPROVED

It prints a deduped, frequency-ranked candidate list in ready-to-paste
--seed form. It does NOT clean anything — you review the list, drop the
junk (CAD software like Unigraphics, generic words), and seed the rest
for ONE comprehensive --clobber rerun.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire.metadata import ImageOCR  # noqa: E402

COMPANY_RE = re.compile(
    r'([A-Z][A-Za-z&.\-]*(?:\s+[A-Z0-9][A-Za-z0-9&.\-]*){0,4}'
    r'\s*(?:Co\.?\s*,?\s*,?\s*LTD|Co\.?\s*,?\s*Limited|Technologies|'
    r'Technology|Electronics?|Industries|Industrial|Inc\.?|Corp\.?|'
    r'Limited|GmbH))',
    re.IGNORECASE)

PERSON_RE = re.compile(
    r'(?:DR\.?\s*BY|APP\.?\s*BY|APPROVED(?:\s*BY)?|CHECKED(?:\s*BY)?|'
    r'DESIGNED(?:\s*BY)?|DRAWN(?:\s*BY)?|CUSTOMER)[:\s.]*'
    r'([A-Z][A-Za-z]{1,20})',
    re.IGNORECASE)

# Obvious non-client noise to pre-filter from the output.
DROP = {'unigraphics', 'solidworks', 'autocad', 'inventor', 'catia',
        'creo', 'nx', 'proe', 'drawing', 'drawings', 'title', 'material',
        'customer', 'approved', 'checked', 'designed', 'drawn', 'scale',
        'date', 'sheet', 'rev', 'model', 'project', 'none', 'unit', 'mm'}


def ocr_pdf(path: Path, ocr: ImageOCR, dpi: int = 200) -> str:
    """OCR every page of a PDF at four rotations; return merged text."""
    chunks = []
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  (skip {path.name}: {e})", file=sys.stderr)
        return ''
    try:
        zoom = dpi / 72.0
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            base = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            for angle in (0, 90, 180, 270):
                img = base if angle == 0 else base.rotate(angle, expand=True)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                tmp = path.with_suffix(f'.ocrtmp{angle}.png')
                try:
                    img.save(tmp)
                    text = ocr.extract_text(tmp, lang='eng+chi_sim')
                    if text:
                        chunks.append(text)
                finally:
                    tmp.unlink(missing_ok=True)
    finally:
        doc.close()
    return '\n'.join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True,
                    help='Dossier root to scan for PDFs.')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    ocr = ImageOCR()
    if not ocr.available:
        print("No OCR backend available.", file=sys.stderr)
        return 1

    src = Path(args.source)
    pdfs = sorted(p for p in src.rglob('*')
                  if p.suffix.lower() == '.pdf' and not p.name.startswith('.'))
    print(f"OCR-scanning {len(pdfs)} PDF(s)...", file=sys.stderr)

    companies: dict[str, set] = defaultdict(set)
    persons: dict[str, set] = defaultdict(set)
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {pdf.name}", file=sys.stderr)
        text = ocr_pdf(pdf, ocr, args.dpi)
        for m in COMPANY_RE.finditer(text):
            val = re.sub(r'\s+', ' ', m.group(1)).strip(' .,')
            if val and val.lower() not in DROP and len(val) > 2:
                companies[val].add(pdf.name)
        for m in PERSON_RE.finditer(text):
            val = m.group(1).strip()
            if val and val.lower() not in DROP and len(val) >= 2:
                persons[val].add(pdf.name)

    def dump(title, d, etype):
        print(f"\n=== {title} ({len(d)}) ===")
        for val, files in sorted(d.items(), key=lambda kv: -len(kv[1])):
            print(f"  --seed '{etype}={val}'"
                  f"    # {len(files)} file(s), e.g. {sorted(files)[0]}")

    dump('COMPANY CANDIDATES', companies, 'company')
    dump('PERSON CANDIDATES', persons, 'person')
    print("\nReview, delete junk (CAD software, generic words), then paste "
          "the keepers into your SEEDS for one --clobber rerun.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
