"""Pixel-level logo template matching for rendered pages and images.

The one leak class OCR can never catch: vector logo art (the Milwaukee
script in a CAD title block is line work, not text — no embedded image
to strip, nothing for OCR to read). But a logo is a STAMP: the same
artwork every time. Template matching finds the stamp wherever it
appears, at any scale, regardless of whether it started as vector art,
an embedded bitmap, or a scan.

Templates come from PROJECT_P_LOGO_TEMPLATES (a directory of image
crops). run_media_review.py --apply exports every cluster the human
marked 'redact' into <audit>/logo_templates/ automatically, so the
media review doubles as template harvesting; drop extra crops (a
screenshot of a missed mark) into the same folder and re-run.

Method: multi-scale cv2.matchTemplate(TM_CCOEFF_NORMED) on INTENSITY,
matching the template in both polarities (as-is and inverted) so a
white-on-black embedded logo still matches its black-on-white vector
rendering. Empirical on the live Milwaukee mark: intensity-inverted
scored 0.77 at the logo while Canny-edge correlation managed 0.26 —
edge maps are too sparse for stroke-width drift. Matches at score >=
PROJECT_P_LOGO_THRESHOLD (default 0.6) become boxes for the caller to
black out. Rotated instances are searched at 90-degree steps;
arbitrary-angle marks (a logo drawn along a 30-degree axis) are NOT
caught — that residual class stays documented.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

_logger = logging.getLogger(__name__)

_TEMPLATE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')

# Search-space constants: page downscaled to this max dimension for
# matching (boxes mapped back), template rescaled across this range.
_PAGE_MAX_DIM = 1800
_SCALES = (0.25, 0.35, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4, 3.0)
_MIN_TEMPLATE_PX = 12
# Reject matched boxes whose short side is tiny at full resolution: a
# wordmark scaled to 15px matches arbitrary line features (live: an
# M18 template at 0.25x nicked the word 'REFLECTIVE').
_MIN_MATCH_SHORT_SIDE = 24

_TEMPLATE_CACHE: dict = {}


def _rotations() -> int:
    """90-degree rotation steps to search. Off by default: title-block
    stamps are axis-aligned, and the 4x cost buys almost nothing."""
    return 4 if os.environ.get('PROJECT_P_LOGO_ROTATIONS', '0') == '1' \
        else 1


def env_templates():
    """Templates from PROJECT_P_LOGO_TEMPLATES, cached by (dir, mtime)."""
    directory = templates_dir()
    if directory is None:
        return []
    key = (str(directory), max((p.stat().st_mtime
                                for p in directory.iterdir()), default=0))
    if key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE.clear()
        _TEMPLATE_CACHE[key] = load_templates(directory)
    return _TEMPLATE_CACHE[key]


def _threshold() -> float:
    try:
        return float(os.environ.get('PROJECT_P_LOGO_THRESHOLD', '0.6'))
    except ValueError:
        return 0.6


def templates_dir() -> Optional[Path]:
    raw = os.environ.get('PROJECT_P_LOGO_TEMPLATES', '').strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _polarities(gray):
    """The template as-is and inverted (embedded art is often
    light-on-dark while page renderings are dark-on-light)."""
    return gray, 255 - gray


def load_templates(directory: Path) -> List[Tuple[str, 'object']]:
    """Load template crops as grayscale arrays (name, image)."""
    import cv2
    out = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _TEMPLATE_EXTS:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape[:2]) < _MIN_TEMPLATE_PX:
            _logger.warning("Logo template unreadable/too small: %s",
                            path.name)
            continue
        out.append((path.name, img))
    return out


def find_logo_boxes(pil_image, templates) -> List[Tuple[int, int, int, int]]:
    """Match every template against a PIL image.

    Returns deduplicated (x, y, w, h) boxes in FULL-RESOLUTION image
    coordinates. Empty list when nothing matches or cv2 is missing.
    """
    if not templates:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError:
        _logger.warning("OpenCV unavailable — logo template matching "
                        "skipped (pip install opencv-python).")
        return []

    page = np.array(pil_image.convert('L'))
    full_h, full_w = page.shape[:2]
    ratio = max(full_w, full_h) / float(_PAGE_MAX_DIM)
    if ratio > 1:
        page = cv2.resize(page, (int(full_w / ratio), int(full_h / ratio)),
                          interpolation=cv2.INTER_AREA)
    else:
        ratio = 1.0
    threshold = _threshold()

    boxes: List[Tuple[int, int, int, int, float]] = []
    for name, tmpl in templates:
        for rot in range(_rotations()):
            t = np.rot90(tmpl, rot) if rot else tmpl
            for scale in _SCALES:
                tw = int(t.shape[1] * scale)
                th = int(t.shape[0] * scale)
                if (tw < _MIN_TEMPLATE_PX or th < _MIN_TEMPLATE_PX
                        or tw >= page.shape[1] or th >= page.shape[0]):
                    continue
                t_scaled = cv2.resize(t, (tw, th),
                                      interpolation=cv2.INTER_AREA)
                if int(t_scaled.std()) == 0:
                    continue
                for variant in _polarities(t_scaled):
                    result = cv2.matchTemplate(page, variant,
                                               cv2.TM_CCOEFF_NORMED)
                    ys, xs = np.where(result >= threshold)
                    for x, y in zip(xs, ys):
                        boxes.append((int(x * ratio), int(y * ratio),
                                      int(tw * ratio), int(th * ratio),
                                      float(result[y, x])))

    boxes = [b for b in boxes
             if min(b[2], b[3]) >= _MIN_MATCH_SHORT_SIDE]
    # Greedy non-max suppression on IoU
    boxes.sort(key=lambda b: -b[4])
    kept: List[Tuple[int, int, int, int]] = []
    for x, y, w, h, _score in boxes:
        clash = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            union = w * h + kw * kh - inter
            if union > 0 and inter / union > 0.3:
                clash = True
                break
        if not clash:
            kept.append((x, y, w, h))
    return kept
