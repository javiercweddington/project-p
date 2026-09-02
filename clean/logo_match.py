"""Pixel-level logo template matching for rendered pages and images.

The one leak class OCR can never catch: vector logo art (the Milwaukee
script in a CAD title block is line work, not text — no embedded image
to strip, nothing for OCR to read). But a logo is a STAMP: the same
artwork every time. Template matching finds the stamp wherever it
appears, at any scale, regardless of whether it started as vector art,
an embedded bitmap, or a scan.

Enrollment model: a human identifies each mark ONCE (the media review
UI's 'logo' decision exports the crop, or drop any crop into
PROJECT_P_LOGO_TEMPLATES); the machine then finds every instance
corpus-wide. Keep the template set to actual marks — a handful of
curated crops. Enrolling product photos or whole drawing sheets as
"templates" makes matching ~40x slower and produces sweeping false
boxes (measured live: 37 uncurated thumbnails covered 16.4% of a
drawing page). A per-template sanity guard drops any template whose
matches look pathological, but curation is the real fix.

Method: multi-scale cv2.matchTemplate(TM_CCOEFF_NORMED) on INTENSITY,
matching the template in both polarities (as-is and inverted) so a
white-on-black embedded logo still matches its black-on-white vector
rendering. Empirical on the live Milwaukee mark: intensity-inverted
scored 0.77 at the logo while Canny-edge correlation managed 0.26 —
edge maps are too sparse for stroke-width drift. Grayscale matching is
color-blind by construction (hue-shift and channel-swap tested within
0.001 of baseline). On top of the scale pyramid, the search covers:

- 90-degree orientations (PROJECT_P_LOGO_ROTATIONS, default ON — a
  rotated scan defeated the axis-aligned search on a live drawing);
- small skews (PROJECT_P_LOGO_ANGLES, default "0,5,-5" degrees —
  measured: a 5-degree skew scored 0.561 vs the 0.6 threshold, i.e.
  just below the axis-aligned miss line);
- horizontal squeeze (PROJECT_P_LOGO_ASPECTS, default "0.8") for marks
  engraved AROUND cylindrical parts (moderate wrap reads as horizontal
  compression; near-edge-on foreshortening stays a documented residual).

Matches at score >= PROJECT_P_LOGO_THRESHOLD (default 0.6) become
boxes; the caller pads them and blanks to the local background color.
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
# Non-upright variants (90-degree, skew, squeeze) use the coarse grid —
# a true hit lands within one scale step of a grid point and the
# genuine-match headroom (live hits score 0.71-0.75) absorbs the gap.
_PAGE_MAX_DIM = 1800
# Both grids MUST include 1.0: templates are normalized to page-space
# proportions at load, so an occurrence at its enrolled size matches at
# scale ~1.0 — the original grid's 0.9->1.1 jump dropped a live
# self-match from 0.663 to 0.527/0.577, below any usable threshold
# (thin-stroke art collapses fast under scale error).
_SCALES = (0.25, 0.35, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4,
           1.8, 2.4, 3.0)
_COARSE_SCALES = (0.25, 0.5, 0.75, 1.0, 1.4, 2.4)
_MIN_TEMPLATE_PX = 12
# Templates are normalized to this max dimension at load. Enrolled
# crops arrive at whatever DPI the source render used (a 200dpi crop
# is ~500px wide) while the search page is capped at 1800px — without
# normalization the correct match scale falls BETWEEN pyramid steps
# (needed 0.385, grid has 0.35/0.5) and thin line art drops below
# threshold at 10% scale error. Normalized to page-space proportions,
# scale 1.0 sits near a typical occurrence and the fine grid brackets
# every real size.
_TEMPLATE_NORM_MAX_DIM = 220
# Reject matches whose short side is tiny AT MATCH RESOLUTION (the
# downscaled page): a wordmark scaled to ~15px matches arbitrary line
# features (live: an M18 template at 0.25x nicked 'REFLECTIVE'; after
# the guard moved to full-res coords, 16px match-space hits on section
# arrows slipped back in at 200dpi because they mapped to 30px).
_MIN_MATCH_SHORT_SIDE = 24
# Cap raw peaks taken per matchTemplate call (bounds NMS cost when a
# junk template lights up everywhere).
_MAX_RAW_PEAKS = 20

_TEMPLATE_CACHE: dict = {}


def _rotations() -> int:
    """90-degree orientation steps to search. Default ON: a 90-degree
    rotated drawing scan shipped with its logo intact when this was
    opt-in. Cost is bounded by the coarse scale grid."""
    return 4 if os.environ.get('PROJECT_P_LOGO_ROTATIONS', '1') == '1' \
        else 1


def _angles() -> Tuple[float, ...]:
    """Small skew angles (degrees) searched on top of each orientation."""
    raw = os.environ.get('PROJECT_P_LOGO_ANGLES', '0,5,-5')
    out = []
    for part in raw.split(','):
        try:
            out.append(float(part.strip()))
        except ValueError:
            continue
    return tuple(out) or (0.0,)


def _aspects() -> Tuple[float, ...]:
    """Horizontal squeeze factors for cylinder-wrapped marks."""
    raw = os.environ.get('PROJECT_P_LOGO_ASPECTS', '0.8')
    out = []
    for part in raw.split(','):
        try:
            f = float(part.strip())
        except ValueError:
            continue
        if 0.3 <= f < 1.0:
            out.append(f)
    return tuple(out)


def _max_hits_per_template() -> int:
    try:
        return int(os.environ.get('PROJECT_P_LOGO_MAX_HITS', '8'))
    except ValueError:
        return 8


def env_templates():
    """Templates from PROJECT_P_LOGO_TEMPLATES, cached by directory
    listing (name+mtime per entry — a max-mtime key survived DELETING a
    template, so a removed template kept matching)."""
    directory = templates_dir()
    if directory is None:
        return []
    key = (str(directory),
           tuple(sorted((p.name, p.stat().st_mtime)
                        for p in directory.iterdir() if p.is_file())))
    if key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE.clear()
        _TEMPLATE_CACHE[key] = load_templates(directory)
    return _TEMPLATE_CACHE[key]


def _threshold() -> float:
    """Match score cutoff. Genuine marks scored 0.71-0.75 in every live
    test, so 0.65 keeps full recall headroom while cutting the
    line-feature false-positive band just above 0.6."""
    try:
        return float(os.environ.get('PROJECT_P_LOGO_THRESHOLD', '0.65'))
    except ValueError:
        return 0.65


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
    """Load template crops as grayscale arrays (name, image).

    RGBA crops are flattened onto WHITE first: cv2's grayscale load
    drops alpha, so a mostly-transparent thumbnail would be matched on
    art the reviewer never saw (live: an alpha-invisible template fired
    a false box on a drawing)."""
    import cv2
    import numpy as np
    out = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _TEMPLATE_EXTS:
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            _logger.warning("Logo template unreadable: %s", path.name)
            continue
        if raw.ndim == 3 and raw.shape[2] == 4:
            alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
            rgb = raw[:, :, :3].astype(np.float32)
            flat = rgb * alpha + 255.0 * (1.0 - alpha)
            img = cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif raw.ndim == 3:
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            img = raw
        if min(img.shape[:2]) < _MIN_TEMPLATE_PX:
            _logger.warning("Logo template too small: %s", path.name)
            continue
        big = max(img.shape[:2])
        if big > _TEMPLATE_NORM_MAX_DIM:
            f = _TEMPLATE_NORM_MAX_DIM / float(big)
            img = cv2.resize(img, (max(_MIN_TEMPLATE_PX,
                                       int(img.shape[1] * f)),
                                   max(_MIN_TEMPLATE_PX,
                                       int(img.shape[0] * f))),
                             interpolation=cv2.INTER_AREA)
        out.append((path.name, img))
    return out


def _template_variants(tmpl):
    """Yield (variant_array, scale_grid) search variants of one template.

    Upright gets the full scale grid; 90-degree orientations, small
    skews and horizontal squeezes get the coarse grid (see _SCALES
    comment). Skew/squeeze are searched on the upright orientation only.
    """
    import cv2
    import numpy as np

    yield tmpl, _SCALES
    for rot in range(1, _rotations()):
        yield np.ascontiguousarray(np.rot90(tmpl, rot)), _COARSE_SCALES
    h, w = tmpl.shape[:2]
    for angle in _angles():
        if not angle:
            continue
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        mat[0, 2] += nw / 2.0 - center[0]
        mat[1, 2] += nh / 2.0 - center[1]
        rotated = cv2.warpAffine(tmpl, mat, (nw, nh),
                                 flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_REPLICATE)
        yield rotated, _COARSE_SCALES
    for factor in _aspects():
        squeezed = cv2.resize(tmpl, (max(_MIN_TEMPLATE_PX,
                                         int(w * factor)), h),
                              interpolation=cv2.INTER_AREA)
        yield squeezed, _COARSE_SCALES


def _iou_clash(box, kept, thresh=0.3) -> int:
    """Index of the first box in `kept` overlapping `box` (IoU >
    thresh), or -1 when none does. Compare with >= 0 — index 0 is a
    valid clash."""
    x, y, w, h = box
    for i, (kx, ky, kw, kh) in enumerate(kept):
        ix = max(0, min(x + w, kx + kw) - max(x, kx))
        iy = max(0, min(y + h, ky + kh) - max(y, ky))
        inter = ix * iy
        union = w * h + kw * kh - inter
        if union > 0 and inter / union > thresh:
            return i
    return -1


def find_logo_boxes(pil_image, templates,
                    reject=None) -> List[Tuple[int, int, int, int]]:
    """Match every template against a PIL image.

    Returns deduplicated (x, y, w, h) boxes in FULL-RESOLUTION image
    coordinates. Empty list when nothing matches or cv2 is missing.

    `reject`: optional callback (x, y, w, h) -> bool applied DURING the
    cross-template merge, in score order. A rejected candidate is
    skipped WITHOUT suppressing overlapping lower-scored candidates —
    filtering after NMS lost true catches whenever a bad higher-scored
    box overlapped one (live: an etched mark vanished because its
    suppressor was later rejected as text). The junk-template guard
    runs on per-template deduped boxes BEFORE the merge for the same
    reason.
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
    # Thin line art (outline/engraved marks are single-pixel strokes at
    # match resolution) makes normalized correlation collapse under
    # slight scale/warp error. A light blur on BOTH sides thickens
    # strokes into gradients and buys the tolerance the scale grid and
    # skew steps rely on.
    page = cv2.GaussianBlur(page, (3, 3), 0)
    threshold = _threshold()

    # (x, y, w, h, score, template_name)
    boxes: List[Tuple[int, int, int, int, float, str]] = []
    for name, tmpl in templates:
        for t, scale_grid in _template_variants(tmpl):
            for scale in scale_grid:
                tw = int(t.shape[1] * scale)
                th = int(t.shape[0] * scale)
                if (tw < _MIN_TEMPLATE_PX or th < _MIN_TEMPLATE_PX
                        or tw >= page.shape[1] or th >= page.shape[0]):
                    continue
                if min(tw, th) < _MIN_MATCH_SHORT_SIDE:
                    continue
                t_scaled = cv2.GaussianBlur(
                    cv2.resize(t, (tw, th),
                               interpolation=cv2.INTER_AREA), (3, 3), 0)
                if int(t_scaled.std()) == 0:
                    continue
                for variant in _polarities(t_scaled):
                    result = cv2.matchTemplate(page, variant,
                                               cv2.TM_CCOEFF_NORMED)
                    ys, xs = np.where(result >= threshold)
                    if len(xs) > _MAX_RAW_PEAKS:
                        top = np.argsort(result[ys, xs])[-_MAX_RAW_PEAKS:]
                        ys, xs = ys[top], xs[top]
                    for x, y in zip(xs, ys):
                        boxes.append((int(x * ratio), int(y * ratio),
                                      int(tw * ratio), int(th * ratio),
                                      float(result[y, x]), name))

    # Stage 1: per-template NMS (dedup one mark hit at many scales/
    # variants), then the junk-template guard on the deduped counts. A
    # real stamp appears a handful of times per page; a template that
    # "matches" dozens of regions or blankets the sheet is a bad
    # enrollment (product photo, whole-page thumbnail) — drop its
    # matches loudly rather than obliterate drawing content.
    max_hits = _max_hits_per_template()
    page_area = float(full_w * full_h)
    by_template: dict = {}
    for box in boxes:
        by_template.setdefault(box[5], []).append(box)
    # Each cluster: winner box + a few distinct-extent lower-scored
    # alternates of the SAME region. Alternates matter because the
    # reject callback runs in stage 2: a winner whose slightly larger
    # extent drifts onto adjacent text gets rejected — without
    # alternates a single-template region would then be lost entirely
    # (live: the sole deduped winner rejected -> logo shipped).
    survivors: List[dict] = []
    for name, tboxes in by_template.items():
        tboxes.sort(key=lambda b: -b[4])
        clusters: List[dict] = []
        for x, y, w, h, score, _n in tboxes:
            idx = _iou_clash((x, y, w, h),
                             [c['box'] for c in clusters])
            if idx < 0:
                clusters.append({'box': (x, y, w, h), 'score': score,
                                 'name': name, 'alts': []})
            elif len(clusters[idx]['alts']) < 4:
                alt = (x, y, w, h)
                if alt != clusters[idx]['box'] \
                        and alt not in clusters[idx]['alts']:
                    clusters[idx]['alts'].append(alt)
        area = sum(c['box'][2] * c['box'][3]
                   for c in clusters) / page_area
        if len(clusters) > max_hits or area > 0.08:
            _logger.warning(
                "Logo template %s looks pathological on this page "
                "(%d boxes, %.1f%% of page) — matches DROPPED. Curate "
                "the template set: only actual marks belong in %s.",
                name, len(clusters), area * 100,
                os.environ.get('PROJECT_P_LOGO_TEMPLATES', ''))
            continue
        survivors.extend(clusters)

    # Stage 2: cross-template merge in score order. A rejected
    # candidate is skipped WITHOUT entering `kept` (it cannot suppress
    # a genuine lower-scored catch), and its own cluster's alternates
    # are tried in order before the region is given up.
    survivors.sort(key=lambda c: -c['score'])
    kept: List[Tuple[int, int, int, int]] = []
    for cluster in survivors:
        if _iou_clash(cluster['box'], kept) >= 0:
            continue
        for cand in [cluster['box']] + cluster['alts']:
            if _iou_clash(cand, kept) >= 0:
                break  # region already covered by a kept box
            x, y, w, h = cand
            if reject is not None and reject(x, y, w, h):
                _logger.debug(
                    "logo candidate %s score=%.3f box=(%d,%d,%d,%d) "
                    "rejected by caller — trying alternates",
                    cluster['name'], cluster['score'], x, y, w, h)
                continue
            _logger.debug("logo candidate %s score=%.3f "
                          "box=(%d,%d,%d,%d)", cluster['name'],
                          cluster['score'], x, y, w, h)
            kept.append(cand)
            break
    return kept
