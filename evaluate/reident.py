#!/usr/bin/env python
"""Re-identification test: does the cleaned corpus still identify its
source?

Every other check in this repo measures whether specific strings were
removed. This one asks the question the project exists to answer: can
an adversary with the cleaned deliverable tell which client it came
from? Train a classifier to predict the source project from cleaned
documents; if it beats chance, something still separates the corpora
— and the top features name it.

    python -m evaluate.reident \
        --corpus FlatBit=~/cleaned/FlatBit_61061 \
        --corpus HoleSaw=~/cleaned/HoleSaw_70198 \
        --out /tmp/clean_audit/reident_report.json

Channels are ablated independently:
  text       extracted document text (the main event)
  filename   file/dir name tokens (flat output should reduce this to
             FILE_nnn — any signal here is a path-anonymization bug)
  structure  extension + size bucket (weak by design; a strong result
             here means the corpus SHAPE is identifying, which no
             string-level cleaner can fix)

Interpretation discipline: corpora about different products separate
on topic vocabulary even under perfect pseudonymization ('hole saw'
vs 'flat bit' is not a leak). Above-chance accuracy is a TRIGGER to
read the top features, not a verdict by itself. Placeholder tokens
([COMPANY_001]) are stripped before vectorizing so the placeholders
themselves cannot carry the signal.

Needs scikit-learn. With only 2 projects and few documents, treat the
accuracy as a smoke alarm, not an estimate.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r'\[[A-Z]+_\d{3}\]')
_MAX_CHARS = 100_000


def load_corpus(label: str, root: Path) -> List[Dict]:
    """One record per document: label, name channel, text channel."""
    from clean.llm_detect import extract_scannable_text

    docs = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name.startswith('.'):
            continue
        rel = path.relative_to(root)
        try:
            text = extract_scannable_text(path, max_chars=_MAX_CHARS) or ''
        except Exception as exc:
            _logger.debug('extract failed for %s: %s', rel, exc)
            text = ''
        text = _PLACEHOLDER_RE.sub(' ', text)
        size = path.stat().st_size
        bucket = 0
        while size > 10 ** (bucket + 3):
            bucket += 1
        docs.append({
            'label': label,
            'rel': str(rel),
            'text': text,
            'filename': ' '.join(re.split(r'[\W_]+', str(rel))),
            'structure': f'ext{path.suffix.lower() or "none"} '
                         f'size{bucket}',
        })
    return docs


def evaluate_channel(docs: List[Dict], channel: str,
                     folds: int) -> Dict:
    """Cross-validated source prediction from one channel."""
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    texts = [d[channel] for d in docs]
    labels = [d['label'] for d in docs]
    counts = Counter(labels)
    chance = max(counts.values()) / len(labels)

    if sum(1 for t in texts if t.strip()) < folds:
        return {'channel': channel, 'skipped': 'not enough content'}

    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                               sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                               min_df=2, sublinear_tf=True,
                               max_features=50_000)
    try:
        x_word = word_vec.fit_transform(texts)
        x_char = char_vec.fit_transform(texts)
    except ValueError:
        return {'channel': channel, 'skipped': 'vocabulary empty'}
    x = hstack([x_word, x_char]).tocsr()
    y = np.array(labels)

    n_splits = min(folds, min(counts.values()))
    if n_splits < 2:
        return {'channel': channel, 'skipped': 'a class has < 2 docs'}
    model = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(
        model, x, y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=13))

    # Fit on everything for the feature attribution — WHICH tokens
    # separate the corpora is the actionable output.
    model.fit(x, y)
    names = list(word_vec.get_feature_names_out()) + [
        f'char:{f}' for f in char_vec.get_feature_names_out()]
    top: Dict[str, List[Tuple[str, float]]] = {}
    coefs = model.coef_
    class_list = list(model.classes_)
    rows = ([(class_list[1], coefs[0]), (class_list[0], -coefs[0])]
            if len(class_list) == 2
            else list(zip(class_list, coefs)))
    for cls, row in rows:
        idx = np.argsort(row)[::-1][:15]
        top[str(cls)] = [(names[i], round(float(row[i]), 3))
                         for i in idx if row[i] > 0]

    return {
        'channel': channel,
        'documents': len(docs),
        'class_counts': dict(counts),
        'chance': round(chance, 3),
        'cv_accuracy_mean': round(float(scores.mean()), 3),
        'cv_accuracy_std': round(float(scores.std()), 3),
        'folds': n_splits,
        'beats_chance': bool(scores.mean() > chance + 0.05),
        'top_features': top,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Predict source project from cleaned output.')
    parser.add_argument('--corpus', action='append', required=True,
                        metavar='LABEL=DIR',
                        help='Cleaned corpus with its source label '
                             '(repeat; need >= 2)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--out', default=None,
                        help='Write the full JSON report here')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname).1s] %(message)s',
        datefmt='%H:%M:%S')
    for noisy in ('PIL', 'pypdf', 'pdfminer'):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    corpora: List[Tuple[str, Path]] = []
    for spec in args.corpus:
        if '=' not in spec:
            print(f'ERROR: bad --corpus {spec!r} (LABEL=DIR)',
                  file=sys.stderr)
            return 2
        label, _, raw = spec.partition('=')
        root = Path(raw).expanduser()
        if not root.is_dir():
            print(f'ERROR: not a directory: {root}', file=sys.stderr)
            return 2
        corpora.append((label.strip(), root))
    if len(corpora) < 2:
        print('ERROR: need at least 2 corpora to test '
              're-identification.', file=sys.stderr)
        return 2

    docs: List[Dict] = []
    for label, root in corpora:
        loaded = load_corpus(label, root)
        print(f'{label}: {len(loaded)} documents from {root}')
        docs.extend(loaded)

    report = {
        'generated': datetime.now(timezone.utc).isoformat(
            timespec='seconds'),
        'corpora': {label: str(root) for label, root in corpora},
        'channels': [],
    }
    print()
    for channel in ('text', 'filename', 'structure'):
        result = evaluate_channel(docs, channel, args.folds)
        report['channels'].append(result)
        if 'skipped' in result:
            print(f'{channel:10s} SKIPPED ({result["skipped"]})')
            continue
        verdict = 'SEPARABLE' if result['beats_chance'] else 'at chance'
        print(f'{channel:10s} accuracy {result["cv_accuracy_mean"]:.2f} '
              f'± {result["cv_accuracy_std"]:.2f} '
              f'(chance {result["chance"]:.2f}) -> {verdict}')
        if result['beats_chance']:
            for cls, feats in result['top_features'].items():
                head = ', '.join(name for name, _w in feats[:8])
                print(f'    {cls}: {head}')
    print()
    print('Read the features, not just the score: product vocabulary '
          'separating corpora is topic, not identity. Names, places, '
          'company fragments, or FILENAME/STRUCTURE separability are '
          'leaks.')

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f'Report: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
