"""Embedded-media inventory: extract, deduplicate, cluster, rank.

Logos in a document corpus are not a detection problem. They are a
*deduplication* problem: the letterhead mark is not a similar image in
each file, it is byte-for-byte the same image part copied into every
file that used the template. Collapse the corpus by content hash and a
few thousand documents typically reduce to a few dozen distinct images
-- small enough for a human to review exhaustively, which is a stronger
guarantee than any detector threshold.

Pipeline
--------
1. Enumerate media parts without rendering anything. OOXML files are
   ZIP archives whose images live at ``word/media/``, ``ppt/media/``,
   ``xl/media/``; PDFs store them as image XObjects. Neither requires
   rasterizing a page.
2. Exact grouping. ZIP central directories already carry a CRC32 and
   an uncompressed size per member, so identical parts can be grouped
   without reading their bytes at all. Only one representative per
   (crc, size) bucket is read and SHA-256'd.
3. Near-duplicate grouping. Only the *distinct* SHA-256s get decoded
   for a perceptual hash, so rescaled copies merge without paying
   decode cost on every occurrence.
4. Ranking. Occurrence spread, placement in headers/footers/slide
   masters, small dimensions, flat palette and alpha are all weak
   individual signals that together float template chrome to the top
   of the review queue.

Nothing here decides what a logo is. It decides what a human has to
look at.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

_logger = logging.getLogger(__name__)

OOXML_SUFFIXES = {
    '.docx', '.docm', '.dotx', '.dotm',
    '.pptx', '.pptm', '.potx', '.potm', '.ppsx',
    '.xlsx', '.xlsm', '.xltx', '.xltm',
    '.vsdx', '.odt', '.ods', '.odp',
}

STANDALONE_IMAGE_SUFFIXES = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp',
}

# Parts inside an OOXML container that hold pictures. ODF uses Pictures/.
_MEDIA_MEMBER = re.compile(
    r'^(?:word|ppt|xl|visio)/media/|^Pictures/', re.IGNORECASE)

# Container parts that constitute page/slide *chrome* rather than body
# content. An image referenced from one of these is template furniture,
# which is where letterheads and slide-master logos live.
_CHROME_PART = re.compile(
    r'^word/(?:header|footer)\d*\.xml$'
    r'|^ppt/(?:slideMasters|slideLayouts|notesMasters)/'
    r'|^xl/(?:drawings/)?.*header',
    re.IGNORECASE)

_REL_TARGET = re.compile(rb'Target="([^"]+)"')

# Perceptual-hash distance at or below which two images are treated as
# the same picture. Kept deliberately tight: logos are small, flat and
# high-contrast, which is precisely the regime where dHash over-merges
# genuinely different marks.
PHASH_MAX_DISTANCE = 4


@dataclass
class MediaOccurrence:
    """One appearance of one image inside one document."""

    document: Path
    member: str
    size: int
    crc: Optional[int] = None
    in_chrome: bool = False

    def as_dict(self, root: Optional[Path] = None) -> dict:
        doc = self.document
        if root is not None:
            try:
                doc = self.document.relative_to(root)
            except ValueError:
                pass
        return {
            'document': str(doc),
            'member': self.member,
            'in_chrome': self.in_chrome,
        }


@dataclass
class MediaCluster:
    """A group of occurrences judged to be the same picture."""

    cluster_id: str
    sha256: List[str] = field(default_factory=list)
    phash: Optional[int] = None
    occurrences: List[MediaOccurrence] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    image_format: Optional[str] = None
    has_alpha: bool = False
    unique_colors: Optional[int] = None
    sample_bytes: Optional[bytes] = None
    decode_error: Optional[str] = None

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)

    @property
    def document_count(self) -> int:
        return len({occ.document for occ in self.occurrences})

    @property
    def chrome_count(self) -> int:
        return sum(1 for occ in self.occurrences if occ.in_chrome)

    def signals(self, corpus_size: int) -> Dict[str, bool]:
        longest = max(self.width or 0, self.height or 0)
        aspect = ((self.width / self.height)
                  if self.width and self.height else 0.0)
        return {
            'in_template_chrome': self.chrome_count > 0,
            'spread_across_corpus': (
                corpus_size > 0
                and self.document_count / corpus_size >= 0.2),
            'small': 0 < longest <= 600,
            'flat_palette': (self.unique_colors is not None
                             and self.unique_colors <= 64),
            'has_alpha': self.has_alpha,
            'wordmark_aspect': 1.5 <= aspect <= 8.0,
        }

    def score(self, corpus_size: int) -> int:
        """Ranking heuristic. Orders the review queue, decides nothing.

        Deliberately crude and deliberately visible in the output: a
        reviewer who disagrees with the ordering can still see every
        cluster, because every cluster is listed.
        """
        sig = self.signals(corpus_size)
        weights = {
            'in_template_chrome': 4,
            'spread_across_corpus': 3,
            'small': 2,
            'flat_palette': 2,
            'has_alpha': 1,
            'wordmark_aspect': 1,
        }
        return sum(w for k, w in weights.items() if sig[k])


def iter_source_files(source: Path) -> Iterator[Path]:
    """Every file under ``source`` that could carry an embedded image."""
    interesting = (OOXML_SUFFIXES | STANDALONE_IMAGE_SUFFIXES | {'.pdf'})
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.suffix.lower() in interesting:
            yield path


def _chrome_media_targets(archive: zipfile.ZipFile) -> set:
    """Media part names referenced from headers, footers or masters.

    Relationship files are small XML; scanning them is far cheaper than
    parsing the document body, and it is the only reliable way to tell
    a letterhead logo from a photograph pasted into the text.
    """
    targets = set()
    for info in archive.infolist():
        name = info.filename
        if not name.endswith('.rels') or '_rels/' not in name:
            continue
        owner = name.replace('_rels/', '').removesuffix('.rels')
        if not _CHROME_PART.search(owner):
            continue
        base = owner.rsplit('/', 1)[0] if '/' in owner else ''
        try:
            blob = archive.read(info)
        except (KeyError, zipfile.BadZipFile, RuntimeError):
            continue
        for match in _REL_TARGET.finditer(blob):
            target = match.group(1).decode('utf-8', 'replace')
            if target.startswith(('http://', 'https://')):
                continue
            resolved = target
            if target.startswith('../'):
                parent = base.rsplit('/', 1)[0] if '/' in base else ''
                resolved = (f'{parent}/{target[3:]}' if parent
                            else target[3:])
            elif not target.startswith('/'):
                resolved = f'{base}/{target}' if base else target
            targets.add(resolved.lstrip('/'))
    return targets


def scan_ooxml(path: Path) -> List[MediaOccurrence]:
    """Media parts in an OOXML/ODF container.

    Reads only the central directory and the ``.rels`` parts. Image
    bytes are not touched here -- CRC32 and uncompressed size come from
    the directory entry, which is enough to group exact duplicates.
    """
    out: List[MediaOccurrence] = []
    try:
        with zipfile.ZipFile(path) as archive:
            chrome = _chrome_media_targets(archive)
            for info in archive.infolist():
                if info.is_dir() or not _MEDIA_MEMBER.search(info.filename):
                    continue
                out.append(MediaOccurrence(
                    document=path,
                    member=info.filename,
                    size=info.file_size,
                    crc=info.CRC,
                    in_chrome=info.filename in chrome,
                ))
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        _logger.warning('Could not read container %s: %s', path, exc)
    return out


def scan_pdf(path: Path) -> List[MediaOccurrence]:
    """Image XObjects in a PDF, enumerated without rendering pages.

    PyMuPDF is the project's primary PDF backend; pypdf is the
    fallback. Poppler's ``pdfimages -list`` is faster still if the
    binary is available, but is not assumed here.
    """
    out: List[MediaOccurrence] = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None

    if fitz is not None:
        try:
            with fitz.open(path) as doc:
                seen = set()
                for page_no in range(doc.page_count):
                    for img in doc.get_page_images(page_no, full=True):
                        xref = img[0]
                        if xref in seen:
                            continue
                        seen.add(xref)
                        out.append(MediaOccurrence(
                            document=path,
                            member=f'xref:{xref}',
                            size=0,
                        ))
            return out
        except Exception as exc:
            _logger.warning('PyMuPDF failed on %s: %s', path, exc)

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages):
            resources = page.get('/Resources')
            if resources is None:
                continue
            xobjects = resources.get_object().get('/XObject')
            if xobjects is None:
                continue
            for name, ref in xobjects.get_object().items():
                obj = ref.get_object()
                if obj.get('/Subtype') != '/Image':
                    continue
                out.append(MediaOccurrence(
                    document=path,
                    member=f'p{page_no}:{name}',
                    size=int(obj.get('/Length', 0) or 0),
                ))
    except Exception as exc:
        _logger.warning('Could not read PDF %s: %s', path, exc)
    return out


def read_occurrence(occ: MediaOccurrence) -> Optional[bytes]:
    """Fetch the bytes for one occurrence. Called sparingly."""
    suffix = occ.document.suffix.lower()
    try:
        if suffix in OOXML_SUFFIXES:
            with zipfile.ZipFile(occ.document) as archive:
                return archive.read(occ.member)
        if suffix == '.pdf':
            return _read_pdf_image(occ)
        return occ.document.read_bytes()
    except Exception as exc:
        _logger.warning('Could not read %s from %s: %s',
                        occ.member, occ.document, exc)
        return None


def _read_pdf_image(occ: MediaOccurrence) -> Optional[bytes]:
    if occ.member.startswith('xref:'):
        try:
            import fitz
            xref = int(occ.member.split(':', 1)[1])
            with fitz.open(occ.document) as doc:
                return doc.extract_image(xref).get('image')
        except Exception:
            return None
    try:
        from pypdf import PdfReader
        page_part, name = occ.member.split(':', 1)
        page_no = int(page_part[1:])
        page = PdfReader(str(occ.document)).pages[page_no]
        xobjects = page['/Resources']['/XObject'].get_object()
        return xobjects[name].get_object().get_data()
    except Exception:
        return None


def dhash(image, size: int = 8) -> int:
    """Difference hash: 64 bits of horizontal-gradient sign.

    Cheap and rotation-intolerant, which is fine -- rescaled template
    art is the case this exists to catch. See PHASH_MAX_DISTANCE for
    the accompanying caveat about flat images.
    """
    from PIL import Image
    grey = image.convert('L').resize((size + 1, size), Image.LANCZOS)
    pixels = list(grey.getdata())
    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            bits <<= 1
            if pixels[offset + col] < pixels[offset + col + 1]:
                bits |= 1
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def _describe(blob: bytes) -> dict:
    """Decode once, at reduced scale, to collect ranking signals."""
    from PIL import Image
    info: dict = {}
    try:
        with Image.open(io.BytesIO(blob)) as img:
            info['image_format'] = img.format
            info['width'], info['height'] = img.size
            info['has_alpha'] = (
                img.mode in ('RGBA', 'LA', 'PA')
                or 'transparency' in img.info)
            # draft() pulls JPEGs straight out of the DCT at 1/N scale,
            # skipping a full-resolution decode.
            try:
                img.draft('RGB', (64, 64))
            except (AttributeError, ValueError):
                pass
            img.load()
            thumb = img.convert('RGB').resize((64, 64))
            info['unique_colors'] = len(set(thumb.getdata()))
            info['phash'] = dhash(img)
    except Exception as exc:
        info['decode_error'] = f'{type(exc).__name__}: {exc}'
    return info


def build_index(source: Path,
                progress: bool = False) -> Tuple[List[MediaCluster], int]:
    """Inventory every embedded image under ``source``.

    Returns the clusters (ranked, highest score first) and the number
    of documents scanned.
    """
    occurrences: List[MediaOccurrence] = []
    documents = 0

    for path in iter_source_files(source):
        suffix = path.suffix.lower()
        if suffix in OOXML_SUFFIXES:
            found = scan_ooxml(path)
        elif suffix == '.pdf':
            found = scan_pdf(path)
        else:
            found = [MediaOccurrence(document=path, member='<file>',
                                     size=path.stat().st_size)]
        if found:
            documents += 1
            occurrences.extend(found)
        if progress and documents and documents % 250 == 0:
            _logger.info('scanned %d documents, %d media parts',
                         documents, len(occurrences))

    _logger.info('Found %d media parts across %d documents',
                 len(occurrences), documents)

    # Stage 1: cheap grouping. Members sharing a CRC32 and an
    # uncompressed size are read once, not once per occurrence.
    cheap: Dict[tuple, List[MediaOccurrence]] = defaultdict(list)
    for occ in occurrences:
        key = ((occ.crc, occ.size) if occ.crc is not None
               else (None, id(occ)))
        cheap[key].append(occ)

    # Stage 2: confirm with SHA-256. CRC32 collides; a content hash
    # does not, for any corpus that is not adversarial toward us.
    by_sha: Dict[str, List[MediaOccurrence]] = defaultdict(list)
    sample: Dict[str, bytes] = {}
    for group in cheap.values():
        blob = read_occurrence(group[0])
        if blob is None:
            continue
        digest = hashlib.sha256(blob).hexdigest()
        by_sha[digest].extend(group)
        sample.setdefault(digest, blob)

    _logger.info('%d distinct images after exact deduplication',
                 len(by_sha))

    # Stage 3: decode the distinct set only, then merge near-duplicates.
    described = {sha: _describe(blob) for sha, blob in sample.items()}

    merged: List[MediaCluster] = []
    for sha, meta in sorted(described.items()):
        phash = meta.get('phash')
        target = None
        if phash is not None:
            for cluster in merged:
                if cluster.phash is None:
                    continue
                # Require a matching aspect bucket as well as a close
                # hash. dHash alone merges distinct simple marks.
                same_shape = (
                    cluster.width and meta.get('width')
                    and abs((cluster.width / max(cluster.height, 1))
                            - (meta['width']
                               / max(meta.get('height', 1), 1))) < 0.15)
                if (same_shape
                        and hamming(cluster.phash, phash)
                        <= PHASH_MAX_DISTANCE):
                    target = cluster
                    break
        if target is None:
            target = MediaCluster(
                cluster_id='',
                phash=phash,
                width=meta.get('width'),
                height=meta.get('height'),
                image_format=meta.get('image_format'),
                has_alpha=bool(meta.get('has_alpha')),
                unique_colors=meta.get('unique_colors'),
                sample_bytes=sample[sha],
                decode_error=meta.get('decode_error'),
            )
            merged.append(target)
        target.sha256.append(sha)
        target.occurrences.extend(by_sha[sha])

    merged.sort(key=lambda c: (-c.score(documents), -c.document_count))
    for index, cluster in enumerate(merged, start=1):
        cluster.cluster_id = f'media_{index:04d}'

    _logger.info('%d clusters after near-duplicate merge', len(merged))
    return merged, documents
