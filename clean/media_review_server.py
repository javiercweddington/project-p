"""Local review server for media_review.json.

Standard library only. Serves the review sheet as a page and accepts
the edited decisions back, writing them to the same file --serve read,
so the round trip through a local download folder disappears.

Bound to 127.0.0.1 by design: this process serves the extracted logo
art and the source document paths, which is exactly the material the
pipeline exists to keep off the network. Reach it by forwarding the
port over SSH (VS Code Remote-SSH offers this automatically), never by
binding a public interface.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Media review &mdash; __PROJECT__</title>
<style>
:root{--bg:#faf9f7;--card:#fff;--ink:#1b1b19;--mute:#6b6a66;--line:#dfddd6;
--keep:#1d7a55;--redact:#a32d2d;--warn:#8a5a0b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--bg);
border-bottom:1px solid var(--line);padding:14px 20px;
display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;font-weight:500;margin:0}
.sub{color:var(--mute);font-size:13px}
.spacer{flex:1}
button{font:inherit;padding:7px 13px;border:1px solid var(--line);
background:var(--card);border-radius:7px;cursor:pointer}
button:hover{border-color:#b9b7ae}
button.primary{background:var(--ink);color:#fff;border-color:var(--ink)}
button.primary:disabled{opacity:.45;cursor:default}
#status{font-size:13px;color:var(--mute);min-width:150px}
main{padding:20px;display:grid;gap:16px;
grid-template-columns:repeat(auto-fill,minmax(290px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;
padding:13px;display:flex;flex-direction:column;gap:9px}
.card.sel{border-color:var(--ink);box-shadow:0 0 0 2px rgba(27,27,25,.12)}
.card.done-keep{border-left:4px solid var(--keep)}
.card.done-redact{border-left:4px solid var(--redact)}
.thumb{height:150px;display:flex;align-items:center;justify-content:center;
background:repeating-conic-gradient(#eee 0 25%,#fff 0 50%) 0 0/16px 16px;
border-radius:7px;overflow:hidden}
.thumb img{max-width:100%;max-height:150px;object-fit:contain}
.id{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mute)}
.meta{font-size:13px;color:var(--mute)}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{font-size:11px;padding:2px 7px;border-radius:20px;
background:#f0eee7;color:#55544f}
.chip.hot{background:#fbeee0;color:var(--warn)}
.acts{display:flex;gap:6px;margin-top:auto}
.acts button{flex:1;padding:7px 0}
.acts button[aria-pressed=true][data-a=keep]{background:var(--keep);
color:#fff;border-color:var(--keep)}
.acts button[aria-pressed=true][data-a=redact]{background:var(--redact);
color:#fff;border-color:var(--redact)}
details{font-size:12px;color:var(--mute)}
details pre{white-space:pre-wrap;word-break:break-all;margin:6px 0 0;
font:11px ui-monospace,Menlo,monospace}
.hide{display:none}
kbd{font:11px ui-monospace,Menlo,monospace;background:#f0eee7;
padding:1px 5px;border-radius:4px}
</style></head><body>
<header>
  <div><h1>Media review &mdash; __PROJECT__</h1>
  <div class="sub">__DOCS__ documents &middot; __OCC__ embedded images
  &middot; __N__ distinct pictures</div></div>
  <div class="spacer"></div>
  <button onclick="filt('all')">All</button>
  <button onclick="filt('review')">Undecided</button>
  <button onclick="keepRest()">Keep all undecided</button>
  <span id="status"></span>
  <button class="primary" id="save" onclick="save()">Save decisions</button>
</header>
<main id="grid"></main>
<script>
const DATA = __DATA__;
let sel = 0, mode = 'all';
const grid = document.getElementById('grid');

function card(c, i) {
  const sig = Object.entries(c.signals || {}).filter(([, v]) => v);
  const hot = new Set(['in_template_chrome', 'spread_across_corpus']);
  return `<div class="card" id="c${i}" data-a="${c.action}">
    <div class="thumb">${c.thumb
      ? `<img src="${c.thumb}" alt="${c.id}">`
      : '<span class="meta">no preview</span>'}</div>
    <div class="id">${c.id} &middot; score ${c.score}</div>
    <div class="meta">${c.width}&times;${c.height} ${c.format || ''} &middot;
      ${c.occurrences} uses in ${c.documents} docs${
      c.in_chrome ? ` &middot; ${c.in_chrome} in headers/masters` : ''}</div>
    <div class="chips">${sig.map(([k]) =>
      `<span class="chip${hot.has(k) ? ' hot' : ''}">${
        k.replace(/_/g, ' ')}</span>`).join('')}</div>
    <details><summary>where it appears</summary><pre>${
      (c.examples || []).map(e => e.document + (e.in_chrome ? '  [chrome]' : ''))
        .join('\\n')}</pre></details>
    <div class="acts">
      <button data-a="keep" aria-pressed="${c.action === 'keep'}"
        onclick="mark(${i},'keep')">Keep</button>
      <button data-a="redact" aria-pressed="${c.action === 'redact'}"
        onclick="mark(${i},'redact')">Redact</button>
    </div></div>`;
}

function render() {
  grid.innerHTML = DATA.clusters.map(card).join('');
  DATA.clusters.forEach((c, i) => paint(i));
  status();
}

function paint(i) {
  const c = DATA.clusters[i], el = document.getElementById('c' + i);
  if (!el) return;
  el.className = 'card' + (i === sel ? ' sel' : '')
    + (c.action === 'keep' ? ' done-keep' : '')
    + (c.action === 'redact' ? ' done-redact' : '')
    + (mode === 'review' && c.action !== 'review' ? ' hide' : '');
  el.querySelectorAll('.acts button').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.a === c.action));
}

function mark(i, a) {
  const c = DATA.clusters[i];
  c.action = (c.action === a) ? 'review' : a;
  sel = i; paint(i); status(); dirty = true;
}

function keepRest() {
  DATA.clusters.forEach((c, i) => {
    if (c.action === 'review') { c.action = 'keep'; paint(i); }
  });
  dirty = true; status();
}

function filt(m) {
  mode = m; DATA.clusters.forEach((_, i) => paint(i));
}

let dirty = false;
function status() {
  const left = DATA.clusters.filter(c => c.action === 'review').length;
  const red = DATA.clusters.filter(c => c.action === 'redact').length;
  document.getElementById('status').textContent =
    left ? `${left} undecided \\u00b7 ${red} to redact`
         : `all decided \\u00b7 ${red} to redact`;
  document.getElementById('save').disabled = false;
}

async function save() {
  const btn = document.getElementById('save');
  btn.disabled = true; btn.textContent = 'Saving\\u2026';
  try {
    const r = await fetch('/save', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({clusters: DATA.clusters.map(
        c => ({id: c.id, action: c.action}))})
    });
    const out = await r.json();
    if (!r.ok) throw new Error(out.error || r.statusText);
    dirty = false;
    btn.textContent = 'Saved';
    document.getElementById('status').textContent =
      `written to ${out.path} \\u00b7 ${out.undecided} undecided`;
    setTimeout(() => { btn.textContent = 'Save decisions'; }, 1600);
  } catch (e) {
    btn.textContent = 'Save failed';
    document.getElementById('status').textContent = String(e);
  } finally { btn.disabled = false; }
}

addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const vis = DATA.clusters.map((c, i) => i)
    .filter(i => mode !== 'review' || DATA.clusters[i].action === 'review');
  const at = vis.indexOf(sel);
  if (e.key === 'j' || e.key === 'ArrowRight') {
    const p = sel; sel = vis[Math.min(at + 1, vis.length - 1)] ?? sel;
    paint(p); paint(sel);
    document.getElementById('c' + sel)
      ?.scrollIntoView({block: 'nearest'});
  } else if (e.key === 'k' || e.key === 'ArrowLeft') {
    const p = sel; sel = vis[Math.max(at - 1, 0)] ?? sel;
    paint(p); paint(sel);
    document.getElementById('c' + sel)
      ?.scrollIntoView({block: 'nearest'});
  } else if (e.key === 'r') { mark(sel, 'redact'); }
  else if (e.key === 'a') { mark(sel, 'keep'); }
  else if (e.key === 's' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault(); save();
  }
});

addEventListener('beforeunload', e => {
  if (dirty) { e.preventDefault(); e.returnValue = ''; }
});

render();
</script></body></html>
"""


def _thumb_data_uri(path: Optional[Path]) -> Optional[str]:
    if not path or not path.is_file():
        return None
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    return 'data:image/png;base64,' + base64.b64encode(blob).decode('ascii')


def render_page(review: dict, audit_dir: Path) -> bytes:
    """Build the review page with thumbnails inlined as data URIs."""
    clusters = []
    for entry in review.get('clusters', []):
        item = dict(entry)
        thumb = entry.get('thumbnail')
        item['thumb'] = _thumb_data_uri(
            audit_dir / thumb if thumb else None)
        clusters.append(item)

    payload = {'clusters': clusters}
    html = (PAGE
            .replace('__PROJECT__', str(review.get('project', 'corpus')))
            .replace('__DOCS__', str(review.get('documents_scanned', '?')))
            .replace('__OCC__', str(review.get('total_occurrences', '?')))
            .replace('__N__', str(len(clusters)))
            .replace('__DATA__', json.dumps(payload)))
    return html.encode('utf-8')


def _write_back(review_path: Path, decisions: list) -> dict:
    """Merge actions into the on-disk review sheet, keeping a backup.

    Only the ``action`` field is taken from the browser. Everything
    else -- hashes, occurrence lists, dimensions -- is whatever the
    scan wrote, so a stale or tampered page cannot change what a
    decision applies to.
    """
    with open(review_path) as handle:
        review = json.load(handle)

    allowed = {'keep', 'redact', 'review'}
    by_id = {str(d.get('id')): d.get('action') for d in decisions}
    changed = 0
    for entry in review.get('clusters', []):
        action = by_id.get(entry['id'])
        if action in allowed and action != entry.get('action'):
            entry['action'] = action
            changed += 1
    review['reviewed'] = datetime.now(timezone.utc).isoformat(
        timespec='seconds')

    backup = review_path.with_suffix('.json.bak')
    shutil.copy2(review_path, backup)
    tmp = review_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as handle:
        json.dump(review, handle, indent=2, ensure_ascii=False)
    tmp.replace(review_path)

    undecided = sum(1 for c in review.get('clusters', [])
                    if c.get('action') == 'review')
    return {'changed': changed, 'undecided': undecided,
            'path': str(review_path)}


def serve(review_path: Path, audit_dir: Path, port: int = 8000,
          host: str = '127.0.0.1') -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            _logger.debug(fmt, *args)

        def _send(self, code, body, ctype='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path not in ('/', '/index.html'):
                self._send(404, b'{"error":"not found"}')
                return
            try:
                with open(review_path) as handle:
                    review = json.load(handle)
            except (OSError, ValueError) as exc:
                self._send(500, json.dumps(
                    {'error': str(exc)}).encode())
                return
            self._send(200, render_page(review, audit_dir),
                       'text/html; charset=utf-8')

        def do_POST(self):
            if self.path != '/save':
                self._send(404, b'{"error":"not found"}')
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length <= 0 or length > 8 * 1024 * 1024:
                    raise ValueError('bad content length')
                payload = json.loads(self.rfile.read(length))
                result = _write_back(review_path,
                                     payload.get('clusters', []))
            except Exception as exc:
                self._send(400, json.dumps({'error': str(exc)}).encode())
                return
            _logger.info('saved %d changes, %d undecided',
                         result['changed'], result['undecided'])
            self._send(200, json.dumps(result).encode())

    server = ThreadingHTTPServer((host, port), Handler)
    print(f'Review server on http://{host}:{port}/')
    print(f'  sheet:  {review_path}')
    print('  forward this port (VS Code PORTS panel, or '
          'ssh -L {p}:localhost:{p} user@host), then open it locally.'
          .format(p=port))
    print('  keys:   j/k move, a keep, r redact, ctrl-s save. '
          'Ctrl-C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        server.server_close()
