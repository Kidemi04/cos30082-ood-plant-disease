"""HTML for the result panels, following the mixed design on the canvas.

Everything here is plain string building, so the look can be changed without touching the models.
Colours: accent green, crop blue, disease amber; amber badge = unseen pair, blue badge = seen pair.
"""
import base64
from functools import lru_cache
from html import escape

from predictor import ASSETS

ACCENT, CROP, DISEASE = '#1F6B45', '#2D5F8A', '#9A4F0E'
LEAF_SVG = ('<svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 '
            '5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"></path><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 '
            '12 13 13 12"></path></svg>')


def pct(p):
    if p >= 0.995:
        return '>99%'
    return f'{p * 100:.0f}%' if p >= 0.01 else '<1%'


def ordinal(n):
    return f'{n}{"th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'


@lru_cache(maxsize=4096)
def thumb_uri(rel):
    path = ASSETS / rel
    if not path.is_file():
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def header(device_label):
    return f'''
<div class="pd-brand">
  <div class="pd-logo">{LEAF_SVG}</div>
  <div>
    <h1>ML Project - COS30082</h1>
    <p>Out-Of-Distribution Plant Disease Identification</p>
  </div>
  <span class="pd-chip pd-chip-blue">{escape(device_label)}</span>
</div>'''


def badge(unseen, text=None):
    cls = 'pd-badge-unseen' if unseen else 'pd-badge-seen'
    return f'<span class="pd-badge {cls}">{escape(text or ("Unseen" if unseen else "Seen"))}</span>'


# ---------------------------------------------------------------- start screen

def how_it_works():
    return f'''
<div class="pd-card">
  <p class="pd-eyebrow">How it works</p>
  <div class="pd-equation">
    <div class="pd-part" style="border-top-color:{CROP}">
      <p class="pd-part-label" style="color:{CROP}">Crop</p>
      <p class="pd-part-name">55 crops</p>
      <p class="pd-part-sub">Tomato, olive, cassava…</p>
    </div>
    <div class="pd-op" aria-hidden="true">+</div>
    <div class="pd-part" style="border-top-color:{DISEASE}">
      <p class="pd-part-label" style="color:{DISEASE}">Disease</p>
      <p class="pd-part-name">175 diseases</p>
      <p class="pd-part-sub">Early blight, rust, mosaic…</p>
    </div>
    <div class="pd-op" aria-hidden="true">=</div>
    <div class="pd-pair">
      <p class="pd-part-label pd-amber">Pairs</p>
      <p class="pd-part-name">323 seen + 10 unseen</p>
      <p class="pd-pair-sub">The 10 unseen pairs were never in training.</p>
    </div>
  </div>
  <p class="pd-note">Upload a leaf photo or pick an example. The model recognises the crop and the disease
  separately, then combines them, which is how it can name a pair it has never seen.</p>
</div>'''


# ---------------------------------------------------------------- identify tab

def result_card(pred, seen_count=None):
    best, crop, disease = pred['best'], pred['crop'], pred['disease']
    unseen = best['unseen']
    pair_label = 'Pair · unseen in training' if unseen else 'Pair · seen in training'
    pair_cls = 'pd-amber' if unseen else 'pd-skyblue'
    if unseen:
        note = (f'The model never trained on a {escape(crop["name"].lower())} leaf with '
                f'{escape(disease["name"].lower())}. It recognised the crop and the disease separately, '
                'then combined them.')
    else:
        photos = f'{seen_count} photos' if seen_count else 'photos'
        note = f'The model trained on {photos} of this pair, so the crop, the disease and the pair all agree.'
    if not pred['calibrated']:
        note += (' <strong>This model is not calibrated yet</strong> (no gamma value), so it rarely picks '
                 'unseen pairs.')
    rows = ''.join(top_row(i + 1, t) for i, t in enumerate(pred['top']))
    return f'''
<div class="pd-card">
  <div class="pd-row-between">
    <p class="pd-eyebrow">How the answer was built</p>
    <p class="pd-meta">{escape(pred['meta']['short'])} · {pred['seconds'] * 1000:.0f} ms on {escape(pred['device'])}</p>
  </div>
  <div class="pd-equation">
    <div class="pd-part" style="border-top-color:{CROP}">
      <p class="pd-part-label" style="color:{CROP}">Crop</p>
      <p class="pd-part-name">{escape(crop['name'])}</p>
      <p class="pd-part-sub">{pct(crop['p'])} · next: {escape(crop['next'][0])} {pct(crop['next'][1])}</p>
    </div>
    <div class="pd-op" aria-hidden="true">+</div>
    <div class="pd-part" style="border-top-color:{DISEASE}">
      <p class="pd-part-label" style="color:{DISEASE}">Disease</p>
      <p class="pd-part-name">{escape(disease['name'])}</p>
      <p class="pd-part-sub">{pct(disease['p'])} · next: {escape(disease['next'][0])} {pct(disease['next'][1])}</p>
    </div>
    <div class="pd-op" aria-hidden="true">=</div>
    <div class="pd-pair">
      <p class="pd-part-label {pair_cls}">{pair_label}</p>
      <p class="pd-part-name">{escape(best['title'])}</p>
      <p class="pd-pair-sub"><strong>{pct(best['p'])}</strong> confidence · 1st of 333</p>
    </div>
  </div>
  <p class="pd-note">{note}</p>
  <h2 class="pd-h2">Top 5</h2>
  <div class="pd-top5">{rows}</div>
</div>'''


def top_row(rank, t):
    color = '#C46A1C' if t['unseen'] else ACCENT
    width = max(t['p'] * 100, 1)
    return f'''
<div class="pd-toprow">
  <span class="pd-rank">{rank}</span>
  <div class="pd-topname">
    <div class="pd-row-wrap"><span>{escape(t['title'])}</span>{badge(True) if t['unseen'] else ''}</div>
    <div class="pd-bar"><div style="width:{width:.1f}%;background:{color}"></div></div>
  </div>
  <span class="pd-kind">{'unseen pair' if t['unseen'] else 'seen pair'}</span>
  <span class="pd-pct">{pct(t['p'])}</span>
</div>'''


def figure(item, titles):
    rel, comp, sim = item
    return f'''
<figure class="pd-fig">
  <img src="{thumb_uri(rel)}" alt="{escape(titles(comp))}">
  <figcaption><span>{escape(titles(comp))}</span><span class="pd-sim">{sim:.2f}</span></figcaption>
</figure>'''


def similar_card(search, pred, titles):
    best = pred['best']
    if search['mode'] == 'none':
        return ('<div class="pd-card"><h2 class="pd-h2">Most similar training images</h2><p class="pd-meta">'
                'The image index is missing: run <code>python ui\\build_index.py</code>.</p></div>')
    if search['mode'] == 'seen':
        grid = ''.join(figure(i, titles) for i in search['images'])
        body = (f'<p class="pd-meta">Training photos of {escape(best["title"])}, closest first.</p>'
                f'<div class="pd-grid">{grid}</div>')
    else:
        crop_grid = ''.join(figure(i, titles) for i in search['crop'])
        dis_grid = ''.join(figure(i, titles) for i in search['disease'])
        body = f'''
<p class="pd-meta">No training images exist for this pair, so here are its two halves: the same crop, and the same disease.</p>
<h3 class="pd-h3" style="color:{CROP}">Same crop · {escape(pred['crop']['name'])}</h3>
<div class="pd-grid">{crop_grid}</div>
<h3 class="pd-h3" style="color:{DISEASE}">Same disease · {escape(pred['disease']['name'])}</h3>
<div class="pd-grid">{dis_grid}</div>'''
    return f'<div class="pd-card"><h2 class="pd-h2">Most similar training images</h2>{body}</div>'


# ---------------------------------------------------------------- compare tab

def compare_intro(truth):
    if truth:
        kind = 'an unseen pair' if truth['unseen'] else 'a seen pair'
        text = (f'True label: <strong>{escape(truth["title"])}</strong>, {kind}. Each model names the crop and '
                'the disease, then combines them.')
    else:
        text = ('No true label for this photo, so the models are not marked right or wrong. Pick an example to '
                'see them marked.')
    return f'<div class="pd-card pd-tight"><h2 class="pd-h2">Same image, every model</h2><p class="pd-note-plain">{text}</p></div>'


def compare_card(pred, truth):
    best, crop, disease = pred['best'], pred['crop'], pred['disease']
    if truth:
        crop_ok, dis_ok = best['crop'] == truth['crop'], best['disease'] == truth['disease']
        correct = best['id'] == truth['id']
        verdict = ('<span class="pd-verdict pd-ok">Correct</span>' if correct
                   else '<span class="pd-verdict pd-bad">Wrong</span>')
        mark = lambda ok: f'<span class="{"pd-ok-text" if ok else "pd-bad-text"}">{"Right" if ok else "Wrong"}</span>'  # noqa: E731
        rank = ordinal(pred['rank_of'][truth['id']])
        footer = f'{pct(best["p"])} confidence · true pair ranked {rank}'
        border = 'pd-winner' if correct else ''
    else:
        verdict, border, footer = '', '', f'{pct(best["p"])} confidence'
        mark = lambda ok: ''  # noqa: E731
        crop_ok = dis_ok = None
    calib = '' if pred['calibrated'] else '<p class="pd-meta">Not calibrated yet (no gamma value).</p>'
    pair_label = 'Pair · unseen in training' if best['unseen'] else 'Pair · seen in training'
    return f'''
<article class="pd-card pd-model {border}">
  <div class="pd-row-between">
    <div><h3 class="pd-model-name">{escape(pred['meta']['short'])}</h3><p class="pd-meta">{escape(pred['meta']['arch'])}</p></div>
    {verdict}
  </div>
  <div class="pd-mini">
    <div class="pd-part" style="border-top-color:{CROP}">
      <p class="pd-part-label" style="color:{CROP}">Crop</p>
      <p class="pd-mini-name">{escape(crop['name'])}</p>
      <p class="pd-part-sub">{pct(crop['p'])} {mark(crop_ok)}</p>
    </div>
    <div class="pd-part" style="border-top-color:{DISEASE}">
      <p class="pd-part-label" style="color:{DISEASE}">Disease</p>
      <p class="pd-mini-name">{escape(disease['name'])}</p>
      <p class="pd-part-sub">{pct(disease['p'])} {mark(dis_ok)}</p>
    </div>
  </div>
  <div class="pd-pair">
    <p class="pd-part-label {'pd-amber' if best['unseen'] else 'pd-skyblue'}">{pair_label}</p>
    <p class="pd-mini-name">{escape(best['title'])}</p>
    <p class="pd-pair-sub">{footer}</p>
  </div>
  {calib}
</article>'''


def compare_grid(preds, truth):
    """Baselines in one row, proposed approaches in the next."""
    html = ''
    for group, title in (('baseline', 'Baselines'), ('proposed', 'Proposed approaches')):
        cards = ''.join(compare_card(p, truth) for p in preds if p['meta'].get('group') == group)
        if cards:
            html += f'<p class="pd-eyebrow pd-group">{title}</p><div class="pd-compare">{cards}</div>'
    return html


def results_table(meta):
    """Each model's own calibrated test results (numbers recorded in ui/config.json with their source)."""
    rows = [(k, m) for k, m in meta.items() if m.get('results')]
    rows.sort(key=lambda km: km[1].get('group') != 'baseline')
    if not rows:
        return ''
    calibrated = [m['results']['h'] for _, m in rows if m['results'].get('calibrated', True)]
    best = max(calibrated) if calibrated else None
    body = ''
    for _, m in rows:
        r = m['results']
        cal = r.get('calibrated', True)
        strong = ' class="pd-strong"' if cal and r['h'] == best else ''
        note = '' if cal else ' <span class="pd-meta">(not calibrated)</span>'
        body += (f'<tr{strong}><td>{escape(m["label"])}{note}</td><td>{r["seen"]:.2f}%</td><td>{r["unseen"]:.2f}%</td>'
                 f'<td>{r["h"]:.2f}%</td><td>{r["h5"]:.2f}%</td></tr>')
    return f'''
<div class="pd-card">
  <h2 class="pd-h2">How they do on the full test sets</h2>
  <div class="pd-scroll"><table class="pd-table">
    <thead><tr><th>Model</th><th>Seen Top-1</th><th>Unseen Top-1</th><th>Harmonic mean</th><th>H Top-5</th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>
  <p class="pd-meta">Per-class accuracy over all 333 pairs, with each model's own calibration, from the team's result files.
  Bold = highest harmonic mean.</p>
</div>'''


def error_card(message):
    return f'<div class="pd-card"><p class="pd-bad-text"><strong>Something went wrong.</strong> {escape(message)}</p></div>'
