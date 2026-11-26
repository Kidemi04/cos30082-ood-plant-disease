r"""Robustness of a trained model to field-like image distortions (Member E).

Each test image is loaded and preprocessed exactly as in evaluation (shorter side 256, centre crop 224).
One distortion is then applied at one of three severities, before normalisation. Distortions are seeded
per image, so every model sees exactly the same distorted images. Metrics follow evaluate.py: per-class
mean accuracy, every image ranked against all 333 compositions, and the harmonic mean of seen and unseen
composition accuracy. The model's calibration constant (gamma from calibration.json) is applied, so the
'clean' row reproduces the model's calibrated results from evaluate.py.

Only the final model of each run is evaluated; nothing here is used to choose hyperparameters.

  python scripts\robustness_eval.py --run runs\clip_bioclip_heads_final --name clip_bioclip_heads
  python scripts\robustness_eval.py --run runs\clip_bioclip_heads_final --name check --only clean
  python scripts\robustness_eval.py --examples docs\robustness\distortion_examples.png

Writes results/robustness/<name>/robustness.csv and summary.md. The 224 px test crops are cached once in
runs/robustness_cache/ (git-ignored), so later models reuse them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from dataset_access import load_labels, load_split  # noqa: E402
from evaluate import Taxonomy, accuracy, harmonic_mean, level_scores, top_k_hits  # noqa: E402
from preprocessing import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO, load_image  # noqa: E402

IMAGE_SIZE = 224
SEED = 30082
SPLITS = ('test_seen', 'test_unseen')
CACHE = ROOT / 'runs' / 'robustness_cache'

# name: (what it simulates, three severity parameters)
DISTORTIONS = {
    'dark': ('low light, shade', (0.6, 0.4, 0.25)),                    # brightness factor
    'bright': ('direct sun, overexposure', (1.4, 1.8, 2.3)),           # brightness factor
    'low_contrast': ('haze, flat lighting', (0.6, 0.4, 0.25)),         # contrast factor
    'gaussian_blur': ('out of focus', (1.0, 2.0, 3.5)),                # blur radius in pixels
    'motion_blur': ('camera shake, moving leaf', (7, 13, 21)),         # blur length in pixels
    'gaussian_noise': ('low-light sensor noise', (10, 20, 35)),        # noise std on the 0-255 scale
    'jpeg': ('heavy compression (messaging apps)', (30, 15, 7)),       # JPEG quality
    'low_resolution': ('small or distant photo', (112, 64, 40)),       # downscale size before upscaling
    'occlusion': ('fingers, other leaves, debris', (0.05, 0.12, 0.25)),  # occluded area fraction
    'rotation': ('tilted camera, viewpoint', (15, 30, 45)),            # rotation angle in degrees
}
# Distortions with no close counterpart in the 'strong' training augmentation: a fairer test of whether
# robustness transfers beyond the exact perturbations seen in training.
HELD_OUT = {'motion_blur', 'gaussian_noise', 'jpeg', 'low_resolution', 'rotation'}


# ---------------------------------------------------------------- distortions

def distort(image, name, severity, rng):
    """Apply one distortion to a 224 px RGB PIL image. severity is 1, 2 or 3."""
    p = DISTORTIONS[name][1][severity - 1]
    if name in ('dark', 'bright'):
        return ImageEnhance.Brightness(image).enhance(p)
    if name == 'low_contrast':
        return ImageEnhance.Contrast(image).enhance(p)
    if name == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur(p))
    if name == 'motion_blur':
        a = np.asarray(image, dtype=np.float32)
        axis = int(rng.integers(2))  # vertical (0) or horizontal (1) shake
        pad = [(0, 0)] * 3
        pad[axis] = (p // 2, p - 1 - p // 2)
        c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = a.shape[axis]
        out = (np.take(c, range(p, p + n), axis=axis) - np.take(c, range(n), axis=axis)) / p
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if name == 'gaussian_noise':
        a = np.asarray(image, dtype=np.float32) + rng.normal(0, p, (IMAGE_SIZE, IMAGE_SIZE, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if name == 'jpeg':
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=p)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    if name == 'low_resolution':
        small = image.resize((p, p), Image.Resampling.BILINEAR)
        return small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    if name == 'occlusion':
        area = p * IMAGE_SIZE * IMAGE_SIZE
        ratio = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        h = int(min(IMAGE_SIZE, round(np.sqrt(area * ratio))))
        w = int(min(IMAGE_SIZE, round(np.sqrt(area / ratio))))
        top, left = int(rng.integers(0, IMAGE_SIZE - h + 1)), int(rng.integers(0, IMAGE_SIZE - w + 1))
        out = image.copy()
        ImageDraw.Draw(out).rectangle([left, top, left + w - 1, top + h - 1],
                                      fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
        return out
    if name == 'rotation':
        angle = p * (1 if rng.integers(2) else -1)
        fill = tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(0))  # mean colour, not black
        return image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)
    raise ValueError(f'Unknown distortion {name!r}')


def conditions(only=None):
    """('clean', 0) plus every (distortion, severity), optionally filtered by name."""
    out = [('clean', 0)] + [(name, s) for name in DISTORTIONS for s in (1, 2, 3)]
    return [c for c in out if not only or c[0] in only]


# ---------------------------------------------------------------- data

def cached_crops(split, limit=None):
    """uint8 array (N, 224, 224, 3) of evaluation crops, plus the split records, cached on disk."""
    records = load_split(split)
    path, names_path = CACHE / f'{split}.npy', CACHE / f'{split}_filenames.json'
    names = [r['filename'] for r in records]
    if path.is_file() and names_path.is_file() and json.loads(names_path.read_text(encoding='utf-8')) == names:
        crops = np.load(path, mmap_mode='r')
    else:
        print(f'Caching {len(records)} {split} crops (once) ...', flush=True)
        resize = round(IMAGE_SIZE / RESIZE_RATIO)
        prep = T.Compose([T.Resize(resize, antialias=True), T.CenterCrop(IMAGE_SIZE)])

        def one(record):
            return np.asarray(prep(load_image(record['image_path'], resize)), dtype=np.uint8)

        with ThreadPoolExecutor(8) as pool:
            crops = np.stack(list(pool.map(one, records)))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, crops)
        names_path.write_text(json.dumps(names), encoding='utf-8')
    if limit:
        return crops[:limit], records[:limit]
    return crops, records


def distorted_batch(crops, start, stop, name, severity, distortion_index, pool):
    def one(i):
        image = Image.fromarray(np.asarray(crops[i]))
        if name == 'clean':
            return np.asarray(image)
        rng = np.random.default_rng([SEED, distortion_index, severity, i])
        return np.asarray(distort(image, name, severity, rng), dtype=np.uint8)
    return np.stack(list(pool.map(one, range(start, stop))))


# ---------------------------------------------------------------- model

def load_model(run, weights, device):
    from train_clip_compositional import CompositionalCLIP
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    taxonomy = Taxonomy(load_labels())
    model = CompositionalCLIP(config['backbone'], taxonomy, config.get('tokens', 'learned'),
                              config.get('unfreeze', 4), config.get('heads', False))
    state = torch.load(weights or run / 'final.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    mean, std = (CLIP_MEAN, CLIP_STD) if config.get('normalisation') == 'CLIP' else (IMAGENET_MEAN, IMAGENET_STD)
    return model.to(device).eval(), taxonomy, config, mean, std


@torch.no_grad()
def scores(model, batch, mean, std, device):
    x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div_(255)
    x = (x - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / torch.tensor(std, device=device).view(1, 3, 1, 1)
    with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
        out = model(x)
    out = {'composition': out['composition']} if 'composition' in out else out
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def composition_log_p(outputs, taxonomy, seen_mask, gamma):
    log_p = level_scores(outputs, taxonomy)['composition']
    log_p[:, seen_mask] -= gamma  # same calibration as gzsl.calibrated
    return log_p


def metrics(log_p, records):
    targets = np.array([r['composition_id'] for r in records])
    return {k: accuracy(top_k_hits(log_p, targets, k), targets) for k in (1, 5)}


# ---------------------------------------------------------------- reporting

FIELDS = ['distortion', 'severity', 'seen_top1', 'unseen_top1', 'h_top1', 'seen_top5', 'unseen_top5', 'h_top5',
          'h_top1_per_image']


def row(name, severity, seen, unseen):
    out = {'distortion': name, 'severity': severity}
    for k in (1, 5):
        s, u = seen[k]['per_class'], unseen[k]['per_class']
        out.update({f'seen_top{k}': s, f'unseen_top{k}': u, f'h_top{k}': harmonic_mean(s, u)})
    out['h_top1_per_image'] = harmonic_mean(seen[1]['per_image'], unseen[1]['per_image'])
    return out


def pct(value):
    return f'{100 * value:.2f}'


def summary_markdown(name, run, gamma, rows, limit):
    clean = next((r for r in rows if r['distortion'] == 'clean'), None)
    lines = [f'# Robustness: {name}', '',
             f'Model run: `{run}`. Calibration gamma = {gamma}. Per-class mean accuracy (%), every image ranked '
             'against all 333 compositions; H = harmonic mean of seen and unseen accuracy.', '']
    if limit:
        lines += [f'**Smoke test only: first {limit} images per split. Not for the report.**', '']
    by_name = {}
    for r in rows:
        if r['distortion'] != 'clean':
            by_name.setdefault(r['distortion'], []).append(r)
    if clean and by_name:
        mean_h = np.mean([r['h_top1'] for rs in by_name.values() for r in rs])
        held = [r['h_top1'] for n, rs in by_name.items() if n in HELD_OUT for r in rs]
        lines += ['## Headline', '', '| | H Top-1 | Relative to clean |', '|---|---|---|',
                  f"| Clean | {pct(clean['h_top1'])} | 100% |",
                  f'| Mean over all distortions and severities | {pct(mean_h)} | {mean_h / clean["h_top1"]:.0%} |']
        if held:
            lines.append(f'| Mean over held-out distortions only | {pct(np.mean(held))} | '
                         f'{np.mean(held) / clean["h_top1"]:.0%} |')
        lines.append('')
    lines += ['## By distortion (H Top-1 per severity)', '',
              '| Distortion | Simulates | Held out from strong aug. | Sev. 1 | Sev. 2 | Sev. 3 | Mean | Drop vs clean |',
              '|---|---|---|---|---|---|---|---|']
    if clean:
        lines.append(f"| clean | original test image | | | | | {pct(clean['h_top1'])} | |")
    for n, rs in by_name.items():
        hs = {r['severity']: r['h_top1'] for r in rs}
        mean = float(np.mean(list(hs.values())))
        drop = f'{100 * (clean["h_top1"] - mean):.2f} pts' if clean else ''
        lines.append(f'| {n} | {DISTORTIONS[n][0]} | {"yes" if n in HELD_OUT else ""} | '
                     + ' | '.join(pct(hs[s]) if s in hs else '' for s in (1, 2, 3)) + f' | {pct(mean)} | {drop} |')
    lines += ['', '## All conditions', '', '| Distortion | Sev. | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | '
              'Unseen Top-5 | H Top-5 |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['distortion']} | {r['severity'] or ''} | " + ' | '.join(
            pct(r[k]) for k in ('seen_top1', 'unseen_top1', 'h_top1', 'seen_top5', 'unseen_top5', 'h_top5')) + ' |')
    return '\n'.join(lines) + '\n'


def examples(path, index=0, split='test_seen'):
    """One test image under every distortion and severity, for the report."""
    crops, records = cached_crops(split)
    tile, label_w, pad = 160, 190, 4
    names = list(DISTORTIONS)
    sheet = Image.new('RGB', (label_w + 4 * (tile + pad), (len(names) + 1) * (tile + pad) + 24), 'white')
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(('original', 'severity 1', 'severity 2', 'severity 3')):
        draw.text((label_w + c * (tile + pad) + 4, 4), title, fill='black')
    base = Image.fromarray(np.asarray(crops[index]))
    for r, name in enumerate(names):
        y = 24 + r * (tile + pad)
        draw.text((6, y + tile // 2 - 12), name, fill='black')
        draw.text((6, y + tile // 2 + 2), DISTORTIONS[name][0][:28], fill='gray')
        for c in range(4):
            img = base if c == 0 else distort(base, name, c, np.random.default_rng([SEED, names.index(name), c, index]))
            sheet.paste(img.resize((tile, tile)), (label_w + c * (tile + pad), y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"Saved {path} ({records[index]['composition_name']})")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, help='Run folder with config.json, final.pth and calibration.json.')
    parser.add_argument('--weights', type=Path, help='Checkpoint (default: <run>/final.pth).')
    parser.add_argument('--name', help='Results name (results/robustness/<name>/).')
    parser.add_argument('--only', nargs='+', choices=['clean', *DISTORTIONS], help='Evaluate only these.')
    parser.add_argument('--no-calibration', action='store_true', help='Ignore calibration.json (gamma = 0).')
    parser.add_argument('--limit', type=int, help='Smoke test on the first N images of each split.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--examples', type=Path, help='Only save an example sheet of the distortions.')
    args = parser.parse_args()
    if args.examples:
        examples(args.examples)
        return 0
    if not (args.run and args.name):
        parser.error('--run and --name are required (or use --examples).')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, taxonomy, config, mean, std = load_model(args.run, args.weights, device)
    calibration = args.run / 'calibration.json'
    gamma = 0.0 if args.no_calibration or not calibration.is_file() else \
        json.loads(calibration.read_text(encoding='utf-8'))['gamma']
    if not args.no_calibration and not calibration.is_file():
        print('No calibration.json: reporting uncalibrated scores (gamma = 0).')
    seen_mask = np.zeros(taxonomy.size('composition'), dtype=bool)
    seen_mask[taxonomy.seen] = True
    data = {split: cached_crops(split, args.limit) for split in SPLITS}
    out = ROOT / 'results' / 'robustness' / args.name
    out.mkdir(parents=True, exist_ok=True)
    names = list(DISTORTIONS)
    rows = []
    with ThreadPoolExecutor(args.threads) as pool, (out / 'robustness.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        for name, severity in conditions(args.only):
            start = time.perf_counter()
            result = {}
            for split, (crops, records) in data.items():
                parts = []
                for b in range(0, len(records), args.batch_size):
                    batch = distorted_batch(crops, b, min(b + args.batch_size, len(records)), name, severity,
                                            names.index(name) if name in names else -1, pool)
                    parts.append(scores(model, batch, mean, std, device))
                outputs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                result[split] = metrics(composition_log_p(outputs, taxonomy, seen_mask, gamma), records)
            r = row(name, severity, result['test_seen'], result['test_unseen'])
            rows.append(r)
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
            f.flush()
            print(f"{name:15s} sev {severity}  seen {pct(r['seen_top1'])}  unseen {pct(r['unseen_top1'])}  "
                  f"H {pct(r['h_top1'])}  ({time.perf_counter() - start:.0f}s)", flush=True)
    (out / 'summary.md').write_text(summary_markdown(args.name, args.run.as_posix(), gamma, rows, args.limit),
                                    encoding='utf-8')
    (out / 'config.json').write_text(json.dumps({'run': args.run.as_posix(), 'gamma': gamma, 'seed': SEED,
                                                 'limit': args.limit, 'model_config': config,
                                                 'distortions': {k: list(v[1]) for k, v in DISTORTIONS.items()},
                                                 'held_out': sorted(HELD_OUT)}, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
