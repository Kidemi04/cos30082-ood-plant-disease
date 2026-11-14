r"""Shared generalised zero-shot tools: pseudo-unseen protocol, calibration and training helpers.

Protocol for every model trained on this branch
  select stage: train on the shared validation-split train records minus the pseudo-unseen compositions;
                choose the epoch by seen validation accuracy (the same rule as the team baselines);
                then save predictions for the pseudo-unseen evaluation set.
  calibrate:    on those predictions, choose the calibration constant gamma that maximises the harmonic mean
                of seen and pseudo-unseen accuracy. gamma is subtracted from the log-probability of every
                composition treated as seen. Test data is never used to choose gamma.
  final stage:  retrain on all 8,000 training images, predict the test splits, apply the chosen gamma and run
                evaluate.py on both the raw and the calibrated predictions.

Commands
  python src\gzsl.py merge --crop crop.npz --disease disease.npz --out both.npz
  python src\gzsl.py calibrate --pseudo P.npz [--seen S.npz --unseen U.npz] --out folder
"""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels, load_split, load_validation_split  # noqa: E402
from evaluate import (Taxonomy, accuracy, align, harmonic_mean, level_scores, load_predictions,  # noqa: E402
                      save_predictions, top_k_hits)
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, DPDDataset, eval_transform, train_transform  # noqa: E402

PSEUDO_MANIFEST = PROJECT / 'dataset/metadata/pseudo_unseen_split.csv'
GAMMAS = np.round(np.arange(0.0, 30.0001, 0.1), 1)


def pseudo_unseen_ids():
    if not PSEUDO_MANIFEST.is_file():
        raise FileNotFoundError('Run scripts/make_pseudo_unseen_split.py first.')
    with PSEUDO_MANIFEST.open(encoding='utf-8', newline='') as source:
        return {int(row['composition_id']) for row in csv.DictReader(source)}


def pseudo_protocol():
    """Records for model selection: train, seen validation, and pseudo-unseen evaluation."""
    pseudo = pseudo_unseen_ids()
    train, val = load_validation_split()
    official_train = load_split('train')
    return {'train': [r for r in train if r['composition_id'] not in pseudo],
            'val_seen': [r for r in val if r['composition_id'] not in pseudo],
            'pseudo_unseen': [r for r in official_train if r['composition_id'] in pseudo]}


def seen_columns(taxonomy, stage):
    """Boolean mask over the 333 compositions: True for the compositions the model was trained on."""
    seen = np.zeros(taxonomy.size('composition'), dtype=bool)
    seen[taxonomy.seen] = True
    if stage == 'select':
        seen[list(pseudo_unseen_ids())] = False
    return seen


def composition_log_probs(predictions, records, taxonomy):
    return level_scores(align(predictions, records), taxonomy)['composition']


def calibrated(log_p, seen, gamma):
    out = log_p.copy()
    out[:, seen] -= gamma
    return out


def split_accuracy(log_p, records, k=1):
    targets = np.array([r['composition_id'] for r in records])
    return accuracy(top_k_hits(log_p, targets, k), targets)['per_class']


def sweep(seen_log_p, seen_records, unseen_log_p, unseen_records, seen):
    """Seen and unseen per-class Top-1 for every gamma; the curve also gives the AUSUC."""
    rows = []
    for gamma in GAMMAS:
        s = split_accuracy(calibrated(seen_log_p, seen, gamma), seen_records)
        u = split_accuracy(calibrated(unseen_log_p, seen, gamma), unseen_records)
        rows.append({'gamma': float(gamma), 'seen': s, 'unseen': u, 'harmonic': harmonic_mean(s, u)})
    best = max(rows, key=lambda row: (round(row['harmonic'], 6), -row['gamma']))
    points = sorted({(r['unseen'], r['seen']) for r in rows} | {(0.0, rows[0]['seen'])})
    ausuc = float(np.trapezoid([p[1] for p in points], [p[0] for p in points])) if len(points) > 1 else 0.0
    return rows, best, ausuc


def split_pseudo(predictions, protocol):
    """Split one prediction file over val_seen + pseudo_unseen into the two parts."""
    index = {str(name): i for i, name in enumerate(predictions['filenames'])}
    parts = {}
    for key in ('val_seen', 'pseudo_unseen'):
        rows = [index[r['filename']] for r in protocol[key]]
        parts[key] = {name: value[rows] for name, value in predictions.items()}
    return parts


def calibrate(pseudo_path, out, seen_path=None, unseen_path=None):
    taxonomy = Taxonomy(load_labels())
    protocol = pseudo_protocol()
    parts = split_pseudo(load_predictions(pseudo_path), protocol)
    seen = seen_columns(taxonomy, 'select')
    seen_log_p = composition_log_probs(parts['val_seen'], protocol['val_seen'], taxonomy)
    unseen_log_p = composition_log_probs(parts['pseudo_unseen'], protocol['pseudo_unseen'], taxonomy)
    rows, best, ausuc = sweep(seen_log_p, protocol['val_seen'], unseen_log_p, protocol['pseudo_unseen'], seen)
    out.mkdir(parents=True, exist_ok=True)
    report = {'gamma': best['gamma'], 'pseudo_at_gamma': best, 'pseudo_at_zero': rows[0], 'pseudo_ausuc': ausuc,
              'source': str(pseudo_path)}
    (out / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (out / 'calibration_curve.csv').open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=['gamma', 'seen', 'unseen', 'harmonic'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    official_seen = seen_columns(taxonomy, 'final')
    for split, path in (('test_seen', seen_path), ('test_unseen', unseen_path)):
        if path:
            log_p = composition_log_probs(load_predictions(path), load_split(split), taxonomy)
            names = [r['filename'] for r in load_split(split)]
            save_predictions(out / f'{split}_calibrated.npz', names,
                             composition=calibrated(log_p, official_seen, best['gamma']))
    print(f"gamma={best['gamma']}: pseudo seen {100 * best['seen']:.2f}, unseen {100 * best['unseen']:.2f}, "
          f"H {100 * best['harmonic']:.2f} (gamma=0: H {100 * rows[0]['harmonic']:.2f}); AUSUC {100 * ausuc:.2f}")
    return report


# ---------------------------------------------------------------- training helpers

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_datasets(stage, mean=IMAGENET_MEAN, std=IMAGENET_STD, image_size=224, augment='standard'):
    """select: train / val_seen / pseudo (val_seen + pseudo_unseen). final: train (8,000) / test splits.

    augment chooses the training augmentation strength ('standard' or 'strong'); evaluation is unchanged.
    """
    train_tf = train_transform(image_size, mean, std, strength=augment)
    eval_tf = eval_transform(image_size, mean, std)
    if stage == 'select':
        p = pseudo_protocol()
        return {'train': DPDDataset(p['train'], train_tf, image_size),
                'val_seen': DPDDataset(p['val_seen'], eval_tf, image_size),
                'pseudo': DPDDataset(p['val_seen'] + p['pseudo_unseen'], eval_tf, image_size)}
    return {'train': DPDDataset(load_split('train'), train_tf, image_size),
            'test_seen': DPDDataset(load_split('test_seen'), eval_tf, image_size),
            'test_unseen': DPDDataset(load_split('test_unseen'), eval_tf, image_size)}


def eval_loader(dataset, batch_size, workers):
    return DataLoader(dataset, batch_size=batch_size * 2, num_workers=workers, pin_memory=True,
                      persistent_workers=False)


@torch.no_grad()
def predict(forward, loader, device):
    """forward(images) -> dict of logits tensors (crop / disease / composition). Returns numpy arrays."""
    names, outputs = [], {}
    for batch in loader:
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            result = forward(batch['image'].to(device, non_blocking=True))
        names.extend(batch['filename'])
        for key, value in result.items():
            outputs.setdefault(key, []).append(value.float().cpu())
    return {'filenames': np.asarray(names), **{k: torch.cat(v).numpy() for k, v in outputs.items()}}


def val_score(predictions, records, taxonomy, level):
    """Per-class Top-1 on seen validation; composition is ranked against all 333 like the official metric."""
    if level == 'composition':
        scores = level_scores(predictions, taxonomy)['composition']
    else:
        scores = predictions[level]
    targets = np.array([r[level + '_id'] for r in records])
    return accuracy(top_k_hits(scores, targets, 1), targets)['per_class']


def merge(crop_path, disease_path, out):
    crop, disease = load_predictions(crop_path), load_predictions(disease_path)
    order = {str(n): i for i, n in enumerate(disease['filenames'])}
    rows = [order[str(n)] for n in crop['filenames']]
    save_predictions(out, crop['filenames'], crop=crop['crop'], disease=disease['disease'][rows])
    print(f'Merged {len(rows)} rows into {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    m = commands.add_parser('merge', help='Combine Baseline 1 crop and disease predictions into one file.')
    m.add_argument('--crop', type=Path, required=True)
    m.add_argument('--disease', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    c = commands.add_parser('calibrate', help='Choose gamma on pseudo-unseen predictions; optionally apply to test.')
    c.add_argument('--pseudo', type=Path, required=True)
    c.add_argument('--seen', type=Path)
    c.add_argument('--unseen', type=Path)
    c.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'merge':
        merge(args.crop, args.disease, args.out)
    else:
        calibrate(args.pseudo, args.out, args.seen, args.unseen)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
