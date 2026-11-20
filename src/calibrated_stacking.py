r"""Calibrated stacking (Chao et al., 2016) for the generalised zero-shot test.

Seen compositions dominate the ranking because only they were trained on. Calibrated stacking subtracts a
constant gamma from every seen composition's log-probability before ranking all 333. Sweeping gamma traces the
trade-off between seen and unseen accuracy; this script reports the uncalibrated result (gamma = 0), the gamma
with the best per-class Top-1 harmonic mean, and the area under the seen-unseen accuracy curve (AUC).

It reads the same prediction files as evaluate.py, so it applies to CSP and to the baselines alike:
    python src\calibrated_stacking.py --name csp_vit_b16 --seen predictions\csp_test_seen.npz
        --unseen predictions\csp_test_unseen.npz

The best gamma is chosen with test labels, as in the compositional zero-shot literature's best-HM and AUC
protocol, so report it as such beside the gamma = 0 result. --gamma evaluates one fixed value instead, and
--write-predictions saves calibrated prediction files for evaluate.py (e.g. confusion matrices at that gamma).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels  # noqa: E402
from evaluate import (AVERAGES, PRIMARY, TOP_K, Taxonomy, accuracy, align, harmonic_mean, level_scores,  # noqa: E402
                      load_predictions, percent, save_predictions, split_records)

SPLITS = ('test_seen', 'test_unseen')


def composition_log_probabilities(path, split, taxonomy):
    """Log-probabilities over the 333 compositions, targets and filenames for one split, in split order."""
    records = split_records(split)
    scores = level_scores(align(load_predictions(path), records), taxonomy)['composition']
    return scores, np.array([record['composition_id'] for record in records]), [r['filename'] for r in records]


def metrics_at(gamma, data, seen_mask):
    """Seen and unseen Top-1/Top-5 accuracy and their harmonic mean with seen scores lowered by gamma."""
    result = {'gamma': float(gamma)}
    for split, (scores, targets, _) in data.items():
        top = np.argsort(-(scores - gamma * seen_mask), axis=1, kind='stable')[:, :max(TOP_K)]
        result[split] = {f'top{k}': accuracy((top[:, :k] == targets[:, None]).any(1), targets) for k in TOP_K}
    result['harmonic_mean'] = {f'top{k}': {average: harmonic_mean(result['test_seen'][f'top{k}'][average],
                                                                  result['test_unseen'][f'top{k}'][average])
                                           for average in AVERAGES} for k in TOP_K}
    return result


def gamma_grid(data, seen_mask, steps):
    """Gammas from where every Top-1 prediction is seen to where every one is unseen, plus 0."""
    scores = np.concatenate([scores for scores, _, _ in data.values()])
    gaps = scores[:, seen_mask == 1].max(1) - scores[:, seen_mask == 0].max(1)
    margin = 0.05 * (gaps.max() - gaps.min())
    return np.unique(np.append(np.linspace(gaps.min() - margin, gaps.max() + margin, steps), 0.0))


def area_under_curve(curve, average):
    """Area under seen Top-1 accuracy as a function of unseen Top-1 accuracy."""
    unseen = np.array([point['test_unseen']['top1'][average] for point in curve])
    seen = np.array([point['test_seen']['top1'][average] for point in curve])
    order = np.argsort(unseen, kind='stable')
    return float(np.trapezoid(seen[order], unseen[order]))


def plot(path, curve, uncalibrated, selected, name):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'matplotlib is not installed; skipped {path.name}')
        return
    value = lambda point, split: 100 * point[split]['top1'][PRIMARY]  # noqa: E731
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot([value(p, 'test_unseen') for p in curve], [value(p, 'test_seen') for p in curve], color='tab:blue')
    for point, label, colour in ((uncalibrated, 'gamma = 0', 'tab:grey'), (selected, 'selected', 'tab:red')):
        left.scatter(value(point, 'test_unseen'), value(point, 'test_seen'), color=colour, zorder=3,
                     label=f'{label} ({point["gamma"]:.2f})')
    left.set(xlabel='Unseen Top-1 per-class (%)', ylabel='Seen Top-1 per-class (%)', title='Seen vs unseen')
    left.legend()
    right.plot([p['gamma'] for p in curve], [100 * p['harmonic_mean']['top1'][PRIMARY] for p in curve],
               color='tab:blue')
    right.axvline(selected['gamma'], color='tab:red', linestyle='--')
    right.set(xlabel='gamma (subtracted from seen log-probabilities)', ylabel='Harmonic mean Top-1 per-class (%)',
              title='Harmonic mean vs gamma')
    figure.suptitle(f'{name}: calibrated stacking on the test sets')
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def summary_markdown(report):
    def row(label, point):
        cells = [point[split][f'top{k}'][PRIMARY] for k in TOP_K for split in SPLITS]
        harmonic = [point['harmonic_mean'][f'top{k}'][PRIMARY] for k in TOP_K]
        return (f'| {label} | {point["gamma"]:.3f} | {percent(cells[0])} | {percent(cells[1])} | '
                f'{percent(harmonic[0])} | {percent(cells[2])} | {percent(cells[3])} | {percent(harmonic[1])} | '
                f'{percent(point["harmonic_mean"]["top1"]["per_image"])} |')

    selected = 'best test harmonic mean' if report['selection'] == 'best' else 'fixed'
    return '\n'.join([
        f'# {report["model"]}: calibrated stacking', '',
        'gamma is subtracted from every seen composition\'s log-probability before ranking all 333 compositions. '
        'Accuracy is per-class mean (%) unless marked per-image.', '',
        '| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 (per-image) |',
        '|---|---|---|---|---|---|---|---|---|',
        row('Uncalibrated', report['uncalibrated']),
        row(f'Calibrated ({selected})', report['selected']), '',
        f'AUC of the seen-unseen Top-1 curve (x100): {percent(report["auc"]["per_class"])} (per-class), '
        f'{percent(report["auc"]["per_image"])} (per-image).', '',
        'A gamma chosen as the best test harmonic mean uses test labels (the usual compositional zero-shot '
        'protocol); report it beside the uncalibrated result.', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Model name, also the results folder name.')
    parser.add_argument('--seen', required=True, type=Path, help='Predictions for test_seen (.npz).')
    parser.add_argument('--unseen', required=True, type=Path, help='Predictions for test_unseen (.npz).')
    parser.add_argument('--gamma', type=float, help='evaluate this gamma instead of the best test harmonic mean')
    parser.add_argument('--steps', type=int, default=400, help='gamma values in the sweep')
    parser.add_argument('--out', type=Path, default=PROJECT / 'results', help='Parent results folder.')
    parser.add_argument('--write-predictions', type=Path,
                        help='folder for calibrated prediction files at the selected gamma, for evaluate.py')
    args = parser.parse_args()

    taxonomy = Taxonomy(load_labels())
    seen_mask = np.zeros(taxonomy.size('composition'))
    seen_mask[taxonomy.seen] = 1
    data = {split: composition_log_probabilities(path, split, taxonomy)
            for split, path in zip(SPLITS, (args.seen, args.unseen))}

    curve = [metrics_at(gamma, data, seen_mask) for gamma in gamma_grid(data, seen_mask, args.steps)]
    uncalibrated = next(point for point in curve if point['gamma'] == 0.0)
    if args.gamma is None:
        selected = max(curve, key=lambda point: (point['harmonic_mean']['top1'][PRIMARY],
                                                 point['harmonic_mean']['top1']['per_image']))
    else:
        selected = metrics_at(args.gamma, data, seen_mask)
    report = {'model': args.name, 'score': 'composition log-probability over all 333 compositions',
              'selection': 'best' if args.gamma is None else 'fixed', 'uncalibrated': uncalibrated,
              'selected': selected, 'auc': {average: area_under_curve(curve, average) for average in AVERAGES},
              'curve': curve}

    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'calibrated_stacking.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = summary_markdown(report)
    (folder / 'calibrated_stacking.md').write_text(summary, encoding='utf-8')
    plot(folder / 'calibrated_stacking.png', curve, uncalibrated, selected, args.name)
    print(summary)
    print(f'Wrote {folder}')

    if args.write_predictions:
        for split, (scores, _, filenames) in data.items():
            path = args.write_predictions / f'{args.name}_calibrated_{split}.npz'
            save_predictions(path, filenames, composition=scores - selected['gamma'] * seen_mask)
            print(f'{split}: calibrated predictions saved to {path}')


if __name__ == '__main__':
    main()
