r"""Calibrated stacking (Chao et al., 2016) for the generalised zero-shot evaluation.

Seen compositions dominate the ranking because only they were trained on. Calibrated stacking subtracts a
constant gamma from every trained composition's log-probability before ranking all 333. Sweeping gamma traces the
trade-off between seen and unseen accuracy; this script reports the uncalibrated result (gamma = 0), a calibrated
result and the area under the seen-unseen accuracy curve (AUC).

It reads the same prediction files as evaluate.py, so it applies to CSP and to the baselines alike.

Choose gamma without test labels, on the pseudo-unseen validation split (pseudo_unseen.py), from a model trained
with those pairs held out:
    python src\calibrated_stacking.py --pseudo-unseen --name csp_vit_b16
        --seen predictions\csp_pseudo_pseudo_val_seen.npz --unseen predictions\csp_pseudo_pseudo_val_unseen.npz
then apply it to the final model's test predictions:
    python src\calibrated_stacking.py --name csp_vit_b16 --gamma <gamma>
        --seen predictions\csp_test_seen.npz --unseen predictions\csp_test_unseen.npz

Without --gamma on the test splits, gamma is chosen with test labels (the best-HM protocol of the compositional
zero-shot literature); report that only as a best case. --write-predictions saves calibrated prediction files for
evaluate.py, e.g. for confusion matrices at the chosen gamma.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels  # noqa: E402
from evaluate import (AVERAGES, PRIMARY, TOP_K, Taxonomy, accuracy, align, harmonic_mean, level_scores,  # noqa: E402
                      load_predictions, percent, save_predictions)
from pseudo_unseen import SPLITS as PSEUDO_SPLITS, records_for, trained_pairs  # noqa: E402

TEST_SPLITS = ('test_seen', 'test_unseen')


def composition_log_probabilities(path, split, taxonomy):
    """Log-probabilities over the 333 compositions, targets and filenames for one split, in split order."""
    records = records_for(split)
    scores = level_scores(align(load_predictions(path), records), taxonomy)['composition']
    return scores, np.array([record['composition_id'] for record in records]), [r['filename'] for r in records]


def metrics_at(gamma, data, seen_mask):
    """Accuracy on the seen and unseen split (data's first and second key) with trained scores lowered by gamma."""
    seen_split, unseen_split = data
    result = {'gamma': float(gamma)}
    for split, (scores, targets, _) in data.items():
        top = np.argsort(-(scores - gamma * seen_mask), axis=1, kind='stable')[:, :max(TOP_K)]
        result[split] = {f'top{k}': accuracy((top[:, :k] == targets[:, None]).any(1), targets) for k in TOP_K}
    result['harmonic_mean'] = {f'top{k}': {average: harmonic_mean(result[seen_split][f'top{k}'][average],
                                                                  result[unseen_split][f'top{k}'][average])
                                           for average in AVERAGES} for k in TOP_K}
    return result


def gamma_grid(data, seen_mask, steps):
    """Gammas from where every Top-1 prediction is a trained pair to where none is, plus 0."""
    scores = np.concatenate([scores for scores, _, _ in data.values()])
    gaps = scores[:, seen_mask == 1].max(1) - scores[:, seen_mask == 0].max(1)
    margin = 0.05 * (gaps.max() - gaps.min())
    return np.unique(np.append(np.linspace(gaps.min() - margin, gaps.max() + margin, steps), 0.0))


def sweep(data, seen_mask, steps=400):
    """(curve, uncalibrated point, point with the best per-class Top-1 harmonic mean)."""
    curve = [metrics_at(gamma, data, seen_mask) for gamma in gamma_grid(data, seen_mask, steps)]
    uncalibrated = next(point for point in curve if point['gamma'] == 0.0)
    best = max(curve, key=lambda point: (point['harmonic_mean']['top1'][PRIMARY],
                                         point['harmonic_mean']['top1']['per_image']))
    return curve, uncalibrated, best


def area_under_curve(curve, splits, average):
    """Area under seen Top-1 accuracy as a function of unseen Top-1 accuracy."""
    seen = np.array([point[splits[0]]['top1'][average] for point in curve])
    unseen = np.array([point[splits[1]]['top1'][average] for point in curve])
    order = np.argsort(unseen, kind='stable')
    return float(np.trapezoid(seen[order], unseen[order]))


def plot(path, report):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'matplotlib is not installed; skipped {path.name}')
        return
    seen_split, unseen_split = report['splits']
    curve, uncalibrated, selected = report['curve'], report['uncalibrated'], report['selected']
    value = lambda point, split: 100 * point[split]['top1'][PRIMARY]  # noqa: E731
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot([value(p, unseen_split) for p in curve], [value(p, seen_split) for p in curve], color='tab:blue')
    for point, label, colour in ((uncalibrated, 'gamma = 0', 'tab:grey'), (selected, 'selected', 'tab:red')):
        left.scatter(value(point, unseen_split), value(point, seen_split), color=colour, zorder=3,
                     label=f'{label} ({point["gamma"]:.2f})')
    left.set(xlabel=f'{unseen_split} Top-1 per-class (%)', ylabel=f'{seen_split} Top-1 per-class (%)',
             title='Seen vs unseen')
    left.legend()
    right.plot([p['gamma'] for p in curve], [100 * p['harmonic_mean']['top1'][PRIMARY] for p in curve],
               color='tab:blue')
    right.axvline(selected['gamma'], color='tab:red', linestyle='--')
    right.set(xlabel='gamma (subtracted from trained pairs\' log-probabilities)',
              ylabel='Harmonic mean Top-1 per-class (%)', title='Harmonic mean vs gamma')
    figure.suptitle(f'{report["model"]}: calibrated stacking on {seen_split} + {unseen_split}')
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def summary_markdown(report):
    seen_split, unseen_split = report['splits']

    def row(label, point):
        cells = [point[split][f'top{k}'][PRIMARY] for k in TOP_K for split in report['splits']]
        harmonic = [point['harmonic_mean'][f'top{k}'][PRIMARY] for k in TOP_K]
        return (f'| {label} | {point["gamma"]:.3f} | {percent(cells[0])} | {percent(cells[1])} | '
                f'{percent(harmonic[0])} | {percent(cells[2])} | {percent(cells[3])} | {percent(harmonic[1])} | '
                f'{percent(point["harmonic_mean"]["top1"]["per_image"])} |')

    lines = [f'# {report["model"]}: calibrated stacking on {seen_split} + {unseen_split}', '',
             f'gamma is subtracted from the log-probability of each of the {report["trained_pairs"]} trained '
             'compositions before ranking all 333. Accuracy is per-class mean (%) unless marked per-image.', '',
             '| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | '
             'H Top-1 (per-image) |', '|---|---|---|---|---|---|---|---|---|',
             row('Uncalibrated', report['uncalibrated']), row(f'Calibrated ({report["selection"]})', report['selected']),
             '', f'AUC of the seen-unseen Top-1 curve (x100): {percent(report["auc"]["per_class"])} (per-class), '
             f'{percent(report["auc"]["per_image"])} (per-image).', '']
    if report['splits'] == list(TEST_SPLITS) and report['selection'].startswith('best'):
        lines += ['This gamma was chosen with test labels; report it only as a best case, beside the uncalibrated '
                  'result.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Model name, also the results folder name.')
    parser.add_argument('--seen', required=True, type=Path, help='Predictions for the seen split (.npz).')
    parser.add_argument('--unseen', required=True, type=Path, help='Predictions for the unseen split (.npz).')
    parser.add_argument('--pseudo-unseen', action='store_true',
                        help='score the pseudo-unseen validation splits of a model trained without those pairs')
    parser.add_argument('--gamma', type=float, help='evaluate this gamma instead of the best harmonic mean')
    parser.add_argument('--steps', type=int, default=400, help='gamma values in the sweep')
    parser.add_argument('--out', type=Path, default=PROJECT / 'results', help='Parent results folder.')
    parser.add_argument('--write-predictions', type=Path,
                        help='folder for calibrated prediction files at the selected gamma, for evaluate.py')
    args = parser.parse_args()

    taxonomy = Taxonomy(load_labels())
    splits = PSEUDO_SPLITS if args.pseudo_unseen else TEST_SPLITS
    trained = trained_pairs(taxonomy) if args.pseudo_unseen else taxonomy.seen
    seen_mask = np.zeros(taxonomy.size('composition'))
    seen_mask[trained] = 1
    data = {split: composition_log_probabilities(path, split, taxonomy)
            for split, path in zip(splits, (args.seen, args.unseen))}

    curve, uncalibrated, best = sweep(data, seen_mask, args.steps)
    selected = best if args.gamma is None else metrics_at(args.gamma, data, seen_mask)
    selection = f'best {"pseudo-unseen validation" if args.pseudo_unseen else "test"} harmonic mean' \
        if args.gamma is None else 'given gamma'
    report = {'model': args.name, 'score': 'composition log-probability over all 333 compositions',
              'splits': list(splits), 'trained_pairs': len(trained), 'selection': selection,
              'uncalibrated': uncalibrated, 'selected': selected,
              'auc': {average: area_under_curve(curve, splits, average) for average in AVERAGES}, 'curve': curve}

    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    stem = 'calibrated_stacking' + ('_pseudo_unseen' if args.pseudo_unseen else '') + \
        ('' if args.gamma is None else '_given_gamma')
    (folder / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = summary_markdown(report)
    (folder / f'{stem}.md').write_text(summary, encoding='utf-8')
    plot(folder / f'{stem}.png', report)
    print(summary)
    print(f'Wrote {folder / stem}.*')

    if args.write_predictions:
        for split, (scores, _, filenames) in data.items():
            path = args.write_predictions / f'{args.name}_calibrated_{split}.npz'
            save_predictions(path, filenames, composition=scores - selected['gamma'] * seen_mask)
            print(f'{split}: calibrated predictions saved to {path}')


if __name__ == '__main__':
    main()
