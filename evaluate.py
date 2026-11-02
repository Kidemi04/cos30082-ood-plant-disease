"""Shared DPD-Small evaluation: Top-1/Top-5, seen and unseen accuracy, harmonic mean, confusion matrices.

Every model saves its scores per split with save_predictions(); this script scores all models identically.
Seen and unseen test images are ranked against all 333 compositions (generalised zero-shot setting).
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from dataset_access import PROJECT, load_labels, load_split, load_validation_split

TOP_K = (1, 5)
LEVELS = ('composition', 'crop', 'disease')
AVERAGES = ('per_class', 'per_image')
PRIMARY = 'per_class'
SPLITS = ('test_seen', 'test_unseen', 'val')


def save_predictions(path, filenames, crop=None, disease=None, composition=None):
    """Save one split's logits or probabilities: one row per image, one column per official label ID."""
    arrays = {'filenames': np.asarray([str(name) for name in filenames])}
    for key, value in (('crop', crop), ('disease', disease), ('composition', composition)):
        if value is not None:
            arrays[key] = np.asarray(value, dtype=np.float32)
    if 'composition' not in arrays and not {'crop', 'disease'} <= arrays.keys():
        raise ValueError('Save composition scores, or both crop and disease scores.')
    for key, value in arrays.items():
        if len(value) != len(arrays['filenames']):
            raise ValueError(f'{key} has {len(value)} rows for {len(arrays["filenames"])} filenames.')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class Taxonomy:
    """Official label spaces: 55 crops, 175 diseases and 333 compositions (323 seen, 10 unseen)."""

    def __init__(self, labels):
        self.names = {level: [labels[level][str(i)] for i in range(len(labels[level]))] for level in LEVELS}
        parts = [labels['composition_components'][str(i)] for i in range(len(self.names['composition']))]
        self.groups = {'crop': np.array([part['crop_id'] for part in parts]),
                       'disease': np.array([part['disease_id'] for part in parts])}
        self.seen = np.array(sorted(labels['seen_composition_ids']))
        self.unseen = np.array(sorted(labels['unseen_composition_ids']))

    def size(self, level):
        return len(self.names[level])


def log_probabilities(scores):
    """Log-softmax logits; rows that are already probabilities are only logged."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.min() >= 0 and np.allclose(scores.sum(1), 1, atol=1e-4):
        return np.log(np.clip(scores, 1e-12, None))
    shifted = scores - scores.max(1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(1, keepdims=True))


def marginalise(log_p, groups, size):
    """Sum composition probabilities onto their crop or disease, in log space."""
    result = np.full((len(log_p), size), -np.inf)
    for group in np.unique(groups):
        columns = log_p[:, groups == group]
        peak = columns.max(1, keepdims=True)
        result[:, group] = peak[:, 0] + np.log(np.exp(columns - peak).sum(1))
    return result


def level_scores(predictions, taxonomy):
    """Log-probabilities for every level. Without composition scores, P(composition) is P(crop) x P(disease)."""
    for level in LEVELS:
        if level in predictions and predictions[level].shape[1] != taxonomy.size(level):
            raise ValueError(f'{level} scores need {taxonomy.size(level)} columns, '
                             f'got {predictions[level].shape[1]}.')
    log_p = {level: log_probabilities(predictions[level]) for level in LEVELS if level in predictions}
    if 'composition' not in log_p:
        joint = log_p['crop'][:, taxonomy.groups['crop']] + log_p['disease'][:, taxonomy.groups['disease']]
        log_p['composition'] = log_probabilities(joint)  # renormalised over the 333 valid pairs
    for level in ('crop', 'disease'):
        if level not in log_p:
            log_p[level] = marginalise(log_p['composition'], taxonomy.groups[level], taxonomy.size(level))
    return log_p


def align(predictions, records):
    """Reorder prediction rows to match the split; every image must appear exactly once."""
    names = [str(name) for name in predictions['filenames']]
    if len(set(names)) != len(names):
        raise ValueError('Predictions contain duplicate filenames.')
    index = {name: row for row, name in enumerate(names)}
    expected = [record['filename'] for record in records]
    missing = sum(name not in index for name in expected)
    extra = len(set(names) - set(expected))
    if missing or extra:
        raise ValueError(f'Predictions do not match the split: {missing} missing, {extra} unexpected filenames.')
    order = np.array([index[name] for name in expected])
    return {key: value[order] for key, value in predictions.items()}


def top_k_hits(scores, targets, k):
    top = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    return (top == targets[:, None]).any(1)


def accuracy(hits, targets):
    """Per-class mean accuracy (every class weighs the same) and per-image accuracy."""
    per_class = [hits[targets == label].mean() for label in np.unique(targets)]
    return {'per_class': float(np.mean(per_class)), 'per_image': float(hits.mean())}


def harmonic_mean(seen, unseen):
    return 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)


def evaluate_split(predictions, records, taxonomy):
    scores = level_scores(align(predictions, records), taxonomy)
    targets = {level: np.array([record[level + '_id'] for record in records]) for level in LEVELS}
    metrics = {level: {f'top{k}': accuracy(top_k_hits(scores[level], targets[level], k), targets[level])
                       for k in TOP_K} for level in LEVELS}
    return metrics, scores, targets


def evaluate(name, predictions, records, taxonomy):
    """Score every split in predictions; both test splits also give the harmonic mean."""
    report = {'model': name, 'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'primary_average': PRIMARY, 'candidates': 'all 333 compositions', 'splits': {}}
    details = {}
    for split in SPLITS:
        if split not in predictions:
            continue
        metrics, scores, targets = evaluate_split(predictions[split], records[split], taxonomy)
        if split == 'test_unseen':
            # Supplementary closed-set result: only the 10 unseen compositions are candidates.
            closed = scores['composition'][:, taxonomy.unseen]
            closed_targets = np.searchsorted(taxonomy.unseen, targets['composition'])
            metrics['composition_unseen_only'] = {
                f'top{k}': accuracy(top_k_hits(closed, closed_targets, k), closed_targets) for k in TOP_K}
        report['splits'][split] = {'images': len(records[split]), **metrics}
        details[split] = scores, targets
    if {'test_seen', 'test_unseen'} <= report['splits'].keys():
        seen = report['splits']['test_seen']['composition']
        unseen = report['splits']['test_unseen']['composition']
        report['harmonic_mean'] = {
            f'top{k}': {average: harmonic_mean(seen[f'top{k}'][average], unseen[f'top{k}'][average])
                        for average in AVERAGES} for k in TOP_K}
    return report, details


def confusion(targets, predicted, rows, columns):
    row = {label: i for i, label in enumerate(rows)}
    column = {label: i for i, label in enumerate(columns)}
    matrix = np.zeros((len(rows), len(columns)), dtype=np.int64)
    np.add.at(matrix, ([row[t] for t in targets], [column[p] for p in predicted]), 1)
    return matrix


def top_confusions(targets, predicted, names, limit=15):
    totals = Counter(targets.tolist())
    pairs = Counter((t, p) for t, p in zip(targets.tolist(), predicted.tolist()) if t != p)
    return [{'true': names[t], 'predicted': names[p], 'count': count, 'share_of_true_class': count / totals[t]}
            for (t, p), count in pairs.most_common(limit)]


def write_matrix(path, matrix, row_names, column_names):
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target, lineterminator='\n')
        writer.writerow(['true \\ predicted', *column_names])
        writer.writerows([name, *counts] for name, counts in zip(row_names, matrix.tolist()))


def plot_matrix(path, matrix, row_names, column_names, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'matplotlib is not installed; skipped {path.name}')
        return
    totals = matrix.sum(1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    labelled = len(column_names) <= 60
    width = 9 if labelled else 12
    figure, axis = plt.subplots(figsize=(width, width * len(row_names) / len(column_names)))
    image = axis.imshow(shares, cmap='Blues', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    if labelled:
        axis.set_xticks(range(len(column_names)), column_names, rotation=90, fontsize=5)
        axis.set_yticks(range(len(row_names)), row_names, fontsize=5)
    axis.set(xlabel='Predicted', ylabel='True', title=title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label='Share of true class')
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)


def write_confusions(report, details, taxonomy, folder):
    """Seen-test confusion matrices at composition, crop and disease level, from each level's Top-1 prediction."""
    scores, targets = details['test_seen']
    report['test_seen_top_confusions'] = {}
    for level in LEVELS:
        names = taxonomy.names[level]
        rows = taxonomy.seen if level == 'composition' else np.arange(taxonomy.size(level))
        predicted = scores[level].argmax(1)
        matrix = confusion(targets[level], predicted, rows, range(taxonomy.size(level)))
        row_names = [names[label] for label in rows]
        write_matrix(folder / f'confusion_test_seen_{level}.csv', matrix, row_names, names)
        plot_matrix(folder / f'confusion_test_seen_{level}.png', matrix, row_names, names,
                    f'{report["model"]}: seen test, {level} (row-normalised)')
        report['test_seen_top_confusions'][level] = top_confusions(targets[level], predicted, names)


def percent(value):
    return f'{100 * value:.2f}'


def summary_markdown(report):
    lines = [f'# {report["model"]}', '',
             f'Candidates: {report["candidates"]}. Primary average: {PRIMARY.replace("_", "-")} mean accuracy (%).', '']
    for split, metrics in report['splits'].items():
        lines += [f'## {split} ({metrics["images"]} images)', '',
                  '| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |',
                  '|---|---|---|---|---|']
        for level, values in metrics.items():
            if level != 'images':
                lines.append(f'| {level} | ' + ' | '.join(percent(values[f'top{k}'][average])
                                                          for average in AVERAGES for k in TOP_K) + ' |')
        lines.append('')
    if 'harmonic_mean' in report:
        h = report['harmonic_mean']
        lines += ['## Harmonic mean of seen and unseen composition accuracy', '',
                  '| | Top-1 | Top-5 |', '|---|---|---|']
        lines += [f'| {average} | {percent(h["top1"][average])} | {percent(h["top5"][average])} |'
                  for average in AVERAGES]
        lines.append('')
    for level, pairs in report.get('test_seen_top_confusions', {}).items():
        lines += [f'## Most confused seen-test {level} pairs', '', '| True | Predicted | Count | Share of true class |',
                  '|---|---|---|---|']
        lines += [f'| {p["true"]} | {p["predicted"]} | {p["count"]} | {percent(p["share_of_true_class"])} |'
                  for p in pairs]
        lines.append('')
    return '\n'.join(lines)


def comparison_markdown(reports):
    """One row per model: composition accuracy (per-class) and harmonic mean, Top-1 and Top-5."""
    def cell(report, split, k):
        values = report['splits'].get(split)
        return percent(values['composition'][f'top{k}'][PRIMARY]) if values else '-'

    def harmonic(report, k, average=PRIMARY):
        return percent(report['harmonic_mean'][f'top{k}'][average]) if 'harmonic_mean' in report else '-'

    lines = ['| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | '
             'H Top-1 (per-image) |', '|---|---|---|---|---|---|---|---|']
    for report in reports:
        lines.append(f'| {report["model"]} | {cell(report, "test_seen", 1)} | {cell(report, "test_unseen", 1)} | '
                     f'{harmonic(report, 1)} | {cell(report, "test_seen", 5)} | {cell(report, "test_unseen", 5)} | '
                     f'{harmonic(report, 5)} | {harmonic(report, 1, "per_image")} |')
    return '\n'.join(lines) + '\n\nComposition accuracy (%), per-class mean, ranked against all 333 compositions.\n'


def split_records(split):
    return load_validation_split()[1] if split == 'val' else load_split(split)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Evaluate one model from its saved prediction files.')
    run.add_argument('--name', required=True, help='Model name, also the results folder name.')
    run.add_argument('--seen', type=Path, help='Predictions for test_seen (.npz).')
    run.add_argument('--unseen', type=Path, help='Predictions for test_unseen (.npz).')
    run.add_argument('--val', type=Path, help='Predictions for the shared validation split (.npz).')
    run.add_argument('--out', type=Path, default=PROJECT / 'results', help='Parent results folder.')
    compare = commands.add_parser('compare', help='Build the model comparison table from metrics.json files.')
    compare.add_argument('metrics', nargs='+', type=Path)
    compare.add_argument('--out', type=Path, default=PROJECT / 'results/comparison.md')
    args = parser.parse_args()

    if args.command == 'compare':
        reports = [json.loads(path.read_text(encoding='utf-8')) for path in args.metrics]
        table = comparison_markdown(reports)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding='utf-8')
        print(table)
        return 0

    files = {'test_seen': args.seen, 'test_unseen': args.unseen, 'val': args.val}
    files = {split: path for split, path in files.items() if path}
    if not files:
        parser.error('Give at least one of --seen, --unseen or --val.')
    taxonomy = Taxonomy(load_labels())
    predictions = {split: load_predictions(path) for split, path in files.items()}
    records = {split: split_records(split) for split in files}
    report, details = evaluate(args.name, predictions, records, taxonomy)
    report['inputs'] = {split: str(path) for split, path in files.items()}
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    if 'test_seen' in details:
        write_confusions(report, details, taxonomy, folder)
    (folder / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = summary_markdown(report)
    (folder / 'summary.md').write_text(summary + '\n', encoding='utf-8')
    print(summary)
    print(f'Wrote {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
