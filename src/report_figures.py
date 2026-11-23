r"""Report figures: model comparison with and without calibrated stacking, WiSE-FT alpha sweep and accuracy on each
unseen pair.

Reads the saved test predictions (predictions/, not in git) and the WiSE-FT alpha sweep (outputs/), so run it on
the machine that produced them:
    python src\report_figures.py
writes 300-dpi PNGs to results/figures/. Each model keeps one colour in every figure.
"""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels, load_split  # noqa: E402
from evaluate import PRIMARY, Taxonomy, align, level_scores, load_predictions  # noqa: E402
from calibrated_stacking import TEST_SPLITS, composition_log_probabilities, metrics_at  # noqa: E402

MODELS = {'Baseline 1': 'baseline1', 'CSP': 'csp', 'CSP + WiSE-FT': 'csp_wiseft'}  # prediction file prefixes
# Each model's own gamma, chosen on the pseudo-unseen validation split without test labels
# (calibrated_stacking_pseudo_unseen.md in its results folder) and applied once to its test predictions.
GAMMAS = {'Baseline 1': 6.081360077439692, 'CSP': 2.738, 'CSP + WiSE-FT': 5.3459}
COLOURS = {'Baseline 1': '#eb6834', 'CSP': '#1baf7a', 'CSP + WiSE-FT': '#2a78d6'}
INK, INK_SECONDARY, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SWEEP = PROJECT / 'outputs/wiseft_ViT-B-16-quickgelu_openai_pseudo_seed42/alpha_sweep.csv'
OUT = PROJECT / 'results/figures'


def style():
    plt.rcParams.update({
        'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9, 'text.color': INK,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK_SECONDARY, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_SECONDARY,
        'ytick.labelcolor': INK_SECONDARY, 'legend.frameon': False, 'savefig.dpi': 300,
        'figure.facecolor': 'white', 'savefig.bbox': 'tight'})


def title(axis, text):
    axis.set_title(text, loc='left', fontsize=10, color=INK, fontweight='semibold', pad=24)


def model_comparison(taxonomy):
    """Test accuracy of every model without calibration (left) and at its own pseudo-unseen gamma (right)."""
    seen_mask = np.zeros(taxonomy.size('composition'))
    seen_mask[taxonomy.seen] = 1
    points = {}
    for model, prefix in MODELS.items():
        data = {split: composition_log_probabilities(PROJECT / f'predictions/{prefix}_{split}.npz', split, taxonomy)
                for split in TEST_SPLITS}
        points[model] = [metrics_at(gamma, data, seen_mask) for gamma in (0.0, GAMMAS[model])]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    groups, width, x = ['Seen', 'Unseen', 'Harmonic\nmean'], 0.26, np.arange(3)
    panels = ('Without calibration (γ = 0)', 'With calibrated stacking (γ chosen on pseudo-unseen)')
    for calibrated, (axis, text) in enumerate(zip(axes, panels)):
        for i, (model, results) in enumerate(points.items()):
            point = results[calibrated]
            values = 100 * np.array([point['test_seen']['top1'][PRIMARY], point['test_unseen']['top1'][PRIMARY],
                                     point['harmonic_mean']['top1'][PRIMARY]])
            bars = axis.bar(x + (i - 1) * width, values, width * 0.9, color=COLOURS[model], label=model)
            for bar, value in zip(bars, values):  # every value printed: a static figure has no tooltip
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f'{value:.1f}', ha='center',
                          va='bottom', fontsize=6.5, color=INK_SECONDARY)
        axis.set_xticks(x, groups)
        axis.tick_params(axis='x', length=0)
        axis.grid(axis='x', visible=False)
        axis.set_title(text, loc='left', fontsize=9, color=INK, fontweight='semibold')
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel('Per-class Top-1 on the test sets (%)')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.08), ncols=3, borderaxespad=0.3, handlelength=1.2)
    figure.tight_layout()
    return figure


def alpha_sweep(path):
    with path.open(encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    alpha = [float(row['alpha']) for row in rows]
    series = {'Seen': ([100 * float(row['seen_top1']) for row in rows], INK_SECONDARY, 's', 1.5),
              'Unseen': ([100 * float(row['unseen_top1']) for row in rows], MUTED, 'D', 1.5),
              'Harmonic mean': ([100 * float(row['h_top1']) for row in rows], COLOURS['CSP + WiSE-FT'], 'o', 2.0)}
    figure, axis = plt.subplots(figsize=(6.5, 3.3))
    axis.axvline(0.5, color=AXIS, linewidth=1, zorder=1)
    for name, (values, colour, marker, width) in series.items():
        axis.plot(alpha, values, color=colour, linewidth=width, marker=marker, markersize=4, label=name, zorder=3)
        axis.text(1.03, values[-1], name, va='center', fontsize=8.5, color=INK_SECONDARY)
    harmonic = series['Harmonic mean'][0]
    axis.annotate(f'α = 0.5 chosen\nH {harmonic[5]:.1f}%', (0.5, harmonic[5]), xytext=(0.53, 34), fontsize=8,
                  color=INK, arrowprops={'arrowstyle': '-', 'color': AXIS, 'linewidth': 0.8})
    axis.annotate(f'plain CSP (α = 0)\nH {harmonic[0]:.1f}%', (0, harmonic[0]), xytext=(0.04, 30), fontsize=8,
                  color=INK_SECONDARY, arrowprops={'arrowstyle': '-', 'color': AXIS, 'linewidth': 0.8})
    axis.set_xlim(-0.02, 1.0)
    axis.set_ylim(0, 75)
    axis.set_xticks(np.linspace(0, 1, 11))
    axis.set_xlabel('α: share of fine-tuned weights in the WiSE-FT blend')
    axis.set_ylabel('Per-class Top-1 (%)')
    title(axis, 'Choosing α on the pseudo-unseen validation split (no test data, γ = 0)')
    axis.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3, borderaxespad=0.3, handlelength=1.6)
    return figure


def unseen_pairs(taxonomy):
    records = load_split('test_unseen')
    targets = np.array([record['composition_id'] for record in records])
    predicted = {model: level_scores(align(load_predictions(PROJECT / f'predictions/{prefix}_test_unseen.npz'),
                                           records), taxonomy)['composition'].argmax(1)
                 for model, prefix in MODELS.items()}
    pairs = sorted(set(targets.tolist()), key=lambda pair: (-(targets == pair).sum(), pair))
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    height, y = 0.26, np.arange(len(pairs))[::-1]
    accuracy = {model: np.array([100 * (prediction[targets == pair] == pair).mean() for pair in pairs])
                for model, prediction in predicted.items()}
    all_zero = np.all([values == 0 for values in accuracy.values()], axis=0)
    for i, (model, values) in enumerate(accuracy.items()):
        positions = y - (i - 1) * height
        axis.barh(positions, values, height * 0.9, color=COLOURS[model], label=model)
        for position, value, zero_group in zip(positions, values, all_zero):
            if value == 0 and not zero_group:  # a zero-length bar is invisible; mark it as zero, not missing
                axis.text(0.4, position, '0', va='center', fontsize=7, color=MUTED)
    for position in y[all_zero]:
        axis.text(0.4, position, '0% for all three models', va='center', fontsize=7.5, color=MUTED)
    axis.set_yticks(y, [f'{taxonomy.names["composition"][pair]} ({(targets == pair).sum()})' for pair in pairs])
    axis.tick_params(axis='y', length=0)
    axis.grid(axis='y', visible=False)
    axis.set_xlim(0, 70)
    axis.set_xlabel('Top-1 accuracy on the pair\'s test images (%)  ·  (n) = number of test images')
    title(axis, 'Accuracy on each of the 10 unseen pairs (test, no calibration)')
    axis.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3, borderaxespad=0.3, handlelength=1.2)
    return figure


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    taxonomy = Taxonomy(load_labels())
    for name, figure in (('model_comparison', model_comparison(taxonomy)), ('wiseft_alpha_sweep', alpha_sweep(SWEEP)),
                         ('unseen_pair_accuracy', unseen_pairs(taxonomy))):
        figure.savefig(OUT / f'{name}.png')
        plt.close(figure)
        print(f'Saved {OUT / name}.png')


if __name__ == '__main__':
    main()
