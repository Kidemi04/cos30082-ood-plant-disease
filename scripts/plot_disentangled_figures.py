"""Generate publication-quality report figures for Disentangled ConvNeXt-Tiny (v1 -> v2).

Outputs saved to `docs/disentangled_figures/`:
  1. fig1_ablation_and_test.png       -- Select-stage ablation (zero test leakage) & held-out test comparison
  2. fig2_disentanglement_evidence.png -- 42-disease cross-crop similarity & disentanglement margin on test
  3. fig3_unseen_and_calibration.png   -- Per-class 10 unseen compositions (v1 vs v2) & pseudo-unseen gamma curve
"""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from dataset_access import load_labels, load_split  # noqa: E402
from evaluate import Taxonomy, align, level_scores, load_predictions  # noqa: E402

OUT_DIR = PROJECT / 'docs/disentangled_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean academic styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 300,
})

C_B2 = '#64748B'       # Slate gray (Baseline 2)
C_B1 = '#94A3B8'       # Light slate (Baseline 1)
C_V1 = '#D97706'       # Warm amber (v1)
C_V2 = '#0F766E'       # Deep teal (v2 winner)
C_ACCENT = '#2563EB'   # Clean blue
C_NEG = '#DC2626'      # Muted red


def make_fig1():
    """Fig 1: Select-stage ablation (left) + Held-out Test comparison on ConvNeXt-Tiny (right)."""
    # Select-stage verified numbers from runs/*/cal/calibration.json
    select_entries = [
        ('B2 Baseline\n(shared 768d)', 44.39, 29.48, C_B2),
        ('v1 + SupCon\n(balanced, p=12)', 41.10, 27.21, '#CBD5E1'),
        ('v1 Branches\n+ Cosine Head', 45.21, 31.35, '#F59E0B'),
        ('v1 Masked GRL\n(λ=0.02, Cosine)', 48.07, 33.69, C_V1),
        ('v2 Clean\n(detach-GRL, 512d)', 50.05, 34.19, '#14B8A6'),
        ('v2 + MixStyle\n(Stage 1&2)', 46.00, 29.70, '#99F6E4'),
        ('v2 + LogitAdj ★\n(τ=0.5, Winner)', 48.70, 34.13, C_V2),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={'width_ratios': [1.18, 1.0]})

    # Left panel: horizontal grouped bars for pseudo_H and AUSUC
    y = np.arange(len(select_entries))
    h = 0.36
    labels = [e[0] for e in select_entries]
    p_h = [e[1] for e in select_entries]
    ausuc = [e[2] for e in select_entries]
    colors = [e[3] for e in select_entries]

    bars_h = ax1.barh(y - h / 2, p_h, height=h, color=colors, edgecolor='white', linewidth=0.6, label='Pseudo-Unseen H (%)')
    bars_a = ax1.barh(y + h / 2, ausuc, height=h, color=colors, alpha=0.45, edgecolor='white', linewidth=0.6, hatch='///', label='Pseudo AUSUC (×100)')

    ax1.axvline(44.39, color=C_B2, linestyle=':', linewidth=1.2, alpha=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.8)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(len(select_entries) - 0.1, -0.95)
    ax1.set_xlabel('Score (%) — Strictly on Validation + Pseudo-Unseen (Zero Test Leakage)')
    ax1.set_title('(a) Model Selection & Ablation Progression')
    ax1.legend(loc='upper right', frameon=True, fontsize=8.2, ncol=2)

    for b, val in zip(bars_h, p_h):
        ax1.text(val + 0.6, b.get_y() + b.get_height() / 2, f'{val:.2f}%', va='center', fontsize=8.2, fontweight='bold')
    for b, val in zip(bars_a, ausuc):
        ax1.text(val + 0.6, b.get_y() + b.get_height() / 2, f'{val:.2f}', va='center', fontsize=8.0, color='#475569')

    # Right panel: Official Held-Out Test Comparison (same ConvNeXt-Tiny backbone)
    def load_metrics(name):
        return json.loads((PROJECT / 'results' / name / 'metrics.json').read_text(encoding='utf-8'))

    m_b1 = load_metrics('b1_calibrated')
    m_b2 = load_metrics('b2_multitask_calibrated')
    m_v1 = load_metrics('disentangled_v1_calibrated')
    m_v2 = load_metrics('disentangled_v2_calibrated')

    metrics_names = ['Calibrated\nTop-1 H', 'Calibrated\nTop-5 H', 'Unseen-Only\nTop-1']
    series = [
        ('B1 Independent', [100 * m_b1['harmonic_mean']['top1']['per_class'],
                            100 * m_b1['harmonic_mean']['top5']['per_class'],
                            100 * m_b1['splits']['test_unseen']['composition_unseen_only']['top1']['per_class']], C_B1),
        ('B2 Multitask', [100 * m_b2['harmonic_mean']['top1']['per_class'],
                          100 * m_b2['harmonic_mean']['top5']['per_class'],
                          100 * m_b2['splits']['test_unseen']['composition_unseen_only']['top1']['per_class']], C_B2),
        ('Disentangled v1', [100 * m_v1['harmonic_mean']['top1']['per_class'],
                             100 * m_v1['harmonic_mean']['top5']['per_class'],
                             100 * m_v1['splits']['test_unseen']['composition_unseen_only']['top1']['per_class']], C_V1),
        ('Disentangled v2 ★', [100 * m_v2['harmonic_mean']['top1']['per_class'],
                               100 * m_v2['harmonic_mean']['top5']['per_class'],
                               100 * m_v2['splits']['test_unseen']['composition_unseen_only']['top1']['per_class']], C_V2),
    ]

    x = np.arange(len(metrics_names))
    w = 0.19
    for idx, (label, vals, col) in enumerate(series):
        offset = (idx - 1.5) * w
        bars = ax2.bar(x + offset, vals, width=w, label=label, color=col, edgecolor='white', linewidth=0.6)
        for b, v in zip(bars, vals):
            weight = 'bold' if 'v2' in label else 'normal'
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.9, f'{v:.1f}', ha='center', va='bottom', fontsize=7.8, fontweight=weight)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=9.5)
    ax2.set_ylim(38, 90)
    ax2.set_ylabel('Per-Class Accuracy (%)')
    ax2.set_title('(b) Held-Out Test Results (Same ConvNeXt-Tiny)')
    ax2.legend(loc='upper left', frameon=True, fontsize=8.3, ncol=2)

    fig.tight_layout(pad=1.4)
    out = OUT_DIR / 'fig1_ablation_and_test.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out)


def make_fig2():
    """Fig 2: Feature disentanglement margin (left) + per-disease cross-crop gain on test (right)."""
    ev_v2 = json.loads((PROJECT / 'results/disentangled_v2_analysis/disentanglement_evidence.json').read_text(encoding='utf-8'))
    ev_v1 = json.loads((PROJECT / 'results/disentangled_v2_analysis/v1_disentanglement_evidence.json').read_text(encoding='utf-8'))

    layers = [
        ('B2 Baseline\n(shared 768d)', ev_v2['test_combined_b2_select']),
        ('v2 Backbone\n(shared 768d)', ev_v2['test_combined_v2_final_backbone']),
        ('v1 z_disease\n(256d)', ev_v1['test_combined_final_winner_zdisease']),
        ('v2 z_disease ★\n(512d)', ev_v2['test_combined_v2_final_zdisease']),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={'width_ratios': [1.0, 1.15]})

    x = np.arange(len(layers))
    w = 0.25
    same_dis = [l[1]['same_disease_diff_crop_macro'] for l in layers]
    same_crop = [l[1]['same_crop_diff_disease_macro'] for l in layers]
    margin = [l[1]['disentanglement_margin'] for l in layers]

    b1 = ax1.bar(x - w, same_dis, width=w, color=C_V2, label='Same Disease, Diff Crop ↑')
    b2 = ax1.bar(x, same_crop, width=w, color='#94A3B8', label='Same Crop, Diff Disease ↓')
    b3 = ax1.bar(x + w, margin, width=w, color=[C_NEG if m < 0 else C_ACCENT for m in margin], label='Disentanglement Margin ↑')

    ax1.axhline(0, color='#334155', linewidth=0.9)
    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            va = 'bottom' if v >= 0 else 'top'
            dy = 0.008 if v >= 0 else -0.008
            ax1.text(b.get_x() + b.get_width() / 2, v + dy, f'{v:+.3f}' if bars is b3 else f'{v:.3f}',
                     ha='center', va=va, fontsize=7.6, fontweight='bold' if bars is b3 else 'normal')

    ax1.set_xticks(x)
    ax1.set_xticklabels([l[0] for l in layers], fontsize=8.8)
    ax1.set_ylim(-0.14, 0.39)
    ax1.set_ylabel('Mean Cosine Similarity (Held-Out Test)')
    ax1.set_title('(a) Cross-Crop Feature Disentanglement (42 Diseases)')
    ax1.legend(loc='upper left', frameon=True, fontsize=8.1, ncol=1)
    ax1.text(0.98, 0.04,
             f"Subspace Orthogonality (Test):\nmean |cos(z_crop, z_dis)| = {ev_v2['test_orthogonality_mean_abs_cos_zcrop_zdisease']:.4f}",
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=8.2,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8FAFC', edgecolor='#CBD5E1'))

    # Right panel: Per-disease delta (v2 z_disease - B2) on test_combined
    rows = []
    with (PROJECT / 'results/disentangled_v2_analysis/cross_crop_similarity_per_disease.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((r['disease_name'], float(r['test_delta']), float(r['test_b2_cos']), float(r['test_ours_cos'])))
    rows.sort(key=lambda r: r[1], reverse=True)
    n_improved = sum(1 for r in rows if r[1] > 0)
    shown = rows[:10] + rows[-4:]

    y = np.arange(len(shown))
    deltas = [r[1] for r in shown]
    names = [r[0].replace('_', ' ') for r in shown]
    cols = [C_V2 if d >= 0 else C_NEG for d in deltas]

    bars = ax2.barh(y, deltas, color=cols, height=0.65, edgecolor='white', linewidth=0.5)
    ax2.axvline(0, color='#334155', linewidth=0.9)
    ax2.axhline(9.5, color='#94A3B8', linestyle=':', linewidth=1.0)
    ax2.text(0.24, 9.5, 'top 10 / bottom 4 of 42', fontsize=7.8, color='#64748B', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    for b, d in zip(bars, deltas):
        ha = 'left' if d >= 0 else 'right'
        dx = 0.008 if d >= 0 else -0.008
        ax2.text(d + dx, b.get_y() + b.get_height() / 2, f'{d:+.3f}', va='center', ha=ha, fontsize=8.0)

    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=8.4)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.12, 0.46)
    ax2.set_xlabel('Test Cross-Crop Cosine Similarity Gain (v2 z_disease − B2)')
    ax2.set_title(f'(b) Per-Disease Cross-Crop Gain ({n_improved}/{len(rows)} Diseases Improved)')

    fig.tight_layout(pad=1.4)
    out = OUT_DIR / 'fig2_disentanglement_evidence.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out)


def make_fig3():
    """Fig 3: Per-class Top-1 accuracy on 10 unseen compositions (v1 vs v2) + pseudo-unseen calibration curve."""
    taxonomy = Taxonomy(load_labels())
    unseen_records = load_split('test_unseen')

    def per_class_unseen(npz_path):
        preds = load_predictions(npz_path)
        scores = level_scores(align(preds, unseen_records), taxonomy)['composition']
        pred_ids = scores.argmax(axis=1)
        stats = {}
        for r, p in zip(unseen_records, pred_ids):
            cid = r['composition_id']
            name = taxonomy.names['composition'][cid]
            if name not in stats:
                stats[name] = [0, 0]
            stats[name][1] += 1
            if int(p) == cid:
                stats[name][0] += 1
        return {k: 100.0 * v[0] / v[1] for k, v in stats.items()}

    v1_acc = per_class_unseen(PROJECT / 'runs/final_winner/cal/test_unseen_calibrated.npz')
    v2_acc = per_class_unseen(PROJECT / 'runs/v2_final_winner/cal/test_unseen_calibrated.npz')

    classes = sorted(v2_acc.keys(), key=lambda k: v2_acc[k], reverse=True)
    y = np.arange(len(classes))
    h = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={'width_ratios': [1.22, 1.0]})

    b1 = ax1.barh(y - h / 2, [v1_acc[c] for c in classes], height=h, color=C_V1, alpha=0.85, label='Disentangled v1 (mean 50.06%)')
    b2 = ax1.barh(y + h / 2, [v2_acc[c] for c in classes], height=h, color=C_V2, label='Disentangled v2 ★ (mean 61.38%)')

    for b in b2:
        v = b.get_height()  # unused
        w_val = b.get_width()
        ax1.text(w_val + 1.2, b.get_y() + b.get_height() / 2, f'{w_val:.1f}%', va='center', fontsize=8.0, fontweight='bold')

    ax1.set_yticks(y)
    ax1.set_yticklabels([c.replace('_', ' ') for c in classes], fontsize=8.4)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 108)
    ax1.set_xlabel('Calibrated Test Top-1 Accuracy (%) — All 333 Candidates')
    ax1.set_title('(a) Per-Class Accuracy on 10 Official Unseen Compositions')
    ax1.legend(loc='lower right', frameon=True, fontsize=8.3)

    # Right panel: Pseudo-unseen calibration curve (gamma 0 -> 15)
    gammas, s_vals, u_vals, h_vals = [], [], [], []
    with (PROJECT / 'results/disentangled_v2_analysis/calibration_curve.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            g = float(r['gamma'])
            if g <= 15.0:
                gammas.append(g)
                s_vals.append(100 * float(r['seen']))
                u_vals.append(100 * float(r['unseen']))
                h_vals.append(100 * float(r['harmonic']))

    ax2.plot(gammas, s_vals, color=C_ACCENT, linewidth=1.9, label='Val Seen Top-1')
    ax2.plot(gammas, u_vals, color=C_V1, linewidth=1.9, label='Pseudo-Unseen Top-1')
    ax2.plot(gammas, h_vals, color=C_V2, linewidth=2.4, label='Harmonic Mean (H)')
    ax2.axvline(5.8, color='#334155', linestyle='--', linewidth=1.2)
    ax2.scatter([5.8], [48.70], color=C_V2, s=45, zorder=5)
    ax2.annotate('Selected γ = 5.8\n(Pseudo H = 48.70%)\nZero test leakage',
                 xy=(5.8, 48.70), xytext=(7.6, 36.0),
                 arrowprops=dict(arrowstyle='->', color='#334155', lw=1.0),
                 fontsize=8.3, bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8FAFC', edgecolor='#CBD5E1'))

    ax2.set_xlim(0, 15)
    ax2.set_ylim(0, 82)
    ax2.set_xlabel('Calibration Constant γ (Subtracted from Seen Log-Probs)')
    ax2.set_ylabel('Per-Class Top-1 Accuracy (%)')
    ax2.set_title('(b) Pseudo-Unseen Calibration Sweep (v2 Winner)')
    ax2.legend(loc='upper right', frameon=True, fontsize=8.3)

    fig.tight_layout(pad=1.4)
    out = OUT_DIR / 'fig3_unseen_and_calibration.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out)


if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
