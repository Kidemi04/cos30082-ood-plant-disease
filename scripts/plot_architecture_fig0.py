"""Generate clean academic architecture diagram comparing Baseline 2 vs Disentangled ConvNeXt-Tiny v2."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'docs/disentangled_figures/fig0_architecture_b2_vs_v2.png'
OUT.parent.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle='', fc='#DBEAFE', ec='#2563EB', tc='#0F172A', lw=1.2, ls='-'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.14',
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center',
                fontsize=8.8, fontweight='bold', color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha='center', va='center',
                fontsize=7.6, color='#334155', zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=8.6, fontweight='bold', color=tc, zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#475569', ls='-', lw=1.2, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_length=5,head_width=3.2',
        connectionstyle=f'arc3,rad={rad}',
        color=color, linewidth=lw, linestyle=ls, zorder=2
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(11.8, 6.2), dpi=300)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 6.6)
    ax.axis('off')

    # Colors (aligned with Nigel's clean palette)
    C_IO_FC, C_IO_EC = '#F8FAFC', '#94A3B8'
    C_TR_FC, C_TR_EC = '#DBEAFE', '#2563EB'
    C_V2_FC, C_V2_EC = '#CCFBF1', '#0F766E'
    C_LOSS_FC, C_LOSS_EC = '#FEF3C7', '#D97706'

    # ------------------------------------------------------------------
    # Panel (a): Baseline 2
    # ------------------------------------------------------------------
    ax.text(0.2, 6.28, '(a) Baseline 2: Shared-Backbone Multitask CNN', fontsize=10.8, fontweight='bold', color='#0F172A')
    ax.text(0.2, 5.98, 'Trained: 1 shared ConvNeXt-Tiny + 2 linear heads (28.0M params)  ·  Test H: 51.30% cal (5.43% raw)  ·  Disentanglement Margin: −0.074',
            fontsize=8.3, color='#475569')

    draw_box(ax, 0.25, 4.75, 1.35, 0.90, 'Leaf Image', '224 × 224', fc=C_IO_FC, ec=C_IO_EC)
    draw_box(ax, 2.05, 4.75, 2.25, 0.90, 'ConvNeXt-Tiny', 'Shared 768-d feature', fc=C_TR_FC, ec=C_TR_EC)
    draw_box(ax, 4.95, 5.26, 2.05, 0.56, 'Crop Head (Linear)', '768 → 55 crops', fc=C_TR_FC, ec=C_TR_EC)
    draw_box(ax, 4.95, 4.54, 2.05, 0.56, 'Disease Head (Linear)', '768 → 175 diseases', fc=C_TR_FC, ec=C_TR_EC)
    draw_box(ax, 7.60, 4.75, 4.10, 0.90, 'Pair Score + Calibrated Stacking',
             'log P(crop) + log P(disease) − γ · [seen]\nShared 768-d binds disease prediction to crop', fc=C_IO_FC, ec=C_IO_EC)

    arrow(ax, 1.62, 5.20, 2.02, 5.20)
    arrow(ax, 4.32, 5.32, 4.92, 5.54)
    arrow(ax, 4.32, 5.08, 4.92, 4.82)
    arrow(ax, 7.02, 5.54, 7.57, 5.34)
    arrow(ax, 7.02, 4.82, 7.57, 5.06)

    # Divider
    ax.plot([0.2, 11.75], [4.22, 4.22], color='#CBD5E1', linestyle='--', linewidth=0.9)

    # ------------------------------------------------------------------
    # Panel (b): Disentangled ConvNeXt-Tiny v2
    # ------------------------------------------------------------------
    ax.text(0.2, 3.88, '(b) Proposed: Disentangled ConvNeXt-Tiny v2 (Inference Architecture + Training-Only Objectives)',
            fontsize=10.8, fontweight='bold', color='#0F172A')
    ax.text(0.2, 3.58, 'Trained: same ConvNeXt-Tiny + 512-d residual orthogonal branches + EMA (30.1M params)  ·  Test H: 51.71% cal / 79.60% Top-5 H  ·  Margin: +0.079',
            fontsize=8.3, color='#0F766E', fontweight='bold')

    # Inference path (top)
    draw_box(ax, 0.25, 2.22, 1.25, 0.92, 'Leaf Image', '224 × 224', fc=C_IO_FC, ec=C_IO_EC)
    draw_box(ax, 1.85, 2.22, 1.95, 0.92, 'ConvNeXt-Tiny', 'Shared 768-d\n(protected from GRL)', fc=C_TR_FC, ec=C_TR_EC)

    draw_box(ax, 4.25, 2.74, 2.00, 0.62, 'Crop Projection', 'Residual 768 → 512 (z_crop)', fc=C_V2_FC, ec=C_V2_EC)
    draw_box(ax, 4.25, 1.94, 2.00, 0.62, 'Disease Projection', 'Residual 768 → 512 (z_dis)', fc=C_V2_FC, ec=C_V2_EC)

    draw_box(ax, 6.65, 2.74, 1.80, 0.62, 'Crop Head', 'Linear 512 → 55', fc=C_V2_FC, ec=C_V2_EC)
    draw_box(ax, 6.65, 1.94, 1.80, 0.62, 'Disease Head', 'Linear 512 → 175', fc=C_V2_FC, ec=C_V2_EC)

    draw_box(ax, 8.85, 2.22, 2.85, 0.92, 'Calibrated Composition',
             'log P(crop) + log P(disease)\n− γ·[seen]  (γ = 5.8, EMA)', fc=C_IO_FC, ec=C_IO_EC)

    arrow(ax, 1.52, 2.68, 1.82, 2.68)
    arrow(ax, 3.82, 2.80, 4.22, 3.05)
    arrow(ax, 3.82, 2.55, 4.22, 2.25)
    arrow(ax, 6.27, 3.05, 6.62, 3.05)
    arrow(ax, 6.27, 2.25, 6.62, 2.25)
    arrow(ax, 8.47, 3.05, 8.82, 2.82)
    arrow(ax, 8.47, 2.25, 8.82, 2.52)

    # Training-only dashed zone (bottom)
    train_zone = FancyBboxPatch(
        (1.85, 0.42), 9.85, 1.24,
        boxstyle='round,pad=0.02,rounding_size=0.14',
        facecolor='#FFFBEB', edgecolor='#F59E0B', linewidth=1.0, linestyle='--', zorder=1
    )
    ax.add_patch(train_zone)
    ax.text(2.02, 1.48, 'Training-Only Objectives (Removed at Inference)',
            fontsize=8.2, fontweight='bold', color='#B45309', zorder=4)

    draw_box(ax, 2.02, 0.56, 2.65, 0.78, 'Orthogonality Loss',
             'w = 0.05 · mean cos²(z_crop, z_dis)\nTest |cos| = 0.027', fc=C_LOSS_FC, ec=C_LOSS_EC)
    draw_box(ax, 4.92, 0.56, 3.45, 0.78, 'Backbone-Protected Masked GRL',
             'disease_proj(shared.detach()) → GRL(λ=0.02) → Adv Crop\nActive ONLY on 42 multi-crop diseases (133 masked)', fc=C_LOSS_FC, ec=C_LOSS_EC)
    draw_box(ax, 8.62, 0.56, 2.90, 0.78, 'Effective-Prior Logit Adj.',
             'τ = 0.5 on disease logits (train only)\n+ Model EMA (decay = 0.995)', fc=C_LOSS_FC, ec=C_LOSS_EC)

    # Dashed links from inference boxes down to training-only boxes
    arrow(ax, 4.65, 2.74, 3.35, 1.36, color='#D97706', ls='--', lw=1.0, rad=0.18)
    arrow(ax, 4.95, 1.94, 3.65, 1.36, color='#D97706', ls='--', lw=1.0, rad=-0.08)
    arrow(ax, 5.45, 1.94, 6.25, 1.36, color='#D97706', ls='--', lw=1.0)
    arrow(ax, 7.55, 1.94, 9.75, 1.36, color='#D97706', ls='--', lw=1.0, rad=-0.08)

    # Legend bar at very bottom
    legend_items = [
        (0.25, C_TR_FC, C_TR_EC, 'Shared ConvNeXt-Tiny (ImageNet-1K, fine-tuned)'),
        (3.85, C_V2_FC, C_V2_EC, 'New v2 disentangled subspaces & heads'),
        (7.00, C_LOSS_FC, C_LOSS_EC, 'Training-only regularisers (not used at test)'),
        (10.10, C_IO_FC, C_IO_EC, 'Input / fixed op'),
    ]
    for lx, fc, ec, lbl in legend_items:
        ax.add_patch(FancyBboxPatch((lx, 0.05), 0.24, 0.18, boxstyle='round,pad=0.01,rounding_size=0.04',
                                    facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(lx + 0.32, 0.14, lbl, va='center', ha='left', fontsize=7.6, color='#334155')

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved', OUT)


if __name__ == '__main__':
    main()
