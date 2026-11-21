r"""Report figure: the architectures of Baseline 1, CSP and CSP + WiSE-FT, drawn in one visual language.

    python src\architecture_figure.py

writes results/figures/architectures.png (300 dpi) and .pdf (vector, for crisp printing). Grey boxes are frozen
pretrained parts, blue boxes are trained on DPD-Small, white boxes are inputs, outputs and fixed operations; each
panel's square matches the model's colour in report_figures.py.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT  # noqa: E402
from report_figures import COLOURS, INK, INK_SECONDARY, MUTED, AXIS, style  # noqa: E402

KINDS = {'frozen': ('#f0efec', MUTED), 'trained': ('#dcebfc', '#2a78d6'), 'data': ('#ffffff', AXIS)}
OUT = PROJECT / 'results/figures'


def box(axis, x, y, width, text, kind, height=7.5):
    """A rounded box centred vertically on y; returns its left, right and centre-y for arrows."""
    fill, edge = KINDS[kind]
    axis.add_patch(FancyBboxPatch((x, y - height / 2), width, height, boxstyle='round,pad=0,rounding_size=1.2',
                                  facecolor=fill, edgecolor=edge, linewidth=1))
    axis.text(x + width / 2, y, text, ha='center', va='center', fontsize=7.5, color=INK, linespacing=1.3)
    return x, x + width, y


def arrow(axis, start, end):
    axis.annotate('', xy=end, xytext=start,
                  arrowprops={'arrowstyle': '-|>', 'color': MUTED, 'linewidth': 1, 'shrinkA': 0, 'shrinkB': 0,
                              'mutation_scale': 8})


def heading(axis, top, model, title, summary):
    axis.add_patch(FancyBboxPatch((1, top - 1.1), 1.8, 1.8, boxstyle='round,pad=0,rounding_size=0.3',
                                  facecolor=COLOURS[model], edgecolor='none'))
    axis.text(4, top, title, va='center', fontsize=9.5, fontweight='semibold', color=INK)
    axis.text(4, top - 3, summary, va='center', fontsize=7.5, color=INK_SECONDARY)


def baseline1(axis, top):
    heading(axis, top, 'Baseline 1', '(a) Baseline 1: two independent CNNs',
            'Trained: both networks, 55.8M parameters  ·  Test H 15.8%')
    upper, lower, middle = top - 10, top - 21, top - 15.5
    image = box(axis, 2, middle, 11, 'Leaf image\n224 × 224', 'data')
    for y, classes, name in ((upper, 55, 'crop'), (lower, 175, 'disease')):
        network = box(axis, 18, y, 20, 'ConvNeXt-Tiny\n(ImageNet weights)', 'trained')
        head = box(axis, 42, y, 14, f'Linear\n768 → {classes}', 'trained')
        output = box(axis, 60, y, 14, f'P({name})\n{classes} {name}s', 'data')
        arrow(axis, (image[1], middle), (network[0], y))
        arrow(axis, (network[1], y), (head[0], y))
        arrow(axis, (head[1], y), (output[0], y))
        arrow(axis, (output[1], y), (79, middle + (2 if y == upper else -2)))
    box(axis, 79, middle, 19, 'P(crop) × P(disease)\nrenormalised over\nthe 333 valid pairs', 'data', height=12)


def clip_model(axis, top, model, title, summary, encoder_kind, encoder_note):
    heading(axis, top, model, title, summary)
    upper, lower, middle = top - 10, top - 22.5, top - 16
    image = box(axis, 2.5, upper, 11, 'Leaf image\n224 × 224', 'data')
    image_encoder = box(axis, 45, upper, 20, f'CLIP image encoder\nViT-B/16\n{encoder_note}', encoder_kind,
                        height=8.5)
    image_feature = box(axis, 68, upper, 13, 'image feature\n512-d', 'data')
    # The three prompt pieces form one token sequence, so a hairline container groups them.
    axis.add_patch(FancyBboxPatch((1, lower - 5.6), 40.5, 11.2, boxstyle='round,pad=0,rounding_size=1.2',
                                  facecolor='none', edgecolor=AXIS, linewidth=0.8))
    axis.text(2, lower + 7.2, 'prompt for every pair: "a photo of [disease] [crop]"', va='center', fontsize=7,
              color=INK_SECONDARY, style='italic')
    box(axis, 2.5, lower, 11, '"a photo of"\nfixed words', 'frozen')
    box(axis, 15.5, lower, 12, '[disease]\n175 learned\nvectors', 'trained', height=8.5)
    box(axis, 29.5, lower, 10.5, '[crop]\n55 learned\nvectors', 'trained', height=8.5)
    text_encoder = box(axis, 45, lower, 20, f'CLIP text encoder\nTransformer\n{encoder_note}', encoder_kind,
                       height=8.5)
    text_features = box(axis, 68, lower, 13, '333 pair\nfeatures\n512-d', 'data', height=8.5)
    score = box(axis, 84, middle, 15, 'cosine similarity\n× 100 → scores\nfor 333 pairs', 'data', height=12)
    arrow(axis, (image[1], upper), (image_encoder[0], upper))
    arrow(axis, (41.5, lower), (text_encoder[0], lower))
    arrow(axis, (image_encoder[1], upper), (image_feature[0], upper))
    arrow(axis, (text_encoder[1], lower), (text_features[0], lower))
    arrow(axis, (image_feature[1], upper), (score[0], middle + 2))
    arrow(axis, (text_features[1], lower), (score[0], middle - 2))


def legend(axis, y):
    x = 4
    for kind, label in (('frozen', 'frozen: pretrained, not updated'), ('trained', 'trained on DPD-Small'),
                        ('data', 'input, output or fixed operation')):
        fill, edge = KINDS[kind]
        axis.add_patch(FancyBboxPatch((x, y - 1), 3, 2, boxstyle='round,pad=0,rounding_size=0.4',
                                      facecolor=fill, edgecolor=edge, linewidth=1))
        axis.text(x + 4, y, label, va='center', fontsize=7.5, color=INK_SECONDARY)
        x += 31


def main():
    style()
    figure, axis = plt.subplots(figsize=(7.2, 8.6))
    axis.set(xlim=(0, 100), ylim=(0, 100))
    axis.axis('off')
    baseline1(axis, 98)
    clip_model(axis, 66, 'CSP', '(b) CSP: frozen CLIP + learned crop and disease words',
               'Trained: only the 230 word vectors, 117,760 numbers  ·  Test H 21.7%', 'frozen', 'frozen')
    clip_model(axis, 34, 'CSP + WiSE-FT', '(c) CSP + WiSE-FT: fine-tuned CLIP blended with CSP (final model)',
               'Trained: CLIP layers + word vectors (124.4M), then blended 0.5 × (b) + 0.5 × fine-tuned  ·  Test H 28.5%',
               'trained', 'fine-tuned + WiSE-FT')
    legend(axis, 2.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        figure.savefig(OUT / f'architectures.{suffix}')
    print(f'Saved {OUT / "architectures"}.png and .pdf')


if __name__ == '__main__':
    main()
