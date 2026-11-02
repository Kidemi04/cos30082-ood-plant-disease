"""Explore DPD-Small and write docs/data_exploration/ (README.md, stats.json and figures)."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_split, load_validation_split  # noqa: E402

OUT = PROJECT / 'docs/data_exploration'
SPLITS = ('train', 'test_seen', 'test_unseen')
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def source(record):
    """Original public dataset of an image, from its filename prefix (for example CDDM or PWV2)."""
    return record['filename'].split('_')[0]


def quantiles(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {'min': values[0], 'p25': pick(0.25), 'median': statistics.median(values), 'p75': pick(0.75),
            'max': values[-1], 'mean': round(statistics.fmean(values), 1)}


def image_properties(records):
    sizes, modes, formats = [], Counter(), Counter()
    for record in records:
        with Image.open(record['image_path']) as image:
            sizes.append(image.size)
            modes[image.mode] += 1
            formats[image.format] += 1
    return {'width': quantiles([w for w, _ in sizes]), 'height': quantiles([h for _, h in sizes]),
            'shorter_side': quantiles([min(s) for s in sizes]), 'modes': dict(modes), 'formats': dict(formats),
            'distinct_sizes': len(set(sizes))}


def collect():
    data = {split: load_split(split) for split in SPLITS}
    train = data['train']
    stats = {'splits': {split: {'images': len(rows),
                                'crops': len({r['crop_id'] for r in rows}),
                                'diseases': len({r['disease_id'] for r in rows}),
                                'compositions': len({r['composition_id'] for r in rows}),
                                'sources': len({source(r) for r in rows})} for split, rows in data.items()}}

    per_composition = Counter(r['composition_name'] for r in train)
    seen_test = Counter(r['composition_name'] for r in data['test_seen'])
    counts = sorted(per_composition.values())
    top10 = per_composition.most_common(10)
    stats['train_composition_imbalance'] = {
        **quantiles(counts), 'imbalance_ratio': counts[-1] / counts[0],
        'compositions_with_1_image': sum(c == 1 for c in counts),
        'compositions_with_at_most_2': sum(c <= 2 for c in counts),
        'compositions_under_10': sum(c < 10 for c in counts),
        'top10_share_of_train': sum(c for _, c in top10) / len(train),
        'largest': top10, 'sorted_counts': counts}
    seen_counts = sorted(seen_test.values())
    stats['test_seen_composition_imbalance'] = {
        **quantiles(seen_counts), 'compositions_with_1_image': sum(c == 1 for c in seen_counts),
        'compositions_with_at_most_2': sum(c <= 2 for c in seen_counts), 'largest': seen_test.most_common(5)}

    crops = Counter(r['crop_name'] for r in train)
    crop_diseases = defaultdict(set)
    disease_crops = defaultdict(set)
    for r in train:
        crop_diseases[r['crop_name']].add(r['disease_name'])
        disease_crops[r['disease_name']].add(r['crop_name'])
    stats['crops'] = {name: {'train_images': n, 'train_diseases': len(crop_diseases[name])}
                      for name, n in crops.most_common()}
    shared = sorted(((d, len(c)) for d, c in disease_crops.items() if len(c) > 1), key=lambda x: (-x[1], x[0]))
    stats['diseases'] = {'shared_by_several_crops': len(shared), 'specific_to_one_crop': len(disease_crops) - len(shared),
                         'most_shared': shared[:10],
                         'diseases_per_crop': quantiles([len(d) for d in crop_diseases.values()])}

    train_sources = Counter(source(r) for r in train)
    diseases_train = Counter(r['disease_name'] for r in train)
    unseen = []
    groups = defaultdict(list)
    for r in data['test_unseen']:
        groups[(r['composition_id'], r['composition_name'], r['crop_name'], r['disease_name'])].append(r)
    for (cid, name, crop, disease), rows in sorted(groups.items(), key=lambda item: -len(item[1])):
        sources = Counter(source(r) for r in rows)
        unseen.append({
            'composition_id': cid, 'composition': name, 'crop': crop, 'disease': disease,
            'test_images': len(rows), 'share_of_unseen_test': len(rows) / len(data['test_unseen']),
            'crop_train_images': crops[crop], 'crop_train_diseases': len(crop_diseases[crop]),
            'disease_train_images': diseases_train[disease], 'disease_train_crops': sorted(disease_crops[disease]),
            'sources': dict(sources), 'sources_in_train': {s: train_sources[s] for s in sources}})
    stats['unseen_compositions'] = unseen

    stats['sources'] = {split: Counter(source(r) for r in rows).most_common() for split, rows in data.items()}
    source_crops = defaultdict(set)
    for r in train:
        source_crops[source(r)].add(r['crop_name'])
    single = [s for s, c in source_crops.items() if len(c) == 1]
    stats['sources_with_one_crop'] = {'sources': len(single), 'of': len(source_crops),
                                      'train_images': sum(train_sources[s] for s in single)}
    stats['sources_only_in_test'] = sorted({source(r) for split in ('test_seen', 'test_unseen')
                                            for r in data[split]} - set(train_sources))
    try:
        rest, validation = load_validation_split()
        stats['validation_split'] = {'train_images': len(rest), 'validation_images': len(validation),
                                     'validation_compositions': len({r['composition_id'] for r in validation})}
    except FileNotFoundError:
        stats['validation_split'] = None
    print('Reading image headers...', flush=True)
    stats['images'] = {split: image_properties(rows) for split, rows in data.items()}
    return stats


def style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK)
    axis.set_axisbelow(True)


def figures(stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                         'figure.facecolor': 'white', 'axes.edgecolor': MUTED})

    crops = sorted(stats['crops'].items(), key=lambda item: item[1]['train_images'])
    figure, axis = plt.subplots(figsize=(7, 10))
    axis.barh([name for name, _ in crops], [v['train_images'] for _, v in crops], color=BLUE, height=0.7)
    axis.grid(axis='x', color=GRID, linewidth=0.8)
    axis.margins(y=0.01)
    axis.set(xlabel='Training images', title='Training images per crop')
    axis.tick_params(axis='y', labelsize=7)
    style(axis)
    figure.savefig(OUT / 'train_images_per_crop.png', dpi=200, bbox_inches='tight')
    plt.close(figure)

    imbalance = stats['train_composition_imbalance']
    counts = imbalance['sorted_counts'][::-1]
    figure, axis = plt.subplots(figsize=(9, 3.6))
    axis.bar(range(1, len(counts) + 1), counts, width=1.0, color=BLUE, linewidth=0)
    axis.set_yscale('log')
    axis.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500], ['1', '2', '5', '10', '20', '50', '100', '200', '500'])
    axis.minorticks_off()
    axis.axhline(imbalance['median'], color=MUTED, linewidth=1, linestyle='--')
    axis.annotate(f'median {imbalance["median"]:g} images', (len(counts), imbalance['median']),
                  xytext=(-4, 4), textcoords='offset points', ha='right', color=MUTED)
    axis.grid(axis='y', color=GRID, linewidth=0.8)
    axis.set(xlim=(0, len(counts) + 1), xlabel='Seen compositions, largest to smallest',
             ylabel='Training images (log scale)', title='Training images per seen composition')
    style(axis)
    figure.savefig(OUT / 'train_images_per_composition.png', dpi=200, bbox_inches='tight')
    plt.close(figure)

    unseen = stats['unseen_compositions'][::-1]
    labels = [u['composition'] for u in unseen]
    rows = range(len(unseen))
    figure, axis = plt.subplots(figsize=(8, 5))
    height = 0.38
    crop_bars = axis.barh([y + height / 2 + 0.01 for y in rows], [u['crop_train_images'] for u in unseen],
                          height=height, color=BLUE, label='Crop: training images with other diseases')
    disease_bars = axis.barh([y - height / 2 - 0.01 for y in rows], [u['disease_train_images'] for u in unseen],
                             height=height, color=ORANGE, label='Disease: training images on other crops')
    for bars in (crop_bars, disease_bars):
        axis.bar_label(bars, padding=3, fontsize=7, color=MUTED)
    axis.set_yticks(list(rows), labels)
    axis.grid(axis='x', color=GRID, linewidth=0.8)
    axis.set(xlabel='Training images', title='Training support for each unseen composition')
    axis.legend(loc='lower right', frameon=False)
    style(axis)
    figure.savefig(OUT / 'unseen_composition_support.png', dpi=200, bbox_inches='tight')
    plt.close(figure)


def augmentation_examples():
    """Row 1: evaluation preprocessing; rows 2-3: two random training augmentations. Needs PyTorch."""
    try:
        import torch
        from torchvision.utils import save_image
        import preprocessing
    except ImportError:
        print('PyTorch is not installed; skipped augmentation_examples.png')
        return False
    torch.manual_seed(0)
    records = load_validation_split()[0][::700][:8]
    views = [preprocessing.DPDDataset(records, preprocessing.eval_transform())]
    views += [preprocessing.DPDDataset(records, preprocessing.train_transform())] * 2
    grid = torch.stack([preprocessing.denormalise(view[i]['image']) for view in views for i in range(len(records))])
    save_image(grid, OUT / 'augmentation_examples.png', nrow=len(records), padding=4, pad_value=1.0)
    return [r['composition_name'] for r in records]


def pct(value):
    return f'{100 * value:.1f}%'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return lines + ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]


def readme(stats):
    s, imb, seen_imb = stats['splits'], stats['train_composition_imbalance'], stats['test_seen_composition_imbalance']
    unseen, diseases, images = stats['unseen_compositions'], stats['diseases'], stats['images']
    biggest = unseen[0]
    weakest = min(unseen, key=lambda u: u['disease_train_images'])
    lines = ['# DPD-Small data exploration', '',
             'Generated by `python scripts/explore_dataset.py` from the prepared official splits. '
             'Do not edit by hand; rerun the script. Full numbers are in [stats.json](stats.json).', '',
             '## Splits', '']
    lines += table(['Split', 'Images', 'Crops', 'Diseases', 'Compositions', 'Source datasets'],
                   [[split, v['images'], v['crops'], v['diseases'], v['compositions'], v['sources']]
                    for split, v in s.items()])
    if stats['validation_split']:
        v = stats['validation_split']
        lines += ['', f'The shared validation split takes {v["validation_images"]} training images from '
                      f'{v["validation_compositions"]} compositions, leaving {v["train_images"]} for training.']
    lines += ['', 'All 323 seen compositions appear in both train and seen test. '
                  'The 10 unseen compositions appear only in unseen test.', '',
              '## Class imbalance', '',
              '![Training images per seen composition](train_images_per_composition.png)', '',
              f'- Training images per composition range from {imb["min"]} to {imb["max"]} '
              f'(median {imb["median"]:g}, mean {imb["mean"]}), a {imb["imbalance_ratio"]:.0f}:1 ratio.',
              f'- {imb["compositions_with_1_image"]} compositions have a single training image, '
              f'{imb["compositions_with_at_most_2"]} have at most two, and {imb["compositions_under_10"]} of 323 '
              f'have fewer than ten.',
              f'- The 10 largest compositions hold {pct(imb["top10_share_of_train"])} of the training set: '
              + ', '.join(f'{name} ({n})' for name, n in imb['largest'][:5]) + ', ...',
              f'- Seen test is just as uneven: {seen_imb["min"]} to {seen_imb["max"]} images per composition '
              f'(median {seen_imb["median"]:g}); {seen_imb["compositions_with_1_image"]} compositions have one test '
              f'image. Per-class accuracy on those is either 0% or 100%, which is why the evaluation reports both '
              f'per-class and per-image accuracy.', '',
              '## Crops and diseases', '',
              '![Training images per crop](train_images_per_crop.png)', '',
              f'- {diseases["shared_by_several_crops"]} diseases occur on more than one crop and '
              f'{diseases["specific_to_one_crop"]} on a single crop. Most shared: '
              + ', '.join(f'{d} ({n} crops)' for d, n in diseases['most_shared'][:6]) + '.',
              f'- Crops have {diseases["diseases_per_crop"]["min"]} to {diseases["diseases_per_crop"]["max"]} '
              f'diseases in training (median {diseases["diseases_per_crop"]["median"]:g}).',
              '- Shared disease names are what make the compositional formulation possible: a disease learnt on one '
              'crop has to be recognised on another.', '',
              '## The 10 unseen compositions', '',
              '![Training support for each unseen composition](unseen_composition_support.png)', '']
    lines += table(['Composition', 'Unseen test images', 'Share', 'Crop train images (diseases)',
                    'Disease train images', 'Disease seen on crops'],
                   [[u['composition'], u['test_images'], pct(u['share_of_unseen_test']),
                     f'{u["crop_train_images"]} ({u["crop_train_diseases"]})', u['disease_train_images'],
                     ', '.join(u['disease_train_crops'])] for u in unseen])
    lines += ['', '- Every unseen crop and disease occurs in training, just never together, so the task is '
                  'recombining known concepts rather than recognising a completely new disease.',
              f'- {biggest["composition"]} is {pct(biggest["share_of_unseen_test"])} of the unseen test set, so '
              f'per-image unseen accuracy mostly measures this one class.',
              f'- {weakest["composition"]} has the least disease evidence: {weakest["disease"]} appears in only '
              f'{weakest["disease_train_images"]} training images, on '
              + ', '.join(weakest['disease_train_crops']) + '.', '']
    lines += table(['Composition', 'Source datasets of unseen images (training images from that source)'],
                   [[u['composition'], ', '.join(f'{src} {n} ({u["sources_in_train"][src]})'
                                                for src, n in u['sources'].items())] for u in unseen])
    one_crop = stats['sources_with_one_crop']
    lines += ['', '## Source datasets and image properties', '',
              f'Filename prefixes show DPD-Small merges {s["train"]["sources"]} source datasets in training. '
              'Largest training sources: '
              + ', '.join(f'{src} ({n})' for src, n in stats['sources']['train'][:6]) + '.', '',
              f'{one_crop["sources"]} of {one_crop["of"]} sources contain a single crop '
              f'({one_crop["train_images"]} training images). Sources usually differ in camera, background and '
              'lighting, so a model can recognise the crop from the photo style instead of the plant. Expect crop '
              'accuracy to be high and disease accuracy to be the real difficulty; unseen compositions from '
              'single-crop sources (for example mango_anthracnose from MLBD) test whether the disease is recognised, '
              'not only the source.']
    if stats['sources_only_in_test']:
        lines += ['', 'Sources that appear only in test: ' + ', '.join(stats['sources_only_in_test']) + '.']
    lines += ['']
    lines += table(['Split', 'Width median (min-max)', 'Height median (min-max)', 'Shorter side median', 'Formats',
                    'Colour modes'],
                   [[split, f'{p["width"]["median"]:g} ({p["width"]["min"]}-{p["width"]["max"]})',
                     f'{p["height"]["median"]:g} ({p["height"]["min"]}-{p["height"]["max"]})',
                     f'{p["shorter_side"]["median"]:g}', ', '.join(f'{k} {v}' for k, v in p['formats'].items()),
                     ', '.join(f'{k} {v}' for k, v in p['modes'].items())] for split, p in images.items()])
    shift = images['test_unseen']['shorter_side']['median'], images['train']['shorter_side']['median']
    lines += ['', 'Image sizes vary widely and a few images are RGBA, so every model resizes in its input pipeline '
                  f'and converts to RGB. Unseen test images are smaller (median shorter side {shift[0]:g} px against '
                  f'{shift[1]:g} px in train), so a model that has only seen sharp images is also tested on lower '
                  'resolution. Random-resized-crop augmentation covers part of this. See the preprocessing section '
                  'of the main README.', '',
              '## Manual observations', '',
              '- Not every image is a single leaf photo. CDDM_wheat_healthy_131678.jpg is a collage of figure panels '
              'captioned "Yellow rust", yet labelled wheat_healthy. Such images are kept, since the official splits '
              'must not change, but they add label noise.', '']
    if (OUT / 'augmentation_examples.png').is_file():
        lines += ['## Preprocessing and augmentation examples', '',
                  '![Preprocessing and augmentation examples](augmentation_examples.png)', '',
                  'Top row: evaluation preprocessing (resize to 256, centre-crop 224). Lower rows: two random '
                  'training augmentations of the same images (random resized crop, flips, colour jitter). '
                  'Images: ' + ', '.join(stats['augmentation_examples']) + '.', '']
    return '\n'.join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = collect()
    examples = augmentation_examples()
    if examples:
        stats['augmentation_examples'] = examples
    elif (OUT / 'augmentation_examples.png').is_file():
        stats['augmentation_examples'] = json.loads((OUT / 'stats.json').read_text())['augmentation_examples']
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    figures(stats)
    (OUT / 'README.md').write_text(readme(stats), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(PROJECT)}')


if __name__ == '__main__':
    main()
