"""Choose the shared pseudo-unseen compositions: seen training pairs held out to mimic the real unseen test.

The validation split only contains seen compositions, so it cannot tune anything that trades seen against
unseen accuracy (for example the calibration constant). This script picks 10 seen compositions that look like
the official unseen ones: their crop and their disease each still occur in other training compositions, so the
pair is a new combination of known concepts. A model selection run trains without them; the held-out images
then act as "unseen" validation data. Only official training images are used; test data is never touched.

Rules (deterministic, seed 30082):
- not 'healthy' (no official unseen composition is healthy);
- 10 to 150 official training images, so per-class accuracy is measurable without removing a large class;
- after removal, the crop keeps at least 2 other compositions and the disease keeps at least 1 other
  composition with at least 5 training images in total;
- at most one chosen composition per crop and per disease.
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_split  # noqa: E402

SEED = 30082
COUNT = 10
MIN_IMAGES, MAX_IMAGES = 10, 150
MANIFEST = PROJECT / 'dataset/metadata/pseudo_unseen_split.csv'


def choose(records, count=COUNT, seed=SEED):
    images = Counter(r['composition_id'] for r in records)
    parts = {r['composition_id']: (r['crop_id'], r['disease_id'], r['composition_name'], r['disease_name'])
             for r in records}
    crop_pairs, disease_pairs, disease_images = defaultdict(set), defaultdict(set), Counter()
    for composition, (crop, disease, _, _) in parts.items():
        crop_pairs[crop].add(composition)
        disease_pairs[disease].add(composition)
        disease_images[disease] += images[composition]
    candidates = sorted(c for c, (_, _, _, disease_name) in parts.items()
                        if disease_name != 'healthy' and MIN_IMAGES <= images[c] <= MAX_IMAGES)
    random.Random(seed).shuffle(candidates)
    chosen, used_crops, used_diseases = [], set(), set()
    for composition in candidates:
        crop, disease, _, _ = parts[composition]
        if crop in used_crops or disease in used_diseases:
            continue
        if len(crop_pairs[crop]) - 1 < 2 or len(disease_pairs[disease]) - 1 < 1:
            continue
        if disease_images[disease] - images[composition] < 5:
            continue
        chosen.append(composition)
        used_crops.add(crop)
        used_diseases.add(disease)
        crop_pairs[crop].discard(composition)
        disease_pairs[disease].discard(composition)
        disease_images[disease] -= images[composition]
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError(f'Only {len(chosen)} compositions satisfy the rules.')
    return sorted(chosen), images, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='Replace an existing manifest. Tell the team first.')
    args = parser.parse_args()
    if MANIFEST.exists() and not args.force:
        print(f'{MANIFEST.relative_to(PROJECT)} already exists; the team shares one fixed list.')
        return 1
    records = load_split('train')
    chosen, images, parts = choose(records)
    with MANIFEST.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target, lineterminator='\n')
        writer.writerow(['composition_id', 'composition_name', 'crop_id', 'disease_id', 'train_images'])
        for composition in chosen:
            crop, disease, name, _ = parts[composition]
            writer.writerow([composition, name, crop, disease, images[composition]])
    total = sum(images[c] for c in chosen)
    print(f'Pseudo-unseen: {len(chosen)} compositions, {total} training images held out of selection runs.')
    for composition in chosen:
        print(f'  {parts[composition][2]} ({images[composition]} images)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
