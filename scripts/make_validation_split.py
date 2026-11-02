"""Create the one shared validation split from the official train data, stratified by composition."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_split  # noqa: E402

FRACTION = 0.15
SEED = 30082
MIN_IMAGES = 3


def choose_validation(records, fraction=FRACTION, seed=SEED, min_images=MIN_IMAGES):
    """Hold out a fixed share of each composition; compositions with fewer than min_images stay in training."""
    groups = defaultdict(list)
    for record in records:
        groups[record['composition_id']].append(record)
    generator = random.Random(seed)
    chosen = []
    for composition in sorted(groups):
        members = sorted(groups[composition], key=lambda record: record['filename'])
        if len(members) < min_images:
            continue
        generator.shuffle(members)
        chosen += members[:max(1, round(len(members) * fraction))]
    return sorted(chosen, key=lambda record: record['filename'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Replace an existing manifest. Tell the team first.')
    args = parser.parse_args()
    manifest = PROJECT / 'dataset/metadata/validation_split.csv'
    if manifest.exists() and not args.force:
        print(f'{manifest.relative_to(PROJECT)} already exists; the team shares one fixed split.')
        return 1
    records = load_split('train')
    chosen = choose_validation(records)
    with manifest.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target, lineterminator='\n')
        writer.writerow(['filename', 'composition_id', 'composition_name'])
        writer.writerows([r['filename'], r['composition_id'], r['composition_name']] for r in chosen)
    compositions = len({r['composition_id'] for r in chosen})
    print(f'Validation: {len(chosen)} images from {compositions} of 323 compositions; '
          f'train keeps {len(records) - len(chosen)} images.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
