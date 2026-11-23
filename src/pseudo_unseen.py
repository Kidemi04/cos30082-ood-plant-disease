r"""Pseudo-unseen validation: hold out 10 seen pairs to measure unseen accuracy without the test set.

The shared validation split holds only seen pairs, so it cannot show unseen accuracy or choose a calibration
gamma. This holds out 10 seen pairs that resemble the real unseen ones: not healthy, at least 20 training images,
distinct crops and diseases, and each crop and disease still present in other training pairs. A model trained
without them is scored on their images as unseen, and on the rest of the validation split as seen.

It is only for choosing settings (epochs, gamma). Final models are still retrained on all 323 seen pairs.

    python src\pseudo_unseen.py    # writes dataset/metadata/pseudo_unseen_pairs.json once (fixed seed)
"""
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels, load_split, load_validation_split  # noqa: E402
from evaluate import Taxonomy, split_records  # noqa: E402

PAIRS_FILE = PROJECT / 'dataset/metadata/pseudo_unseen_pairs.json'
COUNT, SEED, MIN_IMAGES = 10, 30082, 20
SPLITS = ('pseudo_val_seen', 'pseudo_val_unseen')


def choose_pairs(taxonomy, records, count=COUNT, seed=SEED, min_images=MIN_IMAGES):
    """Pick held-out pairs at random among those that look like the real unseen pairs."""
    images = Counter(record['composition_id'] for record in records)
    crop_of = lambda pair: int(taxonomy.groups['crop'][pair])  # noqa: E731
    disease_of = lambda pair: int(taxonomy.groups['disease'][pair])  # noqa: E731
    candidates = [int(pair) for pair in taxonomy.seen if images[pair] >= min_images
                  and taxonomy.names['disease'][disease_of(pair)] != 'healthy']
    random.Random(seed).shuffle(candidates)
    chosen = []
    for pair in candidates:
        if crop_of(pair) in map(crop_of, chosen) or disease_of(pair) in map(disease_of, chosen):
            continue
        remaining = set(taxonomy.seen.tolist()) - set(chosen) - {pair}
        if all(crop_of(held) in map(crop_of, remaining) and disease_of(held) in map(disease_of, remaining)
               for held in chosen + [pair]):
            chosen.append(pair)
        if len(chosen) == count:
            return sorted(chosen)
    raise ValueError(f'Only {len(chosen)} pairs meet the rules; lower min_images.')


def held_out_pairs(path=PAIRS_FILE):
    return set(json.loads(Path(path).read_text(encoding='utf-8'))['composition_ids'])


def trained_pairs(taxonomy, path=PAIRS_FILE):
    """The 313 seen pairs a pseudo-unseen model trains on, sorted."""
    held_out = held_out_pairs(path)
    return np.array([pair for pair in taxonomy.seen if pair not in held_out])


def load_pseudo_unseen_split(path=PAIRS_FILE):
    """Train, seen validation and unseen validation records for a pseudo-unseen run."""
    held_out = held_out_pairs(path)
    train, val = load_validation_split()
    return {'train': [record for record in train if record['composition_id'] not in held_out],
            'pseudo_val_seen': [record for record in val if record['composition_id'] not in held_out],
            'pseudo_val_unseen': [record for record in train + val if record['composition_id'] in held_out]}


def records_for(split):
    """Records of an official split, the shared validation split or a pseudo-unseen validation split."""
    return load_pseudo_unseen_split()[split] if split in SPLITS else split_records(split)


def main():
    taxonomy = Taxonomy(load_labels())
    records = load_split('train')
    pairs = choose_pairs(taxonomy, records)
    images = Counter(record['composition_id'] for record in records)
    PAIRS_FILE.write_text(json.dumps({
        'rule': f'{COUNT} seen pairs, not healthy, >= {MIN_IMAGES} official training images, distinct crops and '
                f'diseases, each crop and disease kept in other training pairs; random.Random({SEED})',
        'composition_ids': pairs,
        'pairs': {taxonomy.names['composition'][pair]: images[pair] for pair in pairs},
    }, indent=2) + '\n', encoding='utf-8')
    split = load_pseudo_unseen_split()
    print(f'Wrote {PAIRS_FILE}')
    for name, count in json.loads(PAIRS_FILE.read_text(encoding='utf-8'))['pairs'].items():
        print(f'  {name}: {count} images')
    print(', '.join(f'{key} {len(value)}' for key, value in split.items()))


if __name__ == '__main__':
    main()
