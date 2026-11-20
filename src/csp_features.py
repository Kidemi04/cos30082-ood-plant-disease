r"""Frozen CLIP image features for CSP, and training-image embeddings for nearest-neighbour lookup in the UI.

The CLIP image encoder never changes during CSP training, so every image is encoded once and cached. Each image
gets a clean view (the shared evaluation transform with CLIP's normalisation); training images also get
augmented views (the shared training transform), and CSP training draws one view per image per epoch.

    python src\csp_features.py

writes the feature cache (all splits) and the training-image embeddings (8,000 images) to outputs/clip_features/.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csp_model import CLIP_MODEL, CLIP_WEIGHTS, image_normalisation, load_clip  # noqa: E402
from dataset_access import PROJECT, load_split  # noqa: E402
from preprocessing import DPDDataset, eval_transform, train_transform  # noqa: E402

FEATURES = PROJECT / f'outputs/clip_features/{CLIP_MODEL}_{CLIP_WEIGHTS}.pt'
TRAIN_EMBEDDINGS = PROJECT / f'outputs/clip_features/{CLIP_MODEL}_{CLIP_WEIGHTS}_train_embeddings.npz'


@torch.no_grad()
def encode_images(clip_model, records, transform, device, batch_size=128, num_workers=4):
    """CLIP image features (not normalised) for the records, in order, as float16."""
    loader = DataLoader(DPDDataset(records, transform), batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device.type == 'cuda')
    features = []
    for batch in loader:
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            features.append(clip_model.encode_image(batch['image'].to(device, non_blocking=True)).float().cpu())
    return torch.cat(features).half()


def build_feature_cache(path=FEATURES, augmented_views=4, seed=0, num_workers=4):
    """Encode the official train split (clean + augmented views) and both test splits (clean) once."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = load_clip(device)
    mean, std = image_normalisation(clip_model)
    train = load_split('train')  # all 8,000 training images; the shared validation split is a subset
    records = train + load_split('test_seen') + load_split('test_unseen')
    filenames = [record['filename'] for record in records]
    if len(set(filenames)) != len(filenames):
        raise ValueError('Filenames must be unique across splits.')

    print(f'Encoding {len(records)} images (clean view) with {CLIP_MODEL} on {device}', flush=True)
    clean = encode_images(clip_model, records, eval_transform(mean=mean, std=std), device, num_workers=num_workers)
    torch.manual_seed(seed)
    augmented = []
    for view in range(augmented_views):
        print(f'Encoding {len(train)} training images (augmented view {view + 1}/{augmented_views})', flush=True)
        augmented.append(encode_images(clip_model, train, train_transform(mean=mean, std=std), device,
                                       num_workers=num_workers))
    cache = {'clip_model': CLIP_MODEL, 'clip_weights': CLIP_WEIGHTS, 'filenames': filenames, 'clean': clean,
             'augmented': torch.stack(augmented) if augmented else clean.new_empty(0, len(train), clean.shape[1])}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


class FeatureCache:
    """Looks up cached CLIP features by record."""

    def __init__(self, path=FEATURES):
        data = torch.load(path, weights_only=True)
        self.clip_model, self.clip_weights = data['clip_model'], data['clip_weights']
        self.row = {name: row for row, name in enumerate(data['filenames'])}
        self.clean, self.augmented = data['clean'], data['augmented']

    def features(self, records, augmented=False):
        """Features (views, images, dim) in record order: the clean view, then augmented views if asked."""
        rows = torch.tensor([self.row[record['filename']] for record in records])
        views = [self.clean[rows]]
        if augmented and len(self.augmented):
            if int(rows.max()) >= self.augmented.shape[1]:
                raise ValueError('Augmented views exist only for the official train split.')
            views += list(self.augmented[:, rows])
        return torch.stack(views)


def save_training_embeddings(cache, path=TRAIN_EMBEDDINGS):
    """Save L2-normalised CLIP embeddings of all 8,000 training images with their paths and labels."""
    train = load_split('train')
    embeddings = F.normalize(cache.features(train)[0].float(), dim=-1)
    np.savez_compressed(path, embeddings=embeddings.half().numpy(),
                        filenames=np.array([record['filename'] for record in train]),
                        image_paths=np.array([Path(record['image_path']).relative_to(PROJECT).as_posix()
                                              for record in train]),
                        compositions=np.array([record['composition_name'] for record in train]))
    return path


def load_training_embeddings(path=TRAIN_EMBEDDINGS):
    with np.load(path) as data:
        bank = {key: data[key] for key in data.files}
    bank['embeddings'] = torch.from_numpy(bank['embeddings'].astype(np.float32))
    return bank


def nearest_training_images(query, bank, k=3):
    """The k training images closest (cosine similarity) to one CLIP image embedding."""
    query = F.normalize(torch.as_tensor(query, dtype=torch.float32).flatten(), dim=0)
    top = (bank['embeddings'] @ query).topk(k)
    return [{'filename': str(bank['filenames'][i]), 'image_path': PROJECT / str(bank['image_paths'][i]),
             'composition': str(bank['compositions'][i]), 'similarity': similarity}
            for similarity, i in zip(top.values.tolist(), top.indices.tolist())]


def main():
    parser = argparse.ArgumentParser(description='Cache CLIP image features for CSP and the UI.')
    parser.add_argument('--augmented-views', type=int, default=4, help='augmented views per training image')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    build_feature_cache(FEATURES, args.augmented_views, args.seed, args.num_workers)
    cache = FeatureCache(FEATURES)
    print(f'Saved {FEATURES}\nSaved {save_training_embeddings(cache)}')


if __name__ == '__main__':
    main()
