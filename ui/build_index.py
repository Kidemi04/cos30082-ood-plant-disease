"""Build everything the UI needs from the dataset, once, so the demo laptop needs no dataset folder.

Writes to ui/assets/:
  index.npz        image embeddings of all 8,000 training images (index model from ui/config.json),
                   with their composition / crop / disease ids and thumbnail paths
  thumbs/          small JPEG thumbnails of the training images (shown as "similar training images")
  examples/        a few test images for the "Try an example" buttons (2 seen pairs, 2 unseen pairs)
  examples.json    their labels
  label_map.json   copy of the label names
Rebuild after the index model changes:  python ui\\build_index.py
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import ASSETS, ROOT, Predictor  # noqa: E402
from dataset_access import load_split  # noqa: E402
from preprocessing import DPDDataset, load_image  # noqa: E402

THUMB = 240
EXAMPLE = 512
# Example pairs: (composition name, seen?). The proposed model gets the chosen image right.
EXAMPLE_PAIRS = [('tomato_early_blight', False), ('grape_black_rot', False),
                 ('olive_peacock_spot', True), ('cassava_mosaic', True)]


def thumbnail(record, folder, size):
    out = (folder / record['filename']).with_suffix('.jpg')
    if not out.is_file():
        image = load_image(record['image_path'], size * 2)
        image.thumbnail((size, size))
        image.save(out, quality=82)
    return out.name


@torch.no_grad()
def embeddings(predictor, records):
    key = predictor.config['index_model']
    net, transform = predictor.load(key)
    loader = DataLoader(DPDDataset(records, transform), batch_size=64, num_workers=4)
    chunks = []
    for i, batch in enumerate(loader):
        x = batch['image'].to(predictor.device)
        with torch.amp.autocast(predictor.device.type, enabled=predictor.device.type == 'cuda'):
            chunks.append(net(x)['embedding'].float().cpu().numpy())
        print(f'  embedded {min((i + 1) * 64, len(records))}/{len(records)}', end='\r', flush=True)
    print()
    return np.concatenate(chunks)


def pick_examples(predictor, folder):
    """One test image per example pair that the index model classifies correctly and confidently."""
    chosen = []
    by_name = {}
    for split in ('test_seen', 'test_unseen'):
        for r in load_split(split):
            by_name.setdefault(r['composition_name'], []).append(r)
    for name, seen in EXAMPLE_PAIRS:
        best = None
        for r in by_name.get(name, [])[:25]:
            out = predictor.predict(predictor.config['index_model'], load_image(r['image_path'], 448))
            if out['best']['id'] == r['composition_id'] and (best is None or out['best']['p'] > best[1]):
                best = (r, out['best']['p'], out['best']['title'])
        if best is None:
            print(f'  no confident correct example for {name}; skipped')
            continue
        r, p, title = best
        file = thumbnail(r, folder, EXAMPLE)
        chosen.append({'file': f'examples/{file}', 'title': title, 'seen': seen, 'source': r['filename']})
        print(f'  example: {title} ({"seen" if seen else "unseen"}), confidence {p:.2f}')
    return chosen


def main():
    predictor = Predictor()
    thumbs, examples = ASSETS / 'thumbs', ASSETS / 'examples'
    thumbs.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'dataset/prepared/label_map.json', ASSETS / 'label_map.json')

    records = load_split('train')
    print(f'Thumbnails for {len(records)} training images...')
    with ThreadPoolExecutor(8) as pool:
        names = list(pool.map(lambda r: thumbnail(r, thumbs, THUMB), records))

    print(f'Embeddings with the {predictor.config["index_model"]} model on {predictor.device_label}...')
    emb = embeddings(predictor, records)
    np.savez_compressed(ASSETS / 'index.npz', embeddings=emb.astype(np.float16),
                        composition=np.array([r['composition_id'] for r in records], dtype=np.int16),
                        crop=np.array([r['crop_id'] for r in records], dtype=np.int16),
                        disease=np.array([r['disease_id'] for r in records], dtype=np.int16),
                        thumbs=np.array([f'thumbs/{n}' for n in names]))

    print('Choosing example images...')
    chosen = pick_examples(predictor, examples)
    (ASSETS / 'examples.json').write_text(json.dumps(chosen, indent=2) + '\n', encoding='utf-8')
    size = sum(f.stat().st_size for f in ASSETS.rglob('*') if f.is_file()) / 2 ** 20
    print(f'Done: ui/assets is {size:.0f} MB.')


if __name__ == '__main__':
    main()
