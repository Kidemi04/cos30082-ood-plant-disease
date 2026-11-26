"""Sanity check: the UI's scoring must reproduce the saved, calibrated test predictions.

For a few test images, compares the predictor's top pair with the top pair in the run's
test_*_calibrated.npz (written by gzsl.py calibrate). Run from the repo root:
  python ui\\check_predictor.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import ROOT, Predictor  # noqa: E402
from dataset_access import load_split  # noqa: E402
from preprocessing import load_image  # noqa: E402


def main(n=40):
    p = Predictor()
    run = ROOT / p.config['models']['proposed']['run']
    print('Device:', p.device_label, '| models:', {k: (m['available'], m['gamma']) for k, m in p.meta.items()})
    for split in ('test_seen', 'test_unseen'):
        saved = np.load(run / f'{split}_calibrated.npz', allow_pickle=False)
        index = {str(f): i for i, f in enumerate(saved['filenames'])}
        records = load_split(split)[:: max(1, len(load_split(split)) // n)][:n]
        agree = correct = 0
        for r in records:
            out = p.predict('proposed', load_image(r['image_path']))
            ours = out['best']['id']
            theirs = int(np.argmax(saved['composition'][index[r['filename']]]))
            agree += ours == theirs
            correct += ours == r['composition_id']
        print(f'{split}: top pair matches saved predictions {agree}/{len(records)}, correct {correct}/{len(records)}')
    for key in p.meta:
        if p.meta[key]['available']:
            out = p.predict(key, load_image(load_split('test_unseen')[0]['image_path']))
            print(f"{key}: {out['best']['title']} {out['best']['p']:.2f} in {out['seconds'] * 1000:.0f} ms")


if __name__ == '__main__':
    main()
