"""Check the UI's Disentangled v2 scoring against Kelvin's own predictor (src/infer_disentangled_v2.py).

  python ui\\check_disentangled.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import ROOT, Predictor  # noqa: E402
from dataset_access import load_split  # noqa: E402
from preprocessing import load_image  # noqa: E402
from infer_disentangled_v2 import DisentangledV2Predictor  # noqa: E402


def main(n=20):
    p = Predictor()
    theirs = DisentangledV2Predictor(ROOT / 'runs/disentangled_v2_final_seed42/final.pth')
    for split in ('test_seen', 'test_unseen'):
        records = load_split(split)
        records = records[:: max(1, len(records) // n)][:n]
        same_top5 = correct = 0
        for r in records:
            image = load_image(r['image_path'])
            ours = [t['id'] for t in p.predict('disentangled', image)['top']]
            ref = [t['composition_id'] for t in theirs.predict_image(image, top_k=5)['top_compositions']]
            same_top5 += ours == ref
            correct += ours[0] == r['composition_id']
        print(f'{split}: identical top-5 to Kelvin\'s predictor {same_top5}/{len(records)}, top-1 correct {correct}/{len(records)}')
    print('gammas:', {k: m['gamma'] for k, m in p.meta.items()}, '| available:', [k for k, m in p.meta.items() if m['available']])


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    main()
