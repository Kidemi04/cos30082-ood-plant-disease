"""Check the UI's CSP + WiSE-FT scoring against Nigel's own CSPPredictor (src/predict_csp.py).

Uses training images only: the checkpoint's README asks that the test sets are not run again.
  python ui\\check_csp.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import ROOT, Predictor  # noqa: E402
from dataset_access import load_split  # noqa: E402
from preprocessing import IMAGE_SIZE, RESIZE_RATIO, load_image  # noqa: E402
from predict_csp import CSPPredictor  # noqa: E402


def main(n=30):
    p = Predictor()
    spec = p.config['models']['csp']
    theirs = CSPPredictor(ROOT / spec['checkpoint'], gamma=spec['gamma'])
    names = p.labels.taxonomy.names['composition']
    records = load_split('train')[::8000 // n][:n]
    same = top1_same = 0
    for r in records:
        image = load_image(r['image_path'], round(IMAGE_SIZE / RESIZE_RATIO))
        ours = [names[t['id']] for t in p.predict('csp', image)['top']]
        ref = [name for name, _, _ in theirs.predict(image)['pairs']]
        same += ours == ref
        top1_same += ours[0] == ref[0]
    print(f'training images: identical top-5 {same}/{len(records)}, identical top-1 {top1_same}/{len(records)}')
    print('gammas:', {k: m['gamma'] for k, m in p.meta.items()}, '| available:', [k for k, m in p.meta.items() if m['available']])


if __name__ == '__main__':
    main()
