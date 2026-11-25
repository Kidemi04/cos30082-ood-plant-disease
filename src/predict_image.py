r"""Predict crop-disease pairs for one image with the final Baseline 1 models, e.g. for the UI.

    python src\predict_image.py path\to\leaf.jpg

In Python:
    from predict_image import Baseline1
    model = Baseline1()
    model.predict('leaf.jpg')  # top 5 (composition, probability, seen) tuples
"""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels  # noqa: E402
from evaluate import Taxonomy  # noqa: E402
from predict_independent import load_model  # noqa: E402
from preprocessing import IMAGE_SIZE, RESIZE_RATIO, eval_transform, load_image  # noqa: E402

CROP_MODEL = PROJECT / 'outputs/baseline1_crop_convnext_tiny_balanced_final_seed42/final.pth'
DISEASE_MODEL = PROJECT / 'outputs/baseline1_disease_convnext_tiny_balanced_final_seed42/final.pth'


class Baseline1:
    """Both final models plus the label names; scores pairs the same way as evaluate.py."""

    def __init__(self, crop=CROP_MODEL, disease=DISEASE_MODEL, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.crop_model = load_model(crop, 'crop', self.device).eval()
        self.disease_model = load_model(disease, 'disease', self.device).eval()
        self.taxonomy = Taxonomy(load_labels())
        self.crop_of = torch.from_numpy(self.taxonomy.groups['crop'])
        self.disease_of = torch.from_numpy(self.taxonomy.groups['disease'])
        self.unseen = set(self.taxonomy.unseen.tolist())
        self.transform = eval_transform()

    @torch.no_grad()
    def predict(self, image, k=5):
        """Return the top-k (composition, probability, seen) tuples for a file path or PIL image."""
        if not isinstance(image, Image.Image):
            # Decode large JPEGs at reduced size, as the test pipeline does.
            image = load_image(image, round(IMAGE_SIZE / RESIZE_RATIO))
        batch = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        crop = self.crop_model(batch).float().log_softmax(1)[0].cpu()
        disease = self.disease_model(batch).float().log_softmax(1)[0].cpu()
        # P(crop) x P(disease), renormalised over the 333 valid compositions.
        probabilities = (crop[self.crop_of] + disease[self.disease_of]).softmax(0)
        top = probabilities.topk(k)
        names = self.taxonomy.names['composition']
        return [(names[i], p, i not in self.unseen) for p, i in zip(top.values.tolist(), top.indices.tolist())]


def main():
    parser = argparse.ArgumentParser(description='Predict crop-disease pairs for one image with Baseline 1.')
    parser.add_argument('image', type=Path)
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()
    for name, probability, seen in Baseline1().predict(args.image, args.top):
        print(f'{probability:7.2%}  {name}{"" if seen else "  (unseen pair)"}')


if __name__ == '__main__':
    main()
