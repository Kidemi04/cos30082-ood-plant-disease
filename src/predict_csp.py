r"""Save CSP predictions for evaluate.py, and CSPPredictor for the UI.

While developing, predict the validation split and score it:
    python src\predict_csp.py --name csp --checkpoint <best.pt>
    python evaluate.py run --name csp_dev --val predictions\csp_val.npz

Once the model is final, predict both test splits once and score them:
    python src\predict_csp.py --name csp --checkpoint <final.pt> --splits test_seen test_unseen
    python evaluate.py run --name csp_vit_b16 --seen predictions\csp_test_seen.npz --unseen predictions\csp_test_unseen.npz

In a UI:
    from predict_csp import CSPPredictor
    predictor = CSPPredictor(<final.pt>)
    predictor.predict(image)  # top pairs and the 3 most similar training images
"""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csp_features import FEATURES, TRAIN_EMBEDDINGS, FeatureCache, load_training_embeddings, \
    nearest_training_images  # noqa: E402
from csp_model import image_normalisation, load_csp  # noqa: E402
from dataset_access import PROJECT, load_labels  # noqa: E402
from evaluate import Taxonomy, save_predictions, split_records  # noqa: E402
from preprocessing import IMAGE_SIZE, RESIZE_RATIO, eval_transform, load_image  # noqa: E402


class CSPPredictor:
    """A trained CSP model plus the training-image embeddings, for predictions with nearest neighbours."""

    def __init__(self, checkpoint, embeddings=TRAIN_EMBEDDINGS, gamma=0.0, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = load_csp(checkpoint, self.device)
        mean, std = image_normalisation(self.model.clip)
        self.transform = eval_transform(mean=mean, std=std)
        self.bank = load_training_embeddings(embeddings)
        taxonomy = Taxonomy(load_labels())
        self.names = taxonomy.names['composition']
        self.unseen = set(taxonomy.unseen.tolist())
        # Calibrated stacking: subtract gamma from seen compositions' log-probabilities (0 = off).
        self.seen_penalty = torch.zeros(len(self.names))
        self.seen_penalty[torch.as_tensor(taxonomy.seen)] = gamma

    @torch.no_grad()
    def predict(self, image, k=5, neighbours=3):
        """Top-k (composition, probability, seen) tuples and the nearest training images for a path or PIL image."""
        if not isinstance(image, Image.Image):
            image = load_image(image, round(IMAGE_SIZE / RESIZE_RATIO))
        batch = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            feature = self.model.clip.encode_image(batch).float()
        log_p = self.model.all_logits(feature)[0].cpu().log_softmax(0) - self.seen_penalty
        top = log_p.softmax(0).topk(k)
        pairs = [(self.names[i], p, i not in self.unseen) for p, i in zip(top.values.tolist(), top.indices.tolist())]
        return {'pairs': pairs, 'neighbours': nearest_training_images(feature[0].cpu(), self.bank, neighbours)}


def main():
    parser = argparse.ArgumentParser(description='Save CSP predictions for evaluate.py.')
    parser.add_argument('--name', required=True, help='Prefix of the prediction files.')
    parser.add_argument('--checkpoint', required=True, type=Path, help='best.pt or final.pt from train_csp.py')
    parser.add_argument('--splits', nargs='+', default=['val'], choices=['val', 'test_seen', 'test_unseen'])
    parser.add_argument('--features', type=Path, default=FEATURES)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_csp(args.checkpoint, device)
    cache = FeatureCache(args.features)
    for split in args.splits:
        records = split_records(split)
        logits = model.all_logits(cache.features(records)[0].to(device)).cpu().numpy()
        path = args.output_dir / f'{args.name}_{split}.npz'
        save_predictions(path, [record['filename'] for record in records], composition=logits)
        print(f'{split}: {len(records)} images saved to {path}', flush=True)


if __name__ == '__main__':
    main()
