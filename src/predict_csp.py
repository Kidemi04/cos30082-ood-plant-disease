r"""Save CSP predictions for evaluate.py, and CSPPredictor for the UI.

For a model trained with --pseudo-unseen, use --splits pseudo_val_seen pseudo_val_unseen and choose gamma with
calibrated_stacking.py --pseudo-unseen.

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
from csp_features import FeatureCache, encode_images, feature_path, load_training_embeddings, \
    nearest_training_images, train_embeddings_path  # noqa: E402
from csp_model import image_normalisation, load_csp  # noqa: E402
from dataset_access import PROJECT, load_labels  # noqa: E402
from evaluate import Taxonomy, save_predictions  # noqa: E402
from preprocessing import IMAGE_SIZE, RESIZE_RATIO, eval_transform, load_image  # noqa: E402
from pseudo_unseen import SPLITS as PSEUDO_SPLITS, records_for  # noqa: E402


class CSPPredictor:
    """A trained CSP model plus the training-image embeddings, for predictions with nearest neighbours."""

    def __init__(self, checkpoint, embeddings=None, gamma=0.0, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = load_csp(checkpoint, self.device)
        mean, std = image_normalisation(self.model.clip)
        self.transform = eval_transform(mean=mean, std=std)
        if embeddings is None:  # a fine-tuned model saves embeddings from its own image encoder beside final.pt
            embeddings = Path(checkpoint).parent / 'train_embeddings.npz' if self.model.finetuned \
                else train_embeddings_path(*self.model.backbone)
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
    parser.add_argument('--splits', nargs='+', default=['val'],
                        choices=['val', 'test_seen', 'test_unseen', *PSEUDO_SPLITS])
    parser.add_argument('--features', type=Path, help='feature cache (default: the checkpoint backbone\'s cache)')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_csp(args.checkpoint, device)
    if model.finetuned:  # fine-tuned image encoder: encode the images again instead of using the cache
        mean, std = image_normalisation(model.clip)
        transform = eval_transform(mean=mean, std=std)
    else:
        cache = FeatureCache(args.features or feature_path(*model.backbone))
    for split in args.splits:
        records = records_for(split)
        features = encode_images(model.clip, records, transform, device) if model.finetuned \
            else cache.features(records)[0]
        logits = model.all_logits(features.to(device)).cpu().numpy()
        path = args.output_dir / f'{args.name}_{split}.npz'
        save_predictions(path, [record['filename'] for record in records], composition=logits)
        print(f'{split}: {len(records)} images saved to {path}', flush=True)


if __name__ == '__main__':
    main()
