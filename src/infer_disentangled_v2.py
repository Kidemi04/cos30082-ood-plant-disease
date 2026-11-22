"""Standalone inference wrapper for Disentangled ConvNeXt-Tiny v2 (for UI integration).

Usage in Streamlit / Gradio / Python UI:
    from src.infer_disentangled_v2 import DisentangledV2Predictor

    predictor = DisentangledV2Predictor("runs/v2_final_winner/final.pth")
    result = predictor.predict_image(pil_image, top_k=5, calibrated=True)
    # result["top_compositions"][0] -> {
    #   "rank": 1, "composition_id": ..., "composition": "tomato_early_blight",
    #   "crop": "tomato", "disease": "early_blight", "is_unseen": True,
    #   "probability": 0.72, "score": -2.14
    # }

CLI quick test:
    python src/infer_disentangled_v2.py --checkpoint runs/v2_final_winner/final.pth --image <path/to/leaf.jpg>
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy, level_scores  # noqa: E402
from gzsl import calibrated as apply_calibration, seen_columns  # noqa: E402
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, eval_transform  # noqa: E402
from train_disentangled_v2 import DisentangledNetV2  # noqa: E402

DEFAULT_GAMMA = 5.8


class DisentangledV2Predictor:
    """Loads `v2_final_winner/final.pth` and runs calibrated GZSL composition inference."""

    def __init__(self, checkpoint_path, device=None, gamma=DEFAULT_GAMMA, label_map_path=None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint_path}')

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.gamma = float(gamma)

        labels = (
            json.loads(Path(label_map_path).read_text(encoding='utf-8'))
            if label_map_path is not None
            else load_labels()
        )
        self.taxonomy = Taxonomy(labels)
        self.seen_mask = seen_columns(self.taxonomy, 'final')
        self.unseen_set = set(int(i) for i in self.taxonomy.unseen)

        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

        self.image_size = int(cfg.get('image_size', 224))
        self.transform = eval_transform(self.image_size, IMAGENET_MEAN, IMAGENET_STD)

        self.model = DisentangledNetV2(
            embed_dim=int(cfg.get('embed_dim', 512)),
            head_type=str(cfg.get('head_type', 'linear')),
            use_grl=float(cfg.get('grl_lambda', 0.02)) > 0.0,
            detach_grl_backbone=not bool(cfg.get('no_detach_grl', False)),
            use_mixstyle=bool(cfg.get('mixstyle', False)),
            dropout=float(cfg.get('dropout', 0.1)),
            pretrained=False,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

    def _to_pil(self, image):
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        raise TypeError(f'Unsupported image input type: {type(image)}')

    @torch.no_grad()
    def predict_image(self, image, top_k=5, calibrated=True, gamma=None):
        """Predict top-k (crop, disease) compositions for a single leaf image."""
        pil_img = self._to_pil(image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        out = self.model(tensor, return_features=False)

        crop_logits = out['crop'].cpu().numpy()
        disease_logits = out['disease'].cpu().numpy()
        scores = level_scores({'crop': crop_logits, 'disease': disease_logits}, self.taxonomy)

        comp_scores = scores['composition']
        used_gamma = self.gamma if gamma is None else float(gamma)
        if calibrated:
            comp_scores = apply_calibration(comp_scores, self.seen_mask, used_gamma)

        # Softmax over the 333 candidate compositions for human-readable UI probabilities
        row = comp_scores[0]
        shifted = row - row.max()
        comp_probs = np.exp(shifted)
        comp_probs /= comp_probs.sum()

        top_indices = np.argsort(row)[::-1][: int(top_k)]
        top_compositions = []
        for rank, cid in enumerate(top_indices, start=1):
            cid = int(cid)
            crop_id = int(self.taxonomy.groups['crop'][cid])
            dis_id = int(self.taxonomy.groups['disease'][cid])
            top_compositions.append({
                'rank': rank,
                'composition_id': cid,
                'composition': self.taxonomy.names['composition'][cid],
                'crop': self.taxonomy.names['crop'][crop_id],
                'disease': self.taxonomy.names['disease'][dis_id],
                'is_unseen': cid in self.unseen_set,
                'probability': float(comp_probs[cid]),
                'score': float(row[cid]),
            })

        crop_probs = np.exp(scores['crop'][0])
        dis_probs = np.exp(scores['disease'][0])
        best_crop_id = int(np.argmax(crop_probs))
        best_dis_id = int(np.argmax(dis_probs))

        return {
            'top_compositions': top_compositions,
            'top_crop': {
                'crop_id': best_crop_id,
                'name': self.taxonomy.names['crop'][best_crop_id],
                'probability': float(crop_probs[best_crop_id]),
            },
            'top_disease': {
                'disease_id': best_dis_id,
                'name': self.taxonomy.names['disease'][best_dis_id],
                'probability': float(dis_probs[best_dis_id]),
            },
            'calibrated': bool(calibrated),
            'gamma': used_gamma if calibrated else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description='Run Disentangled ConvNeXt-Tiny v2 inference on a leaf image.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to v2_final_winner/final.pth')
    parser.add_argument('--image', type=Path, required=True, help='Path to input leaf image')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--raw', action='store_true', help='Disable GZSL calibration (gamma=0)')
    args = parser.parse_args()

    predictor = DisentangledV2Predictor(args.checkpoint, gamma=args.gamma)
    result = predictor.predict_image(args.image, top_k=args.top_k, calibrated=not args.raw)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
