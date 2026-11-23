r"""Save Baseline 1 predictions for evaluate.py from one crop model and one disease model.

While developing, predict the validation split and score it:
    python src\predict_independent.py --name baseline1 --crop <crop best.pth> --disease <disease best.pth>
    python evaluate.py run --name baseline1_dev --val predictions\baseline1_val.npz

For models trained with --pseudo-unseen, use --splits pseudo_val_seen pseudo_val_unseen and choose the
calibration gamma with calibrated_stacking.py --pseudo-unseen.

Once the models are retrained with --final-epochs, predict both test splits once:
    python src\predict_independent.py --name baseline1 --crop <crop final.pth> --disease <disease final.pth>
        --splits test_seen test_unseen
"""
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT  # noqa: E402
from evaluate import save_predictions  # noqa: E402
from preprocessing import DPDDataset, eval_transform  # noqa: E402
from pseudo_unseen import SPLITS as PSEUDO_SPLITS, records_for  # noqa: E402
from train_independent import build_model, predict_logits  # noqa: E402


def load_model(path, target, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['target'] != target:
        raise ValueError(f'{path} is a {checkpoint["target"]} model, not a {target} model.')
    model, _ = build_model(checkpoint['backbone'], checkpoint['num_classes'], pretrained=False)
    model.load_state_dict(checkpoint['model'])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Save Baseline 1 predictions for evaluate.py.')
    parser.add_argument('--name', required=True, help='Prefix of the prediction files.')
    parser.add_argument('--crop', required=True, type=Path, help='best.pth of the crop model.')
    parser.add_argument('--disease', required=True, type=Path, help='best.pth of the disease model.')
    parser.add_argument('--splits', nargs='+', default=['val'],
                        choices=['val', 'test_seen', 'test_unseen', *PSEUDO_SPLITS])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crop_model = load_model(args.crop, 'crop', device)
    disease_model = load_model(args.disease, 'disease', device)
    for split in args.splits:
        loader = DataLoader(DPDDataset(records_for(split), eval_transform()), batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        crop_logits, filenames = predict_logits(crop_model, loader, device)
        disease_logits, _ = predict_logits(disease_model, loader, device)
        path = args.output_dir / f'{args.name}_{split}.npz'
        save_predictions(path, filenames, crop=crop_logits, disease=disease_logits)
        print(f'{split}: {len(filenames)} images saved to {path}', flush=True)


if __name__ == '__main__':
    main()
