r"""Fine-tune a trained CSP model end to end, then apply WiSE-FT (weight-space ensembling).

Starting from a CSP checkpoint (frozen CLIP + learned crop and disease embeddings), this unfreezes CLIP's image
encoder and text transformer and trains them together with the embeddings: cross-entropy over the trained
compositions, as in train_csp.py. WiSE-FT (Wortsman et al., CVPR 2022) then blends the two models,
    weights = (1 - alpha) * starting CSP weights + alpha * fine-tuned weights,
keeping much of the starting model's robustness to new images while gaining the fine-tuned model's accuracy.

Choose alpha without test labels: fine-tune a CSP model trained without the pseudo-unseen pairs, and score every
alpha on the pseudo-unseen validation split (uncalibrated harmonic mean):
    python src\finetune_csp.py --pseudo-unseen --start <CSP --pseudo-unseen final.pt>
Then fine-tune the final CSP model on all 8,000 training images and save it at that alpha:
    python src\finetune_csp.py --start <CSP final.pt> --alpha <alpha>
predict_csp.py and CSPPredictor read the saved final.pt like any CSP checkpoint.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csp_features import encode_images, write_embeddings  # noqa: E402
from csp_model import TEMPLATE, backbone_slug, image_normalisation, load_csp  # noqa: E402
from dataset_access import PROJECT, load_labels, load_split  # noqa: E402
from evaluate import PRIMARY, Taxonomy  # noqa: E402
from preprocessing import DPDDataset, eval_transform, train_transform  # noqa: E402
from pseudo_unseen import load_pseudo_unseen_split, trained_pairs  # noqa: E402
from train_csp import validate_pseudo  # noqa: E402

FROZEN = ('clip.token_embedding', 'clip.logit_scale')  # the prompt template and CLIP's temperature stay fixed


def trainable(model):
    """CLIP's image encoder and text transformer layers, plus the crop and disease embeddings."""
    return {name: parameter for name, parameter in model.named_parameters() if not name.startswith(FROZEN)}


def snapshot(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in trainable(model).items()}


@torch.no_grad()
def blend(model, start, finetuned, alpha):
    """WiSE-FT: set the trainable weights to (1 - alpha) * start + alpha * fine-tuned."""
    for name, parameter in trainable(model).items():
        parameter.copy_(((1 - alpha) * start[name] + alpha * finetuned[name]).to(parameter.device))


def finetune(model, records, trained, args, device):
    parameters = trainable(model)
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    model.clip.set_grad_checkpointing(True)
    optimiser = torch.optim.AdamW([
        {'params': [p for name, p in parameters.items() if name.startswith('clip.')], 'lr': args.lr,
         'weight_decay': args.weight_decay},
        {'params': [model.disease_embeddings, model.crop_embeddings], 'lr': args.embedding_lr,
         'weight_decay': 1e-3}])
    mean, std = image_normalisation(model.clip)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(DPDDataset(records, train_transform(mean=mean, std=std)), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, generator=generator, num_workers=args.num_workers,
                        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    position = torch.full((len(model.all_pairs),), -1, dtype=torch.long)
    position[torch.as_tensor(trained)] = torch.arange(len(trained))
    trained_ids = torch.as_tensor(trained, device=device)

    for epoch in range(1, args.epochs + 1):
        start, total_loss = time.time(), 0.0
        model.train()
        for batch in loader:
            targets = position[batch['composition']].to(device)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                features = model.clip.encode_image(batch['image'].to(device, non_blocking=True))
                loss = F.cross_entropy(model(features, trained_ids).float(), targets)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(targets)
        print(f'epoch {epoch}/{args.epochs} loss {total_loss / (len(loader) * args.batch_size):.4f} | '
              f'{time.time() - start:.0f}s', flush=True)
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    model.clip.set_grad_checkpointing(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune CSP end to end and apply WiSE-FT.')
    parser.add_argument('--start', required=True, type=Path, help='CSP checkpoint (best.pt or final.pt)')
    parser.add_argument('--pseudo-unseen', action='store_true',
                        help='train without the pseudo-unseen pairs and score every alpha on them')
    parser.add_argument('--alpha', type=float, help='WiSE-FT mixing weight for the saved final model')
    parser.add_argument('--alphas', type=float, nargs='+', default=[round(0.1 * i, 1) for i in range(11)])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate of the CLIP layers')
    parser.add_argument('--embedding-lr', type=float, default=1e-3, help='learning rate of the crop/disease words')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='weight decay of the CLIP layers')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs')
    args = parser.parse_args()
    if args.pseudo_unseen == (args.alpha is not None):
        parser.error('use --pseudo-unseen to choose alpha, or --alpha for the final model, not both')
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    taxonomy = Taxonomy(load_labels())
    model = load_csp(args.start, device)
    if model.finetuned:
        raise ValueError('--start must be a frozen-CLIP CSP checkpoint from train_csp.py.')
    slug = backbone_slug(*model.backbone)
    run_dir = args.output_dir / f'wiseft_{slug}{"_pseudo" if args.pseudo_unseen else "_final"}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    if args.pseudo_unseen:
        split = load_pseudo_unseen_split()
        records, trained = split['train'], trained_pairs(taxonomy)
    else:
        records, trained = load_split('train'), taxonomy.seen
    print(f'Fine-tuning {slug} CSP on {len(records)} images of {len(trained)} pairs, {args.epochs} epochs, '
          f'on {device}', flush=True)
    start = snapshot(model)
    finetune(model, records, trained, args, device)
    finetuned = snapshot(model)
    mean, std = image_normalisation(model.clip)
    transform = eval_transform(mean=mean, std=std)

    if args.pseudo_unseen:
        seen_mask = np.zeros(taxonomy.size('composition'))
        seen_mask[trained] = 1
        targets = {name: np.array([record['composition_id'] for record in split[name]])
                   for name in ('pseudo_val_seen', 'pseudo_val_unseen')}
        rows = []
        for alpha in args.alphas:
            blend(model, start, finetuned, alpha)
            data = {name: (encode_images(model.clip, split[name], transform, device).to(device), targets[name])
                    for name in targets}
            metrics = validate_pseudo(model, data, seen_mask)
            uncalibrated, calibrated = metrics['uncalibrated'], metrics['calibrated']
            rows.append([alpha, uncalibrated['pseudo_val_seen']['top1'][PRIMARY],
                         uncalibrated['pseudo_val_unseen']['top1'][PRIMARY],
                         uncalibrated['harmonic_mean']['top1'][PRIMARY], calibrated['gamma'],
                         calibrated['harmonic_mean']['top1'][PRIMARY]])
            print(f'alpha {alpha:.1f} | gamma 0: seen {rows[-1][1]:.3f} unseen {rows[-1][2]:.3f} '
                  f'H {rows[-1][3]:.3f} | best gamma {rows[-1][4]:.2f}: H {rows[-1][5]:.3f}', flush=True)
        with (run_dir / 'alpha_sweep.csv').open('w', encoding='utf-8', newline='') as target:
            writer = csv.writer(target, lineterminator='\n')
            writer.writerow(['alpha', 'seen_top1', 'unseen_top1', 'h_top1', 'best_gamma', 'h_top1_calibrated'])
            writer.writerows([[f'{value:.4f}' for value in row] for row in rows])
        best = max(rows, key=lambda row: (row[3], row[1]))
        print(f'Best alpha {best[0]:.1f} (uncalibrated H {best[3]:.3f}). Saved {run_dir / "alpha_sweep.csv"}')
        return

    blend(model, start, finetuned, args.alpha)
    torch.save({**model.soft_state(), 'clip_model': model.backbone[0], 'clip_weights': model.backbone[1],
                'template': TEMPLATE, 'dropout': model.dropout.p, 'alpha': args.alpha,
                'clip_state': {name[len('clip.'):]: value.half().cpu() for name, value in snapshot(model).items()
                               if name.startswith('clip.')}}, run_dir / 'final.pt')
    train = load_split('train')
    write_embeddings(run_dir / 'train_embeddings.npz', train, encode_images(model.clip, train, transform, device))
    print(f'Saved {run_dir / "final.pt"} (alpha {args.alpha}) and its training-image embeddings for the UI')


if __name__ == '__main__':
    main()
