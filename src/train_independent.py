r"""Train one Baseline 1 model: an independent crop or disease classifier.

Run it once per target, from the repository folder:
    python src\train_independent.py --target crop
    python src\train_independent.py --target disease

Trains on the shared train split and selects the best epoch by per-class Top-1 on the shared
validation split. Test images are never used here. Each run writes config.json, history.csv and
best.pth to its own folder in outputs/. predict_independent.py turns two runs into predictions.

Once the best epoch is known, retrain on the full official train split (train + val) and stop at
that full-model epoch. The learning-rate schedule is unchanged, so it matches the selection run:
    python src\train_independent.py --target crop --balanced --final-epochs 10
This writes final.pth instead of best.pth.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT  # noqa: E402
from evaluate import TOP_K, accuracy, top_k_hits  # noqa: E402
from preprocessing import balanced_sampler, build_datasets  # noqa: E402

NUM_CLASSES = {'crop': 55, 'disease': 175}


def build_model(backbone, num_classes, pretrained=True):
    """Return (model, head): the backbone with a new classifier layer."""
    if backbone == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model, model.classifier[2]
    if backbone == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, model.fc
    if backbone == 'swin_t':
        model = models.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model, model.head
    raise ValueError(f'Unknown backbone: {backbone}')


def train_one_epoch(model, head, head_only, target, loader, criterion, optimiser, scaler, device):
    model.train()
    if head_only:
        # The frozen backbone must not apply stochastic depth.
        model.eval()
        head.train()
    total_loss = 0.0
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch[target].to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Return (logits, filenames) for a whole dataset, in loader order."""
    model.eval()
    logits, filenames = [], []
    for batch in loader:
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            logits.append(model(batch['image'].to(device, non_blocking=True)).float().cpu())
        filenames += batch['filename']
    return torch.cat(logits).numpy(), filenames


def validate(model, loader, target, device):
    """Return Top-1 and Top-5 accuracy, measured the same way as evaluate.py."""
    logits, _ = predict_logits(model, loader, device)
    labels = np.array([record[target + '_id'] for record in loader.dataset.records])
    return {f'top{k}': accuracy(top_k_hits(logits, labels, k), labels) for k in TOP_K}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Baseline 1 crop or disease model.')
    parser.add_argument('--target', required=True, choices=sorted(NUM_CLASSES))
    parser.add_argument('--backbone', default='convnext_tiny', choices=['convnext_tiny', 'resnet50', 'swin_t'])
    parser.add_argument('--balanced', action='store_true', help='draw rare target classes more often')
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30, help='maximum full-model epochs')
    parser.add_argument('--final-epochs', type=int,
                        help='retrain on train + val without validation and stop after this full-model epoch')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs')
    parser.add_argument('--tag', help='suffix for the run folder, e.g. lr3e-4 for an ablation')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = NUM_CLASSES[args.target]
    final = args.final_epochs is not None

    datasets = build_datasets(use_validation=not final)
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers,
                   'pin_memory': device.type == 'cuda', 'persistent_workers': args.num_workers > 0}
    if args.balanced:
        sampler = balanced_sampler(datasets['train'].records, key=args.target + '_id', generator=generator)
        train_loader = DataLoader(datasets['train'], sampler=sampler, **loader_args)
    else:
        train_loader = DataLoader(datasets['train'], shuffle=True, generator=generator, **loader_args)
    val_loader = None if final else DataLoader(datasets['val'], **loader_args)

    model, head = build_model(args.backbone, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    run_name = f'baseline1_{args.target}_{args.backbone}'
    if args.balanced:
        run_name += '_balanced'
    if final:
        run_name += '_final'
    if args.tag:
        run_name += f'_{args.tag}'
    run_dir = args.output_dir / f'{run_name}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    print(f'{run_name}: {len(datasets["train"])} training images on {device}', flush=True)
    best_score = (-1.0, -1.0)

    with (run_dir / 'history.csv').open('w', encoding='utf-8', newline='') as history_file:
        history = csv.writer(history_file, lineterminator='\n')
        history.writerow(['stage', 'epoch', 'lr', 'train_loss', 'val_top1_per_class', 'val_top5_per_class',
                          'val_top1_per_image', 'val_top5_per_image', 'seconds'])

        for stage, stage_epochs, lr in (('head', args.head_epochs, args.head_lr), ('full', args.epochs, args.lr)):
            head_only = stage == 'head'
            for parameter in model.parameters():
                parameter.requires_grad = not head_only
            for parameter in head.parameters():
                parameter.requires_grad = True
            optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                          weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(stage_epochs, 1))
            stale_epochs = 0
            run_epochs = args.final_epochs if final and not head_only else stage_epochs

            for epoch in range(1, run_epochs + 1):
                start = time.time()
                current_lr = optimiser.param_groups[0]['lr']
                train_loss = train_one_epoch(model, head, head_only, args.target, train_loader, criterion,
                                             optimiser, scaler, device)
                scheduler.step()
                if final:
                    seconds = time.time() - start
                    history.writerow([stage, epoch, f'{current_lr:.2e}', f'{train_loss:.4f}', '', '', '', '',
                                      f'{seconds:.0f}'])
                    history_file.flush()
                    print(f'{stage} {epoch}/{run_epochs} loss {train_loss:.4f} | {seconds:.0f}s', flush=True)
                    continue
                metrics = validate(model, val_loader, args.target, device)
                seconds = time.time() - start
                per_class = metrics['top1']['per_class'], metrics['top5']['per_class']
                per_image = metrics['top1']['per_image'], metrics['top5']['per_image']

                history.writerow([stage, epoch, f'{current_lr:.2e}', f'{train_loss:.4f}',
                                  *[f'{value:.4f}' for value in (*per_class, *per_image)], f'{seconds:.0f}'])
                history_file.flush()
                print(f'{stage} {epoch}/{stage_epochs} loss {train_loss:.4f} | '
                      f'per-class {per_class[0]:.3f}/{per_class[1]:.3f} | '
                      f'per-image {per_image[0]:.3f}/{per_image[1]:.3f} | {seconds:.0f}s', flush=True)

                score = (per_class[0], per_image[0])
                if score > best_score:
                    best_score = score
                    stale_epochs = 0
                    torch.save({'model': model.state_dict(), 'target': args.target, 'backbone': args.backbone,
                                'num_classes': num_classes, 'stage': stage, 'epoch': epoch, 'metrics': metrics},
                               run_dir / 'best.pth')
                else:
                    stale_epochs += 1
                if not head_only and stale_epochs >= args.patience:
                    print(f'Stopped early: no improvement for {args.patience} epochs.', flush=True)
                    break

    if final:
        torch.save({'model': model.state_dict(), 'target': args.target, 'backbone': args.backbone,
                    'num_classes': num_classes, 'stage': 'full', 'epoch': args.final_epochs, 'metrics': None},
                   run_dir / 'final.pth')
    if device.type == 'cuda':
        print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
    if final:
        print(f'Saved to {run_dir / "final.pth"}')
    else:
        print(f'Best per-class Top-1 {best_score[0]:.3f}. Saved to {run_dir}')


if __name__ == '__main__':
    main()
