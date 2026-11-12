r"""Baselines 1 and 2 under the pseudo-unseen protocol, so they can be calibrated exactly like the proposed model.

Modes
  multitask : Baseline 2, one backbone with a crop head (55) and a disease head (175)
  crop      : Baseline 1 crop network
  disease   : Baseline 1 disease network (run crop and disease, then `gzsl.py merge` their predictions)

Recipe (copied from the team baselines): torchvision convnext_tiny IMAGENET1K_V1 (team backbone, full-model rate
  1e-4) or vit_b_16 IMAGENET1K_V1 (3e-5), shared 224 px preprocessing (ImageNet statistics), square-root sampling
  by composition (multitask) or by the target (crop / disease), AdamW wd 0.05, label smoothing 0.1, AMP, batch 32;
  3 head-only epochs at 1e-3, then up to 30 full-model epochs with a cosine schedule and patience 5, selecting by
  seen validation per-class Top-1.

  python src\train_baseline.py --mode multitask --stage select --out runs\b2_select
  python src\train_baseline.py --mode multitask --stage final --final-epochs 15 --out runs\b2_final
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, vit_b_16

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy, save_predictions  # noqa: E402
from gzsl import eval_loader, make_datasets, predict, seed_everything, val_score  # noqa: E402
from preprocessing import balanced_sampler  # noqa: E402

HEADS = {'multitask': ('crop', 'disease'), 'crop': ('crop',), 'disease': ('disease',)}
SIZES = {'crop': 55, 'disease': 175}


LEARNING_RATES = {'convnext_tiny': 1e-4, 'vit_b_16': 3e-5}


def save_scores(path, predictions):
    """Save predictions; a single-head Baseline 1 network writes a partial file for `gzsl.py merge`.

    evaluate.save_predictions only accepts complete files (composition, or crop and disease), so a crop-only or
    disease-only file is written directly in the same format.
    """
    if 'composition' in predictions or {'crop', 'disease'} <= predictions.keys():
        save_predictions(path, **predictions)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, filenames=np.asarray([str(n) for n in predictions['filenames']]),
                        **{k: np.asarray(v, dtype=np.float32) for k, v in predictions.items() if k != 'filenames'})


class Heads(nn.Module):
    def __init__(self, backbone, heads, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        if backbone == 'convnext_tiny':
            self.backbone = convnext_tiny(weights=weights)
            width = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
        else:
            self.backbone = vit_b_16(weights=weights)
            width = 768
            self.backbone.heads = nn.Identity()
        self.heads = nn.ModuleDict({name: nn.Linear(width, SIZES[name]) for name in heads})

    def forward(self, images):
        features = self.backbone(images)
        return {name: head(features) for name, head in self.heads.items()}


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    heads = HEADS[args.mode]
    level = 'composition' if args.mode == 'multitask' else args.mode
    data = make_datasets(args.stage)
    key = 'composition_id' if args.mode == 'multitask' else args.mode + '_id'
    sampler = balanced_sampler(data['train'].records, key=key, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(data['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    taxonomy = Taxonomy(load_labels())
    model = Heads(args.backbone, heads).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    args.out.mkdir(parents=True, exist_ok=True)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'weights': f'torchvision {args.backbone} IMAGENET1K_V1', 'image_size': 224, 'sampler': f'{key} square-root',
              'loss': ' + '.join(f'{h} CE' for h in heads), 'label_smoothing': 0.1,
              'train_images': len(data['train']), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    val_loader = eval_loader(data['val_seen'], args.batch_size, args.workers) if args.stage == 'select' else None
    best, best_epoch, stale = -1.0, 0, 0
    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow(['stage', 'epoch', 'lr', 'train_loss', f'val_{level}_top1_per_class', 'seconds'])
        for stage, epochs, lr in (('head', args.head_epochs, args.head_lr), ('full', args.epochs, args.lr)):
            head_only = stage == 'head'
            for p in model.backbone.parameters():
                p.requires_grad = not head_only
            optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                          weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
            run_epochs = args.final_epochs if args.stage == 'final' and not head_only else epochs
            for epoch in range(1, run_epochs + 1):
                start = time.perf_counter()
                model.train()
                if head_only:
                    model.backbone.eval()
                total, count = 0.0, 0
                for batch in train_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    optimiser.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda'):
                        logits = model(images)
                        loss = sum(loss_fn(logits[h], batch[h].to(device, non_blocking=True)) for h in heads)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss at {stage} epoch {epoch}; lower the learning rate.')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    total += loss.item() * len(images)
                    count += len(images)
                lr_now = optimiser.param_groups[0]['lr']
                scheduler.step()
                score = None
                if args.stage == 'select':
                    preds = predict(model, val_loader, device)
                    score = val_score(preds, data['val_seen'].records, taxonomy, level)
                    if score > best:
                        best, best_epoch, stale = score, (stage, epoch), 0
                        torch.save({'model': model.state_dict(), 'stage': stage, 'epoch': epoch, 'score': score},
                                   args.out / 'best.pth')
                    elif not head_only:
                        stale += 1
                seconds = time.perf_counter() - start
                history.writerow([stage, epoch, lr_now, total / count, score, round(seconds, 1)])
                file.flush()
                print(f'{stage} {epoch}/{run_epochs} loss {total / count:.4f} val {score} {seconds:.0f}s', flush=True)
                if args.stage == 'select' and not head_only and stale >= args.patience:
                    break
    if args.stage == 'select':
        model.load_state_dict(torch.load(args.out / 'best.pth', map_location=device, weights_only=True)['model'])
        save_scores(args.out / 'pseudo.npz', predict(model, eval_loader(data['pseudo'], args.batch_size,
                                                                         args.workers), device))
        summary = {'best_stage_epoch': best_epoch, 'best_val_top1_per_class': best}
    else:
        torch.save({'model': model.state_dict(), 'stage': 'final', 'epochs': args.final_epochs}, args.out / 'final.pth')
        for split in ('test_seen', 'test_unseen'):
            save_scores(args.out / f'{split}.npz',
                        predict(model, eval_loader(data[split], args.batch_size, args.workers), device))
        summary = {'final_full_epochs': args.final_epochs}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backbone', choices=LEARNING_RATES, default='convnext_tiny')
    parser.add_argument('--mode', choices=HEADS, required=True)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--final-epochs', type=int, help='Full-model epochs for the final stage (best select epoch).')
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, help='Full-model learning rate; the default depends on the backbone.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage == 'final' and not args.final_epochs:
        parser.error('--final-epochs is required for the final stage.')
    args.lr = args.lr or LEARNING_RATES[args.backbone]
    run(args)


if __name__ == '__main__':
    main()
