"""Train and evaluate the shared-backbone crop and disease baseline."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from dataset_access import load_labels
from evaluate import Taxonomy, evaluate_split, save_predictions
from preprocessing import balanced_sampler, build_datasets


class MultiTaskNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.crop = nn.Linear(features, 55)
        self.disease = nn.Linear(features, 175)

    def forward(self, images):
        features = self.backbone(images)
        return self.crop(features), self.disease(features)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    names, crops, diseases = [], [], []
    for batch in loader:
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            crop, disease = model(batch['image'].to(device, non_blocking=True))
        names.extend(batch['filename'])
        crops.append(crop.float().cpu())
        diseases.append(disease.float().cpu())
    return {'filenames': np.asarray(names), 'crop': torch.cat(crops).numpy(),
            'disease': torch.cat(diseases).numpy()}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final = args.final_full_epochs is not None
    datasets = build_datasets(use_validation=not final)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = balanced_sampler(datasets['train'].records, generator=generator)
    options = {'batch_size': args.batch_size, 'num_workers': args.workers,
               'pin_memory': device.type == 'cuda', 'persistent_workers': args.workers > 0}
    train_loader = DataLoader(datasets['train'], sampler=sampler, **options)
    val_loader = None if final else DataLoader(datasets['val'], **options)
    taxonomy = Taxonomy(load_labels())
    model = MultiTaskNet().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['output'] = str(args.output)
    config.update({'backbone': 'convnext_tiny', 'weights': 'IMAGENET1K_V1', 'image_size': 224,
                   'loss': 'crop CE + disease CE', 'label_smoothing': 0.1,
                   'sampler': 'composition_id square-root', 'device': str(device),
                   'train_images': len(datasets['train']), 'parameters': sum(p.numel() for p in model.parameters())})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    best, stale = -1.0, 0
    with (args.output / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow(['stage', 'epoch', 'learning_rate', 'train_loss', 'val_composition_top1_per_class', 'seconds'])
        for stage, maximum, lr in [('head', args.head_epochs, 1e-3), ('full', args.epochs, 1e-4)]:
            head_only = stage == 'head'
            for p in model.backbone.parameters():
                p.requires_grad = not head_only
            optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(maximum, 1))
            run_epochs = (args.final_head_epochs if head_only else args.final_full_epochs) if final else maximum
            for epoch in range(1, run_epochs + 1):
                start = time.perf_counter()
                model.train()
                if head_only:
                    model.backbone.eval()
                total = 0.0
                for batch in train_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    crop = batch['crop'].to(device, non_blocking=True)
                    disease = batch['disease'].to(device, non_blocking=True)
                    optimiser.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
                        crop_logits, disease_logits = model(images)
                        loss = loss_fn(crop_logits, crop) + loss_fn(disease_logits, disease)
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                    total += loss.item() * len(images)
                learning_rate = optimiser.param_groups[0]['lr']
                scheduler.step()
                score = None
                if not final:
                    metrics, _, _ = evaluate_split(predict(model, val_loader, device), datasets['val'].records, taxonomy)
                    score = metrics['composition']['top1']['per_class']
                    if score > best:
                        best, stale = score, 0
                        torch.save({'model': model.state_dict(), 'stage': stage, 'epoch': epoch,
                                    'validation_score': score}, args.output / 'best.pth')
                    else:
                        stale += 1
                seconds = time.perf_counter() - start
                history.writerow([stage, epoch, learning_rate, total / len(datasets['train']), score, seconds])
                file.flush()
                print(f'{stage} {epoch}/{run_epochs}: loss={total / len(datasets["train"]):.4f} '
                      f'val={score} time={seconds:.1f}s', flush=True)
                if not final and not head_only and stale >= args.patience:
                    break
    if final:
        torch.save({'model': model.state_dict(), 'stage': 'final', 'head_epochs': args.final_head_epochs,
                    'full_epochs': args.final_full_epochs}, args.output / 'final.pth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--batch-size', type=int, default=32)
    training.add_argument('--workers', type=int, default=4)
    training.add_argument('--head-epochs', type=int, default=3)
    training.add_argument('--epochs', type=int, default=30)
    training.add_argument('--patience', type=int, default=5)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--final-head-epochs', type=int, default=3)
    training.add_argument('--final-full-epochs', type=int)
    prediction = sub.add_parser('predict')
    prediction.add_argument('--checkpoint', type=Path, required=True)
    prediction.add_argument('--split', choices=['val', 'test_seen', 'test_unseen'], required=True)
    prediction.add_argument('--output', type=Path, required=True)
    prediction.add_argument('--batch-size', type=int, default=32)
    prediction.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)['model'])
    dataset = build_datasets()[args.split]
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    scores = predict(model, loader, device)
    save_predictions(args.output, **scores)
    print(f'Saved {len(dataset)} predictions to {args.output}')


if __name__ == '__main__':
    main()
