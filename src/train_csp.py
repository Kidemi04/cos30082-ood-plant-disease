r"""Train CSP soft prompts (csp_model.py) on cached, frozen CLIP image features.

    python src\csp_features.py                   # once: cache CLIP features
    python src\train_csp.py                      # select the epoch on the shared validation split
    python src\train_csp.py --final-epochs N     # retrain on all 8,000 training images, stop at epoch N

Training is cross-entropy over the 323 seen compositions; validation ranks all 333, like the test. Test images
are never used here. Each run writes config.json, history.csv and best.pt (or final.pt) to outputs/.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csp_features import FEATURES, FeatureCache  # noqa: E402
from csp_model import CSP, TEMPLATE, load_clip  # noqa: E402
from dataset_access import PROJECT, load_labels, load_split, load_validation_split  # noqa: E402
from evaluate import TOP_K, Taxonomy, accuracy, top_k_hits  # noqa: E402


def validate(model, features, targets):
    """Top-1 and Top-5 composition accuracy over all 333 compositions, measured as in evaluate.py."""
    model.eval()
    logits = model.all_logits(features).cpu().numpy()
    return {f'top{k}': accuracy(top_k_hits(logits, targets, k), targets) for k in TOP_K}


def parse_args():
    parser = argparse.ArgumentParser(description='Train CSP soft prompts on frozen CLIP features.')
    parser.add_argument('--epochs', type=int, default=20, help='maximum epochs, also the cosine schedule length')
    parser.add_argument('--final-epochs', type=int,
                        help='retrain on train + val without validation and stop after this epoch')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout on the disease and crop embeddings')
    parser.add_argument('--no-augmented-views', action='store_true', help='train on the clean view only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--features', type=Path, default=FEATURES)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs')
    parser.add_argument('--tag', help='suffix for the run folder, e.g. lr1e-3 for an ablation')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final = args.final_epochs is not None

    train, val = (load_split('train'), None) if final else load_validation_split()
    cache = FeatureCache(args.features)
    train_views = cache.features(train, augmented=not args.no_augmented_views)
    taxonomy = Taxonomy(load_labels())
    compositions = np.array([record['composition_id'] for record in train])
    targets = np.searchsorted(taxonomy.seen, compositions)
    if not (taxonomy.seen[targets.clip(max=len(taxonomy.seen) - 1)] == compositions).all():
        raise ValueError('Training images must all belong to seen compositions.')
    targets = torch.as_tensor(targets)
    seen = torch.as_tensor(taxonomy.seen, device=device)
    if not final:
        val_features = cache.features(val)[0].to(device)
        val_targets = np.array([record['composition_id'] for record in val])

    clip_model, tokenizer = load_clip(device, cache.clip_model, cache.clip_weights)
    model = CSP(clip_model, tokenizer, taxonomy, args.dropout)
    optimiser = torch.optim.AdamW([model.disease_embeddings, model.crop_embeddings], lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    run_name = f'csp_{cache.clip_model}_{cache.clip_weights}'
    if final:
        run_name += '_final'
    if args.tag:
        run_name += f'_{args.tag}'
    run_dir = args.output_dir / f'{run_name}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    def checkpoint(epoch, metrics):
        return {**model.soft_state(), 'clip_model': cache.clip_model, 'clip_weights': cache.clip_weights,
                'template': TEMPLATE, 'dropout': args.dropout, 'epoch': epoch, 'metrics': metrics}

    print(f'{run_name}: {len(train)} training images, {train_views.shape[0]} view(s) each, on {device}', flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    best_score, stale_epochs = (-1.0, -1.0), 0
    run_epochs = args.final_epochs if final else args.epochs

    with (run_dir / 'history.csv').open('w', encoding='utf-8', newline='') as history_file:
        history = csv.writer(history_file, lineterminator='\n')
        history.writerow(['epoch', 'lr', 'train_loss', 'val_top1_per_class', 'val_top5_per_class',
                          'val_top1_per_image', 'val_top5_per_image', 'seconds'])
        for epoch in range(1, run_epochs + 1):
            start = time.time()
            current_lr = optimiser.param_groups[0]['lr']
            model.train()
            order = torch.randperm(len(train), generator=generator)
            views = torch.randint(len(train_views), (len(train),), generator=generator)
            total_loss = 0.0
            for batch in order.split(args.batch_size):
                features = train_views[views[batch], batch].to(device)
                loss = F.cross_entropy(model(features, seen), targets[batch].to(device))
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(batch)
            scheduler.step()
            train_loss = total_loss / len(train)

            if final:
                history.writerow([epoch, f'{current_lr:.2e}', f'{train_loss:.4f}', '', '', '', '',
                                  f'{time.time() - start:.0f}'])
                print(f'epoch {epoch}/{run_epochs} loss {train_loss:.4f}', flush=True)
                continue
            metrics = validate(model, val_features, val_targets)
            per_class = metrics['top1']['per_class'], metrics['top5']['per_class']
            per_image = metrics['top1']['per_image'], metrics['top5']['per_image']
            history.writerow([epoch, f'{current_lr:.2e}', f'{train_loss:.4f}',
                              *[f'{value:.4f}' for value in (*per_class, *per_image)], f'{time.time() - start:.0f}'])
            history_file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {train_loss:.4f} | per-class {per_class[0]:.3f}/{per_class[1]:.3f}'
                  f' | per-image {per_image[0]:.3f}/{per_image[1]:.3f}', flush=True)

            score = (per_class[0], per_image[0])
            if score > best_score:
                best_score, stale_epochs = score, 0
                torch.save(checkpoint(epoch, metrics), run_dir / 'best.pt')
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f'Stopped early: no improvement for {args.patience} epochs.', flush=True)
                    break

    if final:
        torch.save(checkpoint(args.final_epochs, None), run_dir / 'final.pt')
        print(f'Saved to {run_dir / "final.pt"}')
    else:
        print(f'Best per-class Top-1 {best_score[0]:.3f}. Saved to {run_dir}')


if __name__ == '__main__':
    main()
