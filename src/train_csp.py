r"""Train CSP soft prompts (csp_model.py) on cached, frozen CLIP image features.

    python src\csp_features.py                   # once per backbone: cache CLIP features (--backbone bioclip)
    python src\train_csp.py                      # select the epoch on the shared validation split
    python src\train_csp.py --final-epochs N     # retrain on all 8,000 training images, stop at epoch N

With --pseudo-unseen, the 10 pseudo-unseen pairs (pseudo_unseen.py) are left out of training and every epoch is
scored on seen and unseen validation images, uncalibrated and at the gamma with the best harmonic mean. Without
--final-epochs it then selects the epoch by that calibrated harmonic mean.

Training is cross-entropy over the trained compositions; validation ranks all 333, like the test. Test images are
never used here. Each run writes config.json, history.csv and best.pt (or final.pt) to outputs/.
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
from calibrated_stacking import sweep  # noqa: E402
from csp_features import FeatureCache, feature_path  # noqa: E402
from csp_model import BACKBONES, CSP, TEMPLATE, backbone_slug, load_clip  # noqa: E402
from dataset_access import PROJECT, load_labels, load_split, load_validation_split  # noqa: E402
from evaluate import PRIMARY, TOP_K, Taxonomy, accuracy, top_k_hits  # noqa: E402
from pseudo_unseen import load_pseudo_unseen_split, trained_pairs  # noqa: E402

HISTORY = ['val_top1_per_class', 'val_top5_per_class', 'val_top1_per_image', 'val_top5_per_image']
PSEUDO_HISTORY = ['seen_top1', 'unseen_top1', 'h_top1', 'best_gamma', 'seen_top1_calibrated',
                  'unseen_top1_calibrated', 'h_top1_calibrated']  # per-class


def validate(model, features, targets):
    """Top-1 and Top-5 composition accuracy over all 333 compositions, measured as in evaluate.py."""
    model.eval()
    logits = model.all_logits(features).cpu().numpy()
    return {f'top{k}': accuracy(top_k_hits(logits, targets, k), targets) for k in TOP_K}


def validate_pseudo(model, data, seen_mask):
    """Seen and unseen accuracy with gamma = 0 and with the gamma that maximises the harmonic mean."""
    model.eval()
    scores = {split: (model.all_logits(features).log_softmax(1).cpu().numpy(), targets, None)
              for split, (features, targets) in data.items()}
    _, uncalibrated, best = sweep(scores, seen_mask, steps=200)
    return {'uncalibrated': uncalibrated, 'calibrated': best}


def pseudo_row(metrics):
    uncalibrated, calibrated = metrics['uncalibrated'], metrics['calibrated']
    value = lambda point, key: point[key]['top1'][PRIMARY]  # noqa: E731
    return [value(uncalibrated, 'pseudo_val_seen'), value(uncalibrated, 'pseudo_val_unseen'),
            uncalibrated['harmonic_mean']['top1'][PRIMARY], calibrated['gamma'],
            value(calibrated, 'pseudo_val_seen'), value(calibrated, 'pseudo_val_unseen'),
            calibrated['harmonic_mean']['top1'][PRIMARY]]


def parse_args():
    parser = argparse.ArgumentParser(description='Train CSP soft prompts on frozen CLIP features.')
    parser.add_argument('--epochs', type=int, default=20, help='maximum epochs, also the cosine schedule length')
    parser.add_argument('--final-epochs', type=int,
                        help='train for exactly this many epochs without epoch selection (all 8,000 images '
                             'unless --pseudo-unseen)')
    parser.add_argument('--pseudo-unseen', action='store_true', help='hold out the pseudo-unseen pairs')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout on the disease and crop embeddings')
    parser.add_argument('--no-augmented-views', action='store_true', help='train on the clean view only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone', default='openai', choices=sorted(BACKBONES))
    parser.add_argument('--features', type=Path, help='feature cache (default: the backbone\'s cache)')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs')
    parser.add_argument('--tag', help='suffix for the run folder, e.g. lr1e-3 for an ablation')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final = args.final_epochs is not None
    taxonomy = Taxonomy(load_labels())
    cache = FeatureCache(args.features or feature_path(*BACKBONES[args.backbone]))
    targets_of = lambda records: np.array([record['composition_id'] for record in records])  # noqa: E731

    if args.pseudo_unseen:
        split = load_pseudo_unseen_split()
        train, trained = split['train'], trained_pairs(taxonomy)
        pseudo_data = {name: (cache.features(split[name])[0].to(device), targets_of(split[name]))
                       for name in ('pseudo_val_seen', 'pseudo_val_unseen')}
        seen_mask = np.zeros(taxonomy.size('composition'))
        seen_mask[trained] = 1
    else:
        train, val = (load_split('train'), None) if final else load_validation_split()
        trained = taxonomy.seen
        if not final:
            val_features, val_targets = cache.features(val)[0].to(device), targets_of(val)
    validating = args.pseudo_unseen or not final

    train_views = cache.features(train, augmented=not args.no_augmented_views)
    compositions = targets_of(train)
    targets = np.searchsorted(trained, compositions)
    if not (trained[targets.clip(max=len(trained) - 1)] == compositions).all():
        raise ValueError('Training images must all belong to trained compositions.')
    targets = torch.as_tensor(targets)
    trained_ids = torch.as_tensor(trained, device=device)

    clip_model, tokenizer = load_clip(device, cache.clip_model, cache.clip_weights)
    model = CSP(clip_model, tokenizer, taxonomy, args.dropout)
    optimiser = torch.optim.AdamW([model.disease_embeddings, model.crop_embeddings], lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    run_name = f'csp_{backbone_slug(cache.clip_model, cache.clip_weights)}'
    if args.pseudo_unseen:
        run_name += '_pseudo'
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

    print(f'{run_name}: {len(train)} training images of {len(trained)} pairs, {train_views.shape[0]} view(s) '
          f'each, on {device}', flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    best_score, stale_epochs, metrics = (-1.0, -1.0), 0, None
    run_epochs = args.final_epochs if final else args.epochs

    with (run_dir / 'history.csv').open('w', encoding='utf-8', newline='') as history_file:
        history = csv.writer(history_file, lineterminator='\n')
        history.writerow(['epoch', 'lr', 'train_loss', *(PSEUDO_HISTORY if args.pseudo_unseen else HISTORY),
                          'seconds'])
        for epoch in range(1, run_epochs + 1):
            start = time.time()
            current_lr = optimiser.param_groups[0]['lr']
            model.train()
            order = torch.randperm(len(train), generator=generator)
            views = torch.randint(len(train_views), (len(train),), generator=generator)
            total_loss = 0.0
            for batch in order.split(args.batch_size):
                features = train_views[views[batch], batch].to(device)
                loss = F.cross_entropy(model(features, trained_ids), targets[batch].to(device))
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(batch)
            scheduler.step()
            train_loss = total_loss / len(train)

            if not validating:
                history.writerow([epoch, f'{current_lr:.2e}', f'{train_loss:.4f}', '', '', '', '',
                                  f'{time.time() - start:.0f}'])
                print(f'epoch {epoch}/{run_epochs} loss {train_loss:.4f}', flush=True)
                continue
            if args.pseudo_unseen:
                metrics = validate_pseudo(model, pseudo_data, seen_mask)
                row = pseudo_row(metrics)
                score = (row[6], metrics['calibrated']['harmonic_mean']['top1']['per_image'])
                print(f'epoch {epoch}/{run_epochs} loss {train_loss:.4f} | gamma 0: seen {row[0]:.3f} unseen '
                      f'{row[1]:.3f} H {row[2]:.3f} | gamma {row[3]:.2f}: seen {row[4]:.3f} unseen {row[5]:.3f} '
                      f'H {row[6]:.3f}', flush=True)
            else:
                metrics = validate(model, val_features, val_targets)
                row = [metrics['top1']['per_class'], metrics['top5']['per_class'],
                       metrics['top1']['per_image'], metrics['top5']['per_image']]
                score = (row[0], row[2])
                print(f'epoch {epoch}/{run_epochs} loss {train_loss:.4f} | per-class {row[0]:.3f}/{row[1]:.3f}'
                      f' | per-image {row[2]:.3f}/{row[3]:.3f}', flush=True)
            history.writerow([epoch, f'{current_lr:.2e}', f'{train_loss:.4f}', *[f'{value:.4f}' for value in row],
                              f'{time.time() - start:.0f}'])
            history_file.flush()
            if final:
                continue

            if score > best_score:
                best_score, stale_epochs = score, 0
                torch.save(checkpoint(epoch, metrics), run_dir / 'best.pt')
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f'Stopped early: no improvement for {args.patience} epochs.', flush=True)
                    break

    if final:
        torch.save(checkpoint(args.final_epochs, metrics), run_dir / 'final.pt')
        print(f'Saved to {run_dir / "final.pt"}')
    else:
        print(f'Best score {best_score[0]:.3f}. Saved to {run_dir}')


if __name__ == '__main__':
    main()
