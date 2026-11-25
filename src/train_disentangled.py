"""Disentangled Cross-Crop Contrastive Learning (and Masked GRL ablation) on ConvNeXt-Tiny.

Built directly on the team's pseudo-unseen GZSL protocol (src/gzsl.py), keeping the exact same
ConvNeXt-Tiny IMAGENET1K_V1 backbone and 224 px preprocessing as Baselines 1 and 2.

Methods (--method):
  branches   : Group 1 — B2 + dual orthogonal projection branches (crop & disease subspaces)
  masked_grl : Group 2 — Group 1 + Masked Gradient Reversal (adversarial crop penalty ONLY on
               diseases that appear on >= 2 crops in the current training split, protecting
               single-crop diseases from seen-accuracy collapse)
  supcon     : Group 3 — Group 1 + Cross-Crop Weighted Supervised Contrastive Learning on the
               disease subspace (pulls same-disease pairs together with extra weight on
               different-crop pairs, pushes different-disease pairs apart)

Test images are NEVER used during `--stage select` or calibration.

Example commands (select stage -> calibrate -> final stage):
  python src/train_disentangled.py --method branches   --stage select --out runs/g1_branches
  python src/train_disentangled.py --method masked_grl --stage select --out runs/g2_masked_grl
  python src/train_disentangled.py --method supcon     --stage select --batch-size 64 --out runs/g3_supcon
  python src/gzsl.py calibrate --pseudo runs/g3_supcon/pseudo.npz --out runs/g3_supcon/cal
  python src/train_disentangled.py --method supcon --stage final --final-epochs 15 --batch-size 64 --out runs/g3_final
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision.models import convnext_tiny

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy, save_predictions  # noqa: E402
from gzsl import (  # noqa: E402
    composition_log_probs,
    eval_loader,
    make_datasets,
    predict,
    pseudo_protocol,
    seed_everything,
    seen_columns,
    split_pseudo,
    sweep,
    val_score,
)
from preprocessing import balanced_sampler  # noqa: E402

NUM_CROPS = 55
NUM_DISEASES = 175


# ---------------------------------------------------------------------------
# Slot 1: Model Architecture (ConvNeXt-Tiny + Dual Orthogonal Branches)
# ---------------------------------------------------------------------------

class CosineClassifier(nn.Module):
    """L2-normalised cosine classifier to reduce majority-class norm bias on long-tailed labels."""

    def __init__(self, in_features, num_classes, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale = float(scale)

    def forward(self, x):
        x_norm = F.normalize(x.float(), dim=1)
        w_norm = F.normalize(self.weight.float(), dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class GradReverseFn(torch.autograd.Function):
    """Identity in forward pass; multiplies gradient by -lambd in backward pass."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverseFn.apply(x, lambd)


def make_projection(in_dim, embed_dim):
    return nn.Sequential(
        nn.Linear(in_dim, embed_dim),
        nn.LayerNorm(embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, embed_dim),
    )


class DisentangledNet(nn.Module):
    """Shared ConvNeXt-Tiny backbone split into separate crop and disease subspaces."""

    def __init__(self, embed_dim=256, cosine_head=False, use_grl=False, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        width = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()

        self.crop_proj = make_projection(width, embed_dim)
        self.disease_proj = make_projection(width, embed_dim)

        self.crop_head = nn.Linear(embed_dim, NUM_CROPS)
        if cosine_head:
            self.disease_head = CosineClassifier(embed_dim, NUM_DISEASES, scale=16.0)
        else:
            self.disease_head = nn.Linear(embed_dim, NUM_DISEASES)

        self.adv_crop_head = nn.Linear(embed_dim, NUM_CROPS) if use_grl else None

    def forward(self, images, return_features=False, grl_lambda=0.0):
        shared = self.backbone(images)
        z_crop = self.crop_proj(shared)
        z_disease = self.disease_proj(shared)

        out = {
            'crop': self.crop_head(z_crop),
            'disease': self.disease_head(z_disease),
        }
        if return_features:
            out['z_crop'] = z_crop
            out['z_disease'] = z_disease
            if self.adv_crop_head is not None:
                out['adv_crop'] = self.adv_crop_head(grad_reverse(z_disease, grl_lambda))
        return out


# ---------------------------------------------------------------------------
# Slot 2: P x K Cross-Crop Disease Batch Sampler (for SupCon)
# ---------------------------------------------------------------------------

class PKDiseaseBatchSampler(Sampler):
    """Draw P diseases per batch (square-root balanced) and K images per disease.

    For diseases that appear on multiple crops in the current training split, images are cycled
    across distinct crops so every batch contains genuine cross-crop positive pairs.
    """

    def __init__(self, records, batch_size=64, k_per_disease=4, seed=42):
        if batch_size % k_per_disease != 0:
            raise ValueError(f'batch_size ({batch_size}) must be divisible by k_per_disease ({k_per_disease}).')
        self.batch_size = batch_size
        self.k = k_per_disease
        self.p = batch_size // k_per_disease
        self.num_batches = max(len(records) // batch_size, 1)
        self.seed = seed
        self.epoch = 0

        by_disease = defaultdict(lambda: defaultdict(list))
        counts = defaultdict(int)
        for idx, rec in enumerate(records):
            d = rec['disease_id']
            c = rec['crop_id']
            by_disease[d][c].append(idx)
            counts[d] += 1

        self.diseases = sorted(by_disease.keys())
        self.disease_crops = {d: {c: list(idxs) for c, idxs in crops.items()} for d, crops in by_disease.items()}
        # Weight each disease by sqrt(count) so per-class frequency matches preprocessing.balanced_sampler(power=0.5).
        weights = np.array([math.sqrt(counts[d]) for d in self.diseases], dtype=np.float64)
        self.probs = weights / weights.sum()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        replace_diseases = self.p > len(self.diseases)
        for _ in range(self.num_batches):
            chosen_diseases = rng.choice(self.diseases, size=self.p, replace=replace_diseases, p=self.probs)
            batch = []
            for d in chosen_diseases:
                crop_map = self.disease_crops[int(d)]
                crops = list(crop_map.keys())
                rng.shuffle(crops)
                for step in range(self.k):
                    c = crops[step % len(crops)]
                    pool = crop_map[c]
                    batch.append(int(pool[int(rng.integers(len(pool)))]))
            yield batch


# ---------------------------------------------------------------------------
# Slot 3: Training Objectives (Orthogonality, Cross-Crop SupCon, Masked GRL)
# ---------------------------------------------------------------------------

def orthogonality_loss(z_crop, z_disease):
    """Push crop and disease projections to be orthogonal (mean squared cosine similarity -> 0)."""
    zc = F.normalize(z_crop.float(), dim=1)
    zd = F.normalize(z_disease.float(), dim=1)
    return (zc * zd).sum(dim=1).pow(2).mean()


def cross_crop_supcon_loss(z_disease, diseases, crops, temperature=0.1, cross_crop_weight=2.0):
    """Supervised Contrastive Loss with extra positive weight for same-disease, different-crop pairs."""
    z = F.normalize(z_disease.float(), dim=1)
    batch_size = z.size(0)
    device = z.device

    logits = (z @ z.T) / float(temperature)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    logits_no_self = logits.masked_fill(self_mask, float('-inf'))
    row_max = logits_no_self.max(dim=1, keepdim=True).values
    logits_stable = logits - row_max.detach()

    exp_logits = torch.exp(logits_stable).masked_fill(self_mask, 0.0)
    log_prob = logits_stable - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    pos_mask = (diseases[:, None] == diseases[None, :]) & (~self_mask)
    diff_crop = crops[:, None] != crops[None, :]
    pair_weights = torch.where(pos_mask & diff_crop, float(cross_crop_weight), 1.0) * pos_mask.float()

    weight_sum = pair_weights.sum(dim=1)
    valid = weight_sum > 0
    if not valid.any():
        return z.sum() * 0.0
    loss_per_anchor = -(pair_weights[valid] * log_prob[valid]).sum(dim=1) / weight_sum[valid]
    return loss_per_anchor.mean()


def build_multi_crop_disease_mask(records, device):
    """Boolean mask [175]: True iff disease appears on >= 2 crops in `records` (strictly train-only)."""
    crops_per_disease = defaultdict(set)
    for rec in records:
        crops_per_disease[rec['disease_id']].add(rec['crop_id'])
    mask = torch.zeros(NUM_DISEASES, dtype=torch.bool, device=device)
    for d, crops in crops_per_disease.items():
        if len(crops) >= 2:
            mask[d] = True
    return mask


def grl_schedule(progress, target_lambda):
    """Smooth Ganin & Lempitsky (2015) ramp from 0 to target_lambda as progress goes 0 -> 1."""
    return float(target_lambda) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


# ---------------------------------------------------------------------------
# Slot 4: Training Loop & Pseudo-Unseen Selection (Zero Test Leakage)
# ---------------------------------------------------------------------------

def evaluate_pseudo(preds, protocol, taxonomy):
    parts = split_pseudo(preds, protocol)
    seen = seen_columns(taxonomy, 'select')
    seen_log_p = composition_log_probs(parts['val_seen'], protocol['val_seen'], taxonomy)
    unseen_log_p = composition_log_probs(parts['pseudo_unseen'], protocol['pseudo_unseen'], taxonomy)
    _, best, ausuc = sweep(seen_log_p, protocol['val_seen'], unseen_log_p, protocol['pseudo_unseen'], seen)
    return best['harmonic'], ausuc, best['gamma']


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'

    data = make_datasets(args.stage)
    taxonomy = Taxonomy(load_labels())
    protocol = pseudo_protocol() if args.stage == 'select' else None

    sampler_type = args.sampler
    if sampler_type == 'auto':
        sampler_type = 'pk' if args.method == 'supcon' else 'balanced'

    loader_common = {
        'num_workers': args.workers,
        'pin_memory': use_cuda,
        'persistent_workers': args.workers > 0,
    }
    if sampler_type == 'pk':
        batch_sampler = PKDiseaseBatchSampler(
            data['train'].records,
            batch_size=args.batch_size,
            k_per_disease=args.k_per_disease,
            seed=args.seed,
        )
        train_loader = DataLoader(data['train'], batch_sampler=batch_sampler, **loader_common)
    else:
        sampler = balanced_sampler(
            data['train'].records,
            key='composition_id',
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            data['train'],
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
            **loader_common,
        )

    use_grl = args.method == 'masked_grl'
    model = DisentangledNet(
        embed_dim=args.embed_dim,
        cosine_head=args.cosine_head,
        use_grl=use_grl,
        pretrained=True,
    ).to(device)

    multi_crop_mask = build_multi_crop_disease_mask(data['train'].records, device)
    num_multi_crop = int(multi_crop_mask.sum().item())
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=use_cuda)

    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'resolved_sampler': sampler_type,
        'backbone': 'convnext_tiny',
        'weights': 'torchvision convnext_tiny IMAGENET1K_V1',
        'image_size': 224,
        'train_images': len(data['train']),
        'train_multi_crop_diseases': num_multi_crop,
        'parameters': sum(p.numel() for p in model.parameters()),
        'device': torch.cuda.get_device_name(0) if use_cuda else 'cpu',
        'torch': torch.__version__,
    }
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    pseudo_loader = eval_loader(data['pseudo'], args.batch_size, args.workers) if args.stage == 'select' else None

    best_score, best_epoch, stale = -1.0, None, 0
    total_full_steps = max(len(train_loader) * (args.final_epochs if args.stage == 'final' else args.epochs), 1)
    full_step = 0

    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow([
            'stage', 'epoch', 'lr', 'train_loss', 'ce_loss', 'ortho_loss', 'method_loss',
            'val_seen_comp_top1', 'pseudo_h', 'pseudo_ausuc', 'pseudo_gamma', 'seconds',
        ])

        for stage, stage_epochs, lr in (('head', args.head_epochs, args.head_lr), ('full', args.epochs, args.lr)):
            head_only = stage == 'head'
            for p in model.parameters():
                p.requires_grad = True
            if head_only:
                for p in model.backbone.parameters():
                    p.requires_grad = False

            optimiser = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=0.05,
            )
            run_epochs = args.final_epochs if args.stage == 'final' and not head_only else stage_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(stage_epochs, 1))

            for epoch in range(1, run_epochs + 1):
                start = time.perf_counter()
                model.train()
                if head_only:
                    model.backbone.eval()

                sum_total = sum_ce = sum_ortho = sum_method = 0.0
                count = 0

                for batch in train_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    crops = batch['crop'].to(device, non_blocking=True)
                    diseases = batch['disease'].to(device, non_blocking=True)

                    if head_only:
                        cur_lambda = 0.0
                    else:
                        full_step += 1
                        cur_lambda = grl_schedule(min(full_step / total_full_steps, 1.0), args.grl_lambda)

                    optimiser.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=use_cuda):
                        out = model(images, return_features=True, grl_lambda=cur_lambda)
                        loss_ce = ce_loss(out['crop'], crops) + ce_loss(out['disease'], diseases)
                        loss_ortho = orthogonality_loss(out['z_crop'], out['z_disease'])

                        if args.method == 'supcon':
                            loss_method = cross_crop_supcon_loss(
                                out['z_disease'],
                                diseases,
                                crops,
                                temperature=args.temperature,
                                cross_crop_weight=args.w_cross,
                            )
                            loss = loss_ce + args.w_ortho * loss_ortho + args.w_method * loss_method
                        elif args.method == 'masked_grl':
                            keep = multi_crop_mask[diseases]
                            if keep.any():
                                loss_method = ce_loss(out['adv_crop'][keep], crops[keep])
                            else:
                                loss_method = out['adv_crop'].sum() * 0.0
                            # adv_crop_head trains at full weight; cur_lambda controls reversal strength into z_disease.
                            loss = loss_ce + args.w_ortho * loss_ortho + loss_method
                        else:
                            loss_method = loss_ortho * 0.0
                            loss = loss_ce + args.w_ortho * loss_ortho

                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss at {stage} epoch {epoch}; lower lr or loss weights.')

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()

                    bs = len(images)
                    sum_total += loss.item() * bs
                    sum_ce += loss_ce.item() * bs
                    sum_ortho += loss_ortho.item() * bs
                    sum_method += loss_method.item() * bs
                    count += bs

                lr_now = optimiser.param_groups[0]['lr']
                scheduler.step()

                val_seen_score = pseudo_h = pseudo_ausuc = pseudo_gamma = None
                if args.stage == 'select':
                    model.eval()
                    pseudo_preds = predict(model, pseudo_loader, device)
                    val_seen_preds = split_pseudo(pseudo_preds, protocol)['val_seen']
                    val_seen_score = val_score(val_seen_preds, protocol['val_seen'], taxonomy, 'composition')
                    pseudo_h, pseudo_ausuc, pseudo_gamma = evaluate_pseudo(pseudo_preds, protocol, taxonomy)

                    if args.select_by == 'pseudo_ausuc':
                        metric = pseudo_ausuc
                    elif args.select_by == 'pseudo_h':
                        metric = pseudo_h
                    else:
                        metric = val_seen_score

                    if metric > best_score:
                        best_score, best_epoch, stale = metric, (stage, epoch), 0
                        torch.save(
                            {
                                'model': model.state_dict(),
                                'stage': stage,
                                'epoch': epoch,
                                'select_by': args.select_by,
                                'score': metric,
                                'val_seen_comp_top1': val_seen_score,
                                'pseudo_h': pseudo_h,
                                'pseudo_ausuc': pseudo_ausuc,
                                'pseudo_gamma': pseudo_gamma,
                            },
                            args.out / 'best.pth',
                        )
                    elif not head_only:
                        stale += 1

                seconds = time.perf_counter() - start
                history.writerow([
                    stage, epoch, f'{lr_now:.2e}',
                    round(sum_total / count, 4), round(sum_ce / count, 4),
                    round(sum_ortho / count, 4), round(sum_method / count, 4),
                    val_seen_score, pseudo_h, pseudo_ausuc, pseudo_gamma, round(seconds, 1),
                ])
                file.flush()

                if args.stage == 'select':
                    print(
                        f'{stage} {epoch}/{run_epochs} loss={sum_total/count:.3f} '
                        f'(ce={sum_ce/count:.3f} ortho={sum_ortho/count:.3f} m={sum_method/count:.3f}) | '
                        f'val_seen={100*val_seen_score:.2f} pseudo_H={100*pseudo_h:.2f} '
                        f'AUSUC={100*pseudo_ausuc:.2f} (g={pseudo_gamma}) {seconds:.0f}s',
                        flush=True,
                    )
                else:
                    print(
                        f'{stage} {epoch}/{run_epochs} loss={sum_total/count:.3f} '
                        f'(ce={sum_ce/count:.3f} ortho={sum_ortho/count:.3f} m={sum_method/count:.3f}) {seconds:.0f}s',
                        flush=True,
                    )

                if args.stage == 'select' and not head_only and stale >= args.patience:
                    print(f'Early stop after {args.patience} stale epochs.', flush=True)
                    break

    if args.stage == 'select':
        ckpt = torch.load(args.out / 'best.pth', map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        model.eval()
        final_pseudo_preds = predict(model, pseudo_loader, device)
        save_predictions(args.out / 'pseudo.npz', **final_pseudo_preds)
        summary = {
            'method': args.method,
            'select_by': args.select_by,
            'best_stage_epoch': best_epoch,
            'best_score': best_score,
            'best_val_seen_comp_top1': ckpt['val_seen_comp_top1'],
            'best_pseudo_h': ckpt['pseudo_h'],
            'best_pseudo_ausuc': ckpt['pseudo_ausuc'],
            'best_pseudo_gamma': ckpt['pseudo_gamma'],
        }
    else:
        torch.save(
            {'model': model.state_dict(), 'stage': 'final', 'epochs': args.final_epochs, 'config': config},
            args.out / 'final.pth',
        )
        model.eval()
        for split in ('test_seen', 'test_unseen'):
            preds = predict(model, eval_loader(data[split], args.batch_size, args.workers), device)
            save_predictions(args.out / f'{split}.npz', **preds)
        summary = {'method': args.method, 'final_full_epochs': args.final_epochs}

    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=('branches', 'masked_grl', 'supcon'), required=True)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--final-epochs', type=int, help='Full-model epochs for final stage (from best select epoch).')
    parser.add_argument('--select-by', choices=('val_seen', 'pseudo_h', 'pseudo_ausuc'), default='val_seen')
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--cosine-head', action='store_true', help='Use L2-normalised cosine disease classifier.')
    parser.add_argument('--sampler', choices=('auto', 'balanced', 'pk'), default='auto')
    parser.add_argument('--k-per-disease', type=int, default=4, help='Images per disease in P x K sampler.')
    parser.add_argument('--w-ortho', type=float, default=0.1, help='Weight on crop/disease orthogonality loss.')
    parser.add_argument('--w-method', type=float, default=0.2, help='Weight on SupCon or Masked GRL loss.')
    parser.add_argument('--w-cross', type=float, default=2.0, help='Extra positive weight for cross-crop SupCon pairs.')
    parser.add_argument('--temperature', type=float, default=0.1, help='SupCon softmax temperature.')
    parser.add_argument('--grl-lambda', type=float, default=0.2, help='Target GRL reversal strength for Masked GRL.')
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage == 'final' and not args.final_epochs:
        parser.error('--final-epochs is required for --stage final.')
    run(args)


if __name__ == '__main__':
    main()
