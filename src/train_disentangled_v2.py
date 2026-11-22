"""Disentangled Dual-Branch ConvNeXt-Tiny v2 (Backbone-Protected Masked GRL, MixStyle, Logit Adjustment, EMA).

Upgrades over v1 (src/train_disentangled.py), motivated strictly by select-stage (pseudo-unseen) diagnostics:
  1. Backbone-Protected Masked GRL: adversarial crop reversal is applied to `disease_proj(shared.detach())`,
     so `disease_proj` learns to strip crop cues from `z_disease` WITHOUT sending negative gradients back
     into the shared ConvNeXt-Tiny backbone (preventing crop/disease backbone degradation).
  2. Wide Residual Subspaces (512-dim default, vs 256-dim bottleneck in v1) + matched head geometry
     (`--head-type linear` or symmetric learnable-scale `--head-type cosine`).
  3. Optional Stage 1+2 MixStyle (Zhou et al., ICLR 2021) for cross-crop leaf style/lighting generalisation
     without image-level CutMix label noise.
  4. Optional training-time Logit Adjustment (Menon et al., ICLR 2021) computed strictly from the effective
     square-root-sampled training prior, giving rare diseases stronger decision margins.
  5. Consistent GRL ramp between select and final stages (always normalised by `--epochs`, not `--final-epochs`),
     full cosine LR annealing over `run_epochs`, and Exponential Moving Average (EMA) weights.

Test images are NEVER used during `--stage select` or calibration.

Example commands (select stage -> calibrate -> final stage):
  python src/train_disentangled_v2.py --stage select --out runs/v2_g1_clean
  python src/train_disentangled_v2.py --stage select --mixstyle --out runs/v2_g2_mixstyle
  python src/train_disentangled_v2.py --stage select --logit-adj-tau 0.5 --out runs/v2_g3_logit_adj
  python src/train_disentangled_v2.py --stage select --mixstyle --logit-adj-tau 0.5 --out runs/v2_g4_mixstyle_la
"""
import argparse
from collections import Counter, defaultdict
import copy
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
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy, accuracy, align, level_scores, save_predictions, top_k_hits  # noqa: E402
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
# Building Blocks: MixStyle, Residual Subspace, GRL, Symmetric Cosine Head, EMA
# ---------------------------------------------------------------------------

class MixStyle(nn.Module):
    """Domain Generalization with MixStyle (Zhou et al., ICLR 2021).

    Mixes per-instance channel mean/std across random images in the mini-batch during training,
    perturbing leaf colour/lighting/background style while preserving spatial lesion structure.
    Strict no-op during eval().
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x):
        if not self.training or self.p <= 0.0 or x.size(0) < 2:
            return x
        if torch.rand(1).item() > self.p:
            return x

        orig_dtype = x.dtype
        xf = x.float()
        bsz = xf.size(0)
        mu = xf.mean(dim=(2, 3), keepdim=True)
        var = xf.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_normed = (xf - mu) / sig

        beta = torch.distributions.Beta(
            torch.tensor(self.alpha, device=x.device, dtype=torch.float32),
            torch.tensor(self.alpha, device=x.device, dtype=torch.float32),
        )
        lmda = beta.sample((bsz, 1, 1, 1))
        perm = torch.randperm(bsz, device=x.device)

        mu_mix = lmda * mu + (1.0 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1.0 - lmda) * sig[perm]
        return (x_normed * sig_mix + mu_mix).to(orig_dtype)


class LearnableCosineClassifier(nn.Module):
    """L2-normalised cosine classifier with learnable temperature scale."""

    def __init__(self, in_features, num_classes, init_scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.log_scale = nn.Parameter(torch.tensor(math.log(float(init_scale))))

    def forward(self, x):
        scale = self.log_scale.exp().clamp(1.0, 64.0)
        x_norm = F.normalize(x.float(), dim=1)
        w_norm = F.normalize(self.weight.float(), dim=1)
        return scale * F.linear(x_norm, w_norm)


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


class ResidualProjection(nn.Module):
    """Wide residual projection preserving backbone variance while allowing orthogonal subspaces."""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.skip(x) + self.net(x))


class DisentangledNetV2(nn.Module):
    """ConvNeXt-Tiny + optional Stage 1&2 MixStyle + wide residual orthogonal branches + backbone-protected GRL."""

    def __init__(
        self,
        embed_dim=512,
        head_type='linear',
        use_grl=True,
        detach_grl_backbone=True,
        use_mixstyle=False,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1,
        dropout=0.1,
        pretrained=True,
    ):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        width = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()

        self.use_mixstyle = bool(use_mixstyle)
        self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()
        self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()

        self.crop_proj = ResidualProjection(width, embed_dim, dropout=dropout)
        self.disease_proj = ResidualProjection(width, embed_dim, dropout=dropout)

        if head_type == 'cosine':
            self.crop_head = LearnableCosineClassifier(embed_dim, NUM_CROPS, init_scale=16.0)
            self.disease_head = LearnableCosineClassifier(embed_dim, NUM_DISEASES, init_scale=16.0)
        else:
            self.crop_head = nn.Linear(embed_dim, NUM_CROPS)
            self.disease_head = nn.Linear(embed_dim, NUM_DISEASES)

        self.use_grl = bool(use_grl)
        self.detach_grl_backbone = bool(detach_grl_backbone)
        self.adv_crop_head = nn.Linear(embed_dim, NUM_CROPS) if self.use_grl else None

    def extract_backbone(self, images):
        x = images
        for idx, block in enumerate(self.backbone.features):
            x = block(x)
            if self.use_mixstyle and idx == 1:
                x = self.mixstyle1(x)
            elif self.use_mixstyle and idx == 3:
                x = self.mixstyle2(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)
        return x

    def forward(self, images, return_features=False, grl_lambda=0.0):
        shared = self.extract_backbone(images)
        z_crop = self.crop_proj(shared)
        z_disease = self.disease_proj(shared)

        out = {
            'crop': self.crop_head(z_crop),
            'disease': self.disease_head(z_disease),
        }
        if return_features:
            out['shared'] = shared
            out['z_crop'] = z_crop
            out['z_disease'] = z_disease
            if self.adv_crop_head is not None:
                # Re-run disease_proj on detached shared features so adversarial gradients clean
                # `disease_proj` without corrupting the shared ConvNeXt-Tiny backbone.
                if self.detach_grl_backbone:
                    z_disease_adv = self.disease_proj(shared.detach())
                else:
                    z_disease_adv = z_disease
                out['adv_crop'] = self.adv_crop_head(grad_reverse(z_disease_adv, grl_lambda))
        return out


class ModelEMA:
    """Exponential Moving Average of model parameters & buffers (warmup-stabilised)."""

    def __init__(self, model, decay=0.995):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0

    def reset_from(self, model):
        self.ema.load_state_dict(model.state_dict())
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1.0 - d)
            else:
                v.copy_(src)


# ---------------------------------------------------------------------------
# Losses & Strictly Train-Only Statistics
# ---------------------------------------------------------------------------

def orthogonality_loss(z_crop, z_disease):
    """Push crop and disease subspaces to be orthogonal (mean squared cosine similarity -> 0)."""
    zc = F.normalize(z_crop.float(), dim=1)
    zd = F.normalize(z_disease.float(), dim=1)
    return (zc * zd).sum(dim=1).pow(2).mean()


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


def compute_effective_log_priors(records, power=0.5, device='cpu'):
    """Effective log-priors under composition square-root sampling (`power=0.5`), strictly train-only.

    If composition k has n_k images, each image gets weight n_k^{-power}, so composition k has
    total sampling mass n_k^{1 - power}. Summing that mass over compositions gives the true
    batch-level marginal prior for crops and diseases, avoiding double-counting with the sampler.
    """
    comp_counts = Counter(r['composition_id'] for r in records)
    comp_meta = {}
    for r in records:
        comp_meta[r['composition_id']] = (r['crop_id'], r['disease_id'])

    crop_mass = np.zeros(NUM_CROPS, dtype=np.float64)
    disease_mass = np.zeros(NUM_DISEASES, dtype=np.float64)
    for cid, cnt in comp_counts.items():
        mass = float(cnt) ** (1.0 - float(power))
        c_id, d_id = comp_meta[cid]
        crop_mass[c_id] += mass
        disease_mass[d_id] += mass

    crop_prior = np.clip(crop_mass / max(crop_mass.sum(), 1e-12), 1e-12, 1.0)
    disease_prior = np.clip(disease_mass / max(disease_mass.sum(), 1e-12), 1e-12, 1.0)
    return (
        torch.tensor(np.log(crop_prior), dtype=torch.float32, device=device),
        torch.tensor(np.log(disease_prior), dtype=torch.float32, device=device),
    )


def grl_schedule(progress, target_lambda):
    """Smooth Ganin & Lempitsky (2015) ramp from 0 to target_lambda as progress goes 0 -> 1."""
    p = min(max(float(progress), 0.0), 1.0)
    return float(target_lambda) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ---------------------------------------------------------------------------
# Pseudo-Unseen Evaluation (Zero Test Leakage)
# ---------------------------------------------------------------------------

def evaluate_pseudo_detailed(preds, protocol, taxonomy):
    """Return full select-stage metrics so we monitor both GZSL H/AUSUC and raw discrimination."""
    parts = split_pseudo(preds, protocol)
    seen = seen_columns(taxonomy, 'select')
    seen_log_p = composition_log_probs(parts['val_seen'], protocol['val_seen'], taxonomy)
    unseen_log_p = composition_log_probs(parts['pseudo_unseen'], protocol['pseudo_unseen'], taxonomy)
    rows, best, ausuc = sweep(seen_log_p, protocol['val_seen'], unseen_log_p, protocol['pseudo_unseen'], seen)

    pseudo_unseen_mask = np.zeros(taxonomy.size('composition'), dtype=bool)
    for r in protocol['pseudo_unseen']:
        pseudo_unseen_mask[r['composition_id']] = True

    def acc_level(part, records, level, unseen_only=False):
        aligned = align(part, records)
        scores = level_scores(aligned, taxonomy)[level]
        if unseen_only:
            scores = np.where(pseudo_unseen_mask[None, :], scores, -1e9)
        targets = np.array([r[level + '_id'] for r in records])
        return float(accuracy(top_k_hits(scores, targets, 1), targets)['per_class'])

    return {
        'val_seen_comp': val_score(parts['val_seen'], protocol['val_seen'], taxonomy, 'composition'),
        'val_seen_crop': acc_level(parts['val_seen'], protocol['val_seen'], 'crop'),
        'val_seen_disease': acc_level(parts['val_seen'], protocol['val_seen'], 'disease'),
        'pseudo_unseen_crop': acc_level(parts['pseudo_unseen'], protocol['pseudo_unseen'], 'crop'),
        'pseudo_unseen_disease': acc_level(parts['pseudo_unseen'], protocol['pseudo_unseen'], 'disease'),
        'pseudo_unseen_only': acc_level(parts['pseudo_unseen'], protocol['pseudo_unseen'], 'composition', unseen_only=True),
        'pseudo_h': float(best['harmonic']),
        'pseudo_ausuc': float(ausuc),
        'pseudo_gamma': float(best['gamma']),
        'pseudo_raw_h': float(rows[0]['harmonic']),
    }


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'

    data = make_datasets(args.stage)
    taxonomy = Taxonomy(load_labels())
    protocol = pseudo_protocol() if args.stage == 'select' else None

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
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0,
    )

    use_grl = args.grl_lambda > 0.0
    model = DisentangledNetV2(
        embed_dim=args.embed_dim,
        head_type=args.head_type,
        use_grl=use_grl,
        detach_grl_backbone=not args.no_detach_grl,
        use_mixstyle=args.mixstyle,
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    multi_crop_mask = build_multi_crop_disease_mask(data['train'].records, device)
    crop_log_prior, disease_log_prior = compute_effective_log_priors(data['train'].records, power=0.5, device=device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=use_cuda)

    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'backbone': 'convnext_tiny',
        'weights': 'torchvision convnext_tiny IMAGENET1K_V1',
        'image_size': 224,
        'train_images': len(data['train']),
        'train_multi_crop_diseases': int(multi_crop_mask.sum().item()),
        'parameters': sum(p.numel() for p in model.parameters()),
        'device': torch.cuda.get_device_name(0) if use_cuda else 'cpu',
        'torch': torch.__version__,
    }
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    pseudo_loader = eval_loader(data['pseudo'], args.batch_size, args.workers) if args.stage == 'select' else None

    best_score, best_epoch, stale = -1.0, None, 0
    # CRITICAL FIX vs v1: always normalise GRL schedule by `args.epochs` (not `args.final_epochs`),
    # so epoch k in `final` experiences the exact same GRL strength as epoch k in `select`.
    grl_total_steps = max(len(train_loader) * args.epochs, 1)
    full_step = 0

    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow([
            'stage', 'epoch', 'lr', 'cur_lambda', 'train_loss', 'ce_loss', 'ortho_loss', 'adv_loss',
            'val_seen_comp', 'val_seen_crop', 'val_seen_disease',
            'pseudo_unseen_crop', 'pseudo_unseen_only', 'pseudo_h', 'pseudo_ausuc', 'pseudo_gamma', 'seconds',
        ])

        for stage, stage_epochs, lr in (('head', args.head_epochs, args.head_lr), ('full', args.epochs, args.lr)):
            head_only = stage == 'head'
            for p in model.parameters():
                p.requires_grad = True
            if head_only:
                for p in model.backbone.parameters():
                    p.requires_grad = False
            elif ema is not None:
                # Start EMA cleanly from the post-warmup weights so frozen-head epochs do not lag EMA.
                ema.reset_from(model)

            optimiser = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=0.05,
            )
            run_epochs = args.final_epochs if args.stage == 'final' and not head_only else stage_epochs
            # CRITICAL FIX vs v1: in final stage, anneal cosine LR smoothly over `run_epochs`.
            t_max = run_epochs if args.stage == 'final' and not head_only else stage_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(t_max, 1))

            for epoch in range(1, run_epochs + 1):
                start = time.perf_counter()
                model.train()
                if head_only:
                    model.backbone.eval()

                sum_total = sum_ce = sum_ortho = sum_adv = 0.0
                count = 0
                cur_lambda = 0.0

                for batch in train_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    crops = batch['crop'].to(device, non_blocking=True)
                    diseases = batch['disease'].to(device, non_blocking=True)

                    if head_only or not use_grl:
                        cur_lambda = 0.0
                    else:
                        full_step += 1
                        cur_lambda = grl_schedule(full_step / grl_total_steps, args.grl_lambda)

                    optimiser.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=use_cuda):
                        out = model(images, return_features=True, grl_lambda=cur_lambda)
                        crop_logits = out['crop']
                        disease_logits = out['disease']
                        if args.logit_adj_tau > 0.0:
                            crop_logits = crop_logits + args.logit_adj_tau * crop_log_prior
                            disease_logits = disease_logits + args.logit_adj_tau * disease_log_prior

                        loss_ce = ce_loss(crop_logits, crops) + ce_loss(disease_logits, diseases)
                        loss_ortho = orthogonality_loss(out['z_crop'], out['z_disease'])

                        if use_grl and not head_only:
                            keep = multi_crop_mask[diseases]
                            if keep.any():
                                loss_adv = ce_loss(out['adv_crop'][keep], crops[keep])
                            else:
                                loss_adv = out['adv_crop'].sum() * 0.0
                            loss = loss_ce + args.w_ortho * loss_ortho + loss_adv
                        else:
                            loss_adv = loss_ortho * 0.0
                            loss = loss_ce + args.w_ortho * loss_ortho

                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss at {stage} epoch {epoch}; lower lr or loss weights.')

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()

                    if ema is not None and not head_only:
                        ema.update(model)

                    bs = len(images)
                    sum_total += loss.item() * bs
                    sum_ce += loss_ce.item() * bs
                    sum_ortho += loss_ortho.item() * bs
                    sum_adv += loss_adv.item() * bs
                    count += bs

                lr_now = optimiser.param_groups[0]['lr']
                scheduler.step()

                eval_model = ema.ema if (ema is not None and not head_only) else model
                metrics = None
                if args.stage == 'select':
                    eval_model.eval()
                    pseudo_preds = predict(eval_model, pseudo_loader, device)
                    metrics = evaluate_pseudo_detailed(pseudo_preds, protocol, taxonomy)

                    if args.select_by == 'pseudo_ausuc':
                        metric = metrics['pseudo_ausuc']
                    elif args.select_by == 'pseudo_h':
                        metric = metrics['pseudo_h']
                    else:
                        metric = metrics['val_seen_comp']

                    if metric > best_score:
                        best_score, best_epoch, stale = metric, (stage, epoch), 0
                        torch.save(
                            {
                                'model': eval_model.state_dict(),
                                'stage': stage,
                                'epoch': epoch,
                                'select_by': args.select_by,
                                'score': metric,
                                **metrics,
                            },
                            args.out / 'best.pth',
                        )
                    elif not head_only:
                        stale += 1

                seconds = time.perf_counter() - start
                if metrics is not None:
                    history.writerow([
                        stage, epoch, f'{lr_now:.2e}', round(cur_lambda, 5),
                        round(sum_total / count, 4), round(sum_ce / count, 4),
                        round(sum_ortho / count, 4), round(sum_adv / count, 4),
                        metrics['val_seen_comp'], metrics['val_seen_crop'], metrics['val_seen_disease'],
                        metrics['pseudo_unseen_crop'], metrics['pseudo_unseen_only'],
                        metrics['pseudo_h'], metrics['pseudo_ausuc'], metrics['pseudo_gamma'], round(seconds, 1),
                    ])
                    print(
                        f'{stage} {epoch}/{run_epochs} loss={sum_total/count:.3f} '
                        f'(ce={sum_ce/count:.3f} ortho={sum_ortho/count:.4f} adv={sum_adv/count:.3f} lam={cur_lambda:.4f}) | '
                        f'S_comp={100*metrics["val_seen_comp"]:.2f} S_dis={100*metrics["val_seen_disease"]:.2f} '
                        f'U_crop={100*metrics["pseudo_unseen_crop"]:.2f} U_u10={100*metrics["pseudo_unseen_only"]:.2f} | '
                        f'pseudo_H={100*metrics["pseudo_h"]:.2f} AUSUC={100*metrics["pseudo_ausuc"]:.2f} '
                        f'(g={metrics["pseudo_gamma"]}) {seconds:.0f}s',
                        flush=True,
                    )
                else:
                    history.writerow([
                        stage, epoch, f'{lr_now:.2e}', round(cur_lambda, 5),
                        round(sum_total / count, 4), round(sum_ce / count, 4),
                        round(sum_ortho / count, 4), round(sum_adv / count, 4),
                        '', '', '', '', '', '', '', '', round(seconds, 1),
                    ])
                    print(
                        f'{stage} {epoch}/{run_epochs} loss={sum_total/count:.3f} '
                        f'(ce={sum_ce/count:.3f} ortho={sum_ortho/count:.4f} adv={sum_adv/count:.3f} lam={cur_lambda:.4f}) {seconds:.0f}s',
                        flush=True,
                    )
                file.flush()

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
            'version': 'v2',
            'select_by': args.select_by,
            'best_stage_epoch': best_epoch,
            'best_score': best_score,
            'best_val_seen_comp': ckpt['val_seen_comp'],
            'best_val_seen_disease': ckpt['val_seen_disease'],
            'best_pseudo_unseen_crop': ckpt['pseudo_unseen_crop'],
            'best_pseudo_unseen_only': ckpt['pseudo_unseen_only'],
            'best_pseudo_h': ckpt['pseudo_h'],
            'best_pseudo_ausuc': ckpt['pseudo_ausuc'],
            'best_pseudo_gamma': ckpt['pseudo_gamma'],
            'best_pseudo_raw_h': ckpt['pseudo_raw_h'],
        }
    else:
        final_model = ema.ema if ema is not None else model
        torch.save(
            {'model': final_model.state_dict(), 'stage': 'final', 'epochs': args.final_epochs, 'config': config},
            args.out / 'final.pth',
        )
        final_model.eval()
        for split in ('test_seen', 'test_unseen'):
            preds = predict(final_model, eval_loader(data[split], args.batch_size, args.workers), device)
            save_predictions(args.out / f'{split}.npz', **preds)
        summary = {'version': 'v2', 'final_full_epochs': args.final_epochs}

    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--final-epochs', type=int, help='Full-model epochs for final stage (from best select epoch).')
    parser.add_argument('--select-by', choices=('val_seen', 'pseudo_h', 'pseudo_ausuc'), default='pseudo_ausuc')
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--head-type', choices=('linear', 'cosine'), default='linear')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--w-ortho', type=float, default=0.05, help='Weight on crop/disease orthogonality loss.')
    parser.add_argument('--grl-lambda', type=float, default=0.02, help='Target Masked GRL strength (0 disables GRL).')
    parser.add_argument('--no-detach-grl', action='store_true', help='Ablation: allow GRL gradient into backbone.')
    parser.add_argument('--mixstyle', action='store_true', help='Enable Stage 1&2 MixStyle during full training.')
    parser.add_argument('--mixstyle-p', type=float, default=0.5)
    parser.add_argument('--mixstyle-alpha', type=float, default=0.1)
    parser.add_argument('--logit-adj-tau', type=float, default=0.0, help='Training-time Logit Adjustment tau (0 disables).')
    parser.add_argument('--ema-decay', type=float, default=0.995, help='EMA decay during full training (0 disables).')
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=12)
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
