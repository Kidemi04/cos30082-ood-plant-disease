r"""CCD-Net: Cross-Crop Disease network. A disease-only model trained against the crop shortcut.

Why: on test_unseen every model's disease head is right only 18-20% of the time, and 63-72% of its mistakes name a
disease that was trained with the image's own crop. Separate training alone (Baseline 1) does not fix that: B1's
disease network is still at 19.0%, because in the training data the crop predicts the disease, so any disease
classifier learns leaf shape as a shortcut. CCD-Net keeps B1's idea (a disease network that never sees crop labels)
and adds two things aimed at the shortcut:

  disease sampler   square-root sampling by disease_id, as B1's disease network (CPN and B2 sample by composition).
  cross-crop batch  each batch of 32 = 12 images drawn by the disease sampler, 10 "bridge" images (5 diseases seen on
                    >= 2 crops, 2 different crops each) and 10 "decoy" images (for every bridge image, another image
                    of the same crop with a different disease).
  cross-crop loss   supervised contrastive loss on a projection of z_d. Positives: same disease, different crop.
                    Images of the same composition are left out of both sides, so "same leaf" is never rewarded.
                    Decoys sit in the denominator: same crop, different disease, so leaf shape cannot be the cue.

Architecture (same encoder recipe as CPN, so the crop head can come from CPN-p0 unchanged)
  BioCLIP ViT-B/16, last 4 blocks + ln_post trained; one learned disease query attends over the 196 patch tokens;
  2-layer MLP to z_d (256); linear disease head (175); projection MLP (128) for the contrastive loss only.
  EMA weights + horizontal-flip TTA at evaluation. Epoch selected by seen validation disease Top-1 (as B1).

The disease file is merged with CPN-p0's crop predictions (`gzsl.py merge`), then gamma is chosen on the
pseudo-unseen split exactly as for every other model.

  python src\train_ccd.py --stage select --out runs\ccd_select
  python src\train_ccd.py --stage final --final-epochs 10 --out runs\ccd_final
  python src\train_ccd.py --stage select --contrastive 0 --out runs\ccd_nocon_select   (ablation: no cross-crop loss)
"""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import open_clip
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy  # noqa: E402
from gzsl import eval_loader, make_datasets, predict, seed_everything, val_score  # noqa: E402
from preprocessing import CLIP_MEAN, CLIP_STD  # noqa: E402
from train_clip_compositional import BACKBONES  # noqa: E402
from train_cpn import mlp, tta  # noqa: E402


class CrossCropBatches(Sampler):
    """Batches of random (sqrt by disease) + bridge (same disease, different crops) + decoy (same crop) images."""

    def __init__(self, records, batch_size=32, groups=5, seed=42, steps=None, power=0.5):
        self.rng = np.random.default_rng(seed)
        self.groups, self.n_random = groups, batch_size - 4 * groups
        if self.n_random < 0:
            raise ValueError('batch_size must be at least 4 x groups.')
        self.steps = steps or len(records) // batch_size
        disease = np.array([r['disease_id'] for r in records])
        crop = np.array([r['crop_id'] for r in records])
        counts = Counter(disease.tolist())
        weights = np.array([counts[d] ** -power for d in disease.tolist()])
        self.weights = weights / weights.sum()
        self.by_pair = defaultdict(list)
        for i, (c, d) in enumerate(zip(crop.tolist(), disease.tolist())):
            self.by_pair[(c, d)].append(i)
        self.crops_of = defaultdict(list)
        self.diseases_of = defaultdict(list)
        for c, d in self.by_pair:
            self.crops_of[d].append(c)
            self.diseases_of[c].append(d)
        self.bridge_diseases = sorted(d for d, cs in self.crops_of.items() if len(cs) >= 2)
        self.n = len(records)

    def __len__(self):
        return self.steps

    def pick(self, c, d):
        return int(self.rng.choice(self.by_pair[(c, d)]))

    def __iter__(self):
        for _ in range(self.steps):
            batch = self.rng.choice(self.n, self.n_random, replace=True, p=self.weights).tolist()
            for d in self.rng.choice(self.bridge_diseases, self.groups, replace=False):
                for c in self.rng.choice(self.crops_of[d], 2, replace=False):
                    batch.append(self.pick(c, d))
                    others = [o for o in self.diseases_of[c] if o != d]
                    if others:  # decoy: same crop, different disease
                        batch.append(self.pick(c, int(self.rng.choice(others))))
                    else:  # crop has one disease: fall back to a random image
                        batch.append(int(self.rng.choice(self.n, p=self.weights)))
            yield batch


class CCDNet(nn.Module):
    def __init__(self, taxonomy, device, unfreeze=4, dim=256, proj_dim=128):
        super().__init__()
        arch, pretrained = BACKBONES['bioclip']
        clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = clip.visual.to(device)
        del clip
        self.visual.requires_grad_(False)
        for block in self.visual.transformer.resblocks[-unfreeze:]:
            block.requires_grad_(True)
        self.visual.ln_post.requires_grad_(True)  # proj stays frozen: the token path never uses it
        self.visual.output_tokens = True
        width = self.visual.ln_post.normalized_shape[0]
        self.query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.to_d = mlp(width, width, dim)
        self.disease_head = nn.Linear(dim, taxonomy.size('disease'))
        self.proj = mlp(dim, dim, proj_dim)

    def features(self, images):
        _, tokens = self.visual(images)
        q = self.query.unsqueeze(0).expand(len(tokens), -1, -1).to(tokens.dtype)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.to_d(self.norm(out[:, 0]))

    def forward(self, images):
        return {'disease': self.disease_head(self.features(images))}


def cross_crop_loss(z, disease, crop, composition, temperature=0.1):
    """Supervised contrastive loss. Positives: same disease, different crop. Same composition: excluded."""
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T / temperature
    same_d = disease[:, None] == disease[None]
    same_c = crop[:, None] == crop[None]
    same_pair = composition[:, None] == composition[None]  # includes the anchor itself
    positive = same_d & ~same_c
    keep = positive.any(1)
    if not keep.any():
        return sim.sum() * 0.0
    log_prob = sim - torch.logsumexp(sim.masked_fill(same_pair, float('-inf')), dim=1, keepdim=True)
    loss = -(log_prob * positive).sum(1)[keep] / positive.sum(1)[keep]
    return loss.mean()


def save_disease(path, predictions):
    """Disease-only prediction file in the evaluate.save_predictions format, for `gzsl.py merge`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, filenames=np.asarray([str(n) for n in predictions['filenames']]),
                        disease=np.asarray(predictions['disease'], dtype=np.float32))


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    taxonomy = Taxonomy(load_labels())
    stage = 'select' if args.stage == 'select' else 'final'
    data = make_datasets(stage, CLIP_MEAN, CLIP_STD)
    model = CCDNet(taxonomy, device, args.unfreeze, args.dim).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    sampler = CrossCropBatches(data['train'].records, args.batch_size, args.groups, args.seed)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': 'CCD-Net', 'backbone_weights': BACKBONES['bioclip'], 'image_size': 224, 'normalisation': 'CLIP',
              'labels_used': 'disease only', 'batch': f'{sampler.n_random} random (sqrt by disease) + '
              f'{2 * args.groups} bridge + {2 * args.groups} decoy', 'bridge_diseases': len(sampler.bridge_diseases),
              'trainable_parameters': sum(p.numel() for p in trainable), 'train_images': len(data['train']),
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('trainable', config['trainable_parameters'], '| bridge diseases', len(sampler.bridge_diseases), flush=True)

    train_loader = DataLoader(data['train'], batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    encoder = {id(p) for p in model.visual.parameters()}
    optimiser = torch.optim.AdamW([
        {'params': [p for p in trainable if id(p) in encoder], 'lr': args.lr, 'weight_decay': 0.05},
        {'params': [p for p in trainable if id(p) not in encoder], 'lr': args.head_lr, 'weight_decay': 0.05}])
    run_epochs = args.final_epochs if args.stage == 'final' else args.epochs
    steps, warmup = run_epochs * len(train_loader), len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + np.cos(np.pi * min(s, steps) / steps)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema else None
    ema_pairs = ([(e, p) for e, p in zip(ema.parameters(), model.parameters()) if p.requires_grad]
                 if ema is not None else [])
    scorer = lambda: tta(ema if ema is not None else model) if args.tta else (ema or model)  # noqa: E731
    val_loader = eval_loader(data['val_seen'], args.batch_size, args.workers) if args.stage == 'select' else None
    best, best_epoch, stale, step = -1.0, 0, 0, 0
    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow(['epoch', 'train_loss', 'loss_disease', 'loss_cross_crop', 'val_disease_top1_per_class',
                          'seconds'])
        for epoch in range(1, run_epochs + 1):
            start = time.perf_counter()
            model.train()
            sums, count = Counter(), 0
            for batch in train_loader:
                images = batch['image'].to(device, non_blocking=True)
                disease = batch['disease'].to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    z = model.features(images)
                    terms = {'disease': loss_fn(model.disease_head(z), disease)}
                    if args.contrastive > 0:
                        terms['cross_crop'] = args.contrastive * cross_crop_loss(
                            model.proj(z), disease, batch['crop'].to(device), batch['composition'].to(device),
                            args.temperature)
                    loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}; lower the learning rates.')
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                step += 1
                if ema is not None:
                    decay = min(args.ema, (1 + step) / (10 + step))
                    with torch.no_grad():
                        for e, p in ema_pairs:
                            e.lerp_(p.detach(), 1 - decay)
                n = len(images)
                sums['total'] += loss.item() * n
                for k, v in terms.items():
                    sums[k] += v.item() * n
                count += n
            score = None
            if args.stage == 'select':
                model.eval()
                score = val_score(predict(scorer(), val_loader, device), data['val_seen'].records, taxonomy,
                                  'disease')
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save((ema or model).state_dict(), args.out / 'best.pth')
                else:
                    stale += 1
            seconds = time.perf_counter() - start
            means = {k: sums[k] / count for k in sums}
            history.writerow([epoch, means['total'], round(means['disease'], 5), round(means.get('cross_crop', 0.0), 5),
                              score, round(seconds, 1)])
            file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {means["total"]:.4f} disease {means["disease"]:.3f} '
                  f'cross_crop {means.get("cross_crop", 0.0):.3f} val {score} {seconds:.0f}s', flush=True)
            if args.stage == 'select' and stale >= args.patience:
                break
    model.eval()
    target = ema or model
    if args.stage == 'select':
        target.load_state_dict(torch.load(args.out / 'best.pth', map_location=device, weights_only=True))
        save_disease(args.out / 'pseudo.npz', predict(scorer(), eval_loader(data['pseudo'], args.batch_size,
                                                                              args.workers), device))
        summary = {'best_epoch': best_epoch, 'best_val_disease_top1_per_class': best}
    else:
        torch.save(target.state_dict(), args.out / 'final.pth')
        for split in ('test_seen', 'test_unseen'):
            save_disease(args.out / f'{split}.npz',
                         predict(scorer(), eval_loader(data[split], args.batch_size, args.workers), device))
        summary = {'final_epochs': args.final_epochs}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--contrastive', type=float, default=0.5, help='Cross-crop loss weight (0 = off).')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--groups', type=int, default=5, help='Bridge diseases per batch (each adds 4 images).')
    parser.add_argument('--ema', type=float, default=0.998, help='EMA decay for the evaluated weights (0 = off).')
    parser.add_argument('--tta', type=int, default=1, help='Average over horizontal flip at evaluation (1/0).')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--unfreeze', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--final-epochs', type=int)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help='Image encoder blocks.')
    parser.add_argument('--head-lr', type=float, default=1e-3, help='New modules: attention, MLPs, head.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage == 'final' and not args.final_epochs:
        parser.error('--final-epochs is required for the final stage.')
    run(args)


if __name__ == '__main__':
    main()
