r"""CPN: Concept Purification Network. Crop and disease features with the other concept's "noise" removed.

Why: every unseen pair (crop c, disease d) has a disease that was seen on other crops. A disease feature that
also encodes the crop learns shortcuts like "potato leaf -> early blight", and on a cucumber with downy mildew it
drifts towards cucumber's training diseases. Likewise a crop feature that encodes lesions drifts towards crops
that carry the same disease. CPN treats that leakage, and two other sources of noise, as the thing to remove:

  pooling      two learned queries attend over the 196 BioCLIP patch tokens, one for crop evidence (z_c) and one
               for disease evidence (z_d) (the DCI-Net A1 branch, its best variant).
  purify       conditional adversaries with a confusion loss. Among diseases seen on >= 2 crops, an adversary
               (trained on detached features) tries to tell the crop from z_d, choosing only among the crops
               that disease was seen on; the encoder is trained to push the adversary's answer towards uniform
               over those crops (bounded, unlike gradient reversal, which diverged here). The same is done for z_c against the diseases of its
               crop. So within a disease, z_d carries no crop identity (and vice versa) while each branch keeps
               everything needed for its own label. This is invariance P(crop | z_d, d) = P(crop | d), which is
               stronger than DCI-Net's decorrelation of z_c with z_d.
  token drop   during training a random share of patch tokens is hidden from the attention pooling, so the
               branches cannot rely on a few background patches.
  cosine heads optional: L2-normalised features and class weights with a fixed scale, so frequent classes do
               not win through larger logit norms.
  EMA + TTA    predictions come from an exponential moving average of the weights and are averaged over the
               image and its horizontal flip (less estimation noise at test time).

Score for pair (c, d) = log P(c) + log P(d) renormalised over the 333 pairs (evaluate.level_scores); gamma
calibration is applied afterwards by gzsl.py, exactly as for every other model.

  python src\train_cpn.py --stage select --out runs\cpn_select
  python src\train_cpn.py --stage final --final-epochs 10 --out runs\cpn_final
"""
import argparse
from collections import Counter
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy, save_predictions  # noqa: E402
from gzsl import eval_loader, make_datasets, predict, seed_everything, seen_columns, val_score  # noqa: E402
from preprocessing import CLIP_MEAN, CLIP_STD, balanced_sampler  # noqa: E402
from train_clip_compositional import BACKBONES  # noqa: E402


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))


def frozen_mlp(net, z):
    """Run a 2-layer mlp with detached weights: gradients reach z but never the adversary."""
    first, last = net[0], net[2]
    return F.linear(F.gelu(F.linear(z, first.weight.detach(), first.bias.detach())), last.weight.detach(),
                    last.bias.detach())


class CosineHead(nn.Module):
    def __init__(self, dim, classes, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(classes, dim) * 0.02)
        self.scale = scale

    def forward(self, z):
        return self.scale * F.normalize(z, dim=-1) @ F.normalize(self.weight, dim=-1).T


class CPN(nn.Module):
    def __init__(self, taxonomy, device, unfreeze=4, dim=256, cosine=False):
        super().__init__()
        arch, pretrained = BACKBONES['bioclip']
        clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = clip.visual.to(device)
        del clip
        self.visual.requires_grad_(False)
        for block in self.visual.transformer.resblocks[-unfreeze:]:
            block.requires_grad_(True)
        self.visual.ln_post.requires_grad_(True)
        self.visual.proj.requires_grad_(True)
        self.visual.output_tokens = True
        width = self.visual.ln_post.normalized_shape[0]
        n_crop, n_disease = taxonomy.size('crop'), taxonomy.size('disease')
        self.queries = nn.Parameter(torch.randn(2, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.to_c, self.to_d = mlp(width, width, dim), mlp(width, width, dim)
        head = (lambda n: CosineHead(dim, n)) if cosine else (lambda n: nn.Linear(dim, n))
        self.crop_head, self.disease_head = head(n_crop), head(n_disease)
        # adversaries: which disease (of this crop) from z_c; which crop (of this disease) from z_d
        self.adv_c, self.adv_d = mlp(dim, dim, n_disease), mlp(dim, dim, n_crop)
        self.token_drop = 0.0

    def primitives(self, images):
        _, tokens = self.visual(images)
        mask = None
        if self.training and self.token_drop > 0:
            mask = torch.rand(tokens.shape[:2], device=tokens.device) < self.token_drop
            mask[:, 0] &= mask.all(1).logical_not()  # never hide every token of an image
        q = self.queries.unsqueeze(0).expand(len(tokens), -1, -1).to(tokens.dtype)
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=mask, need_weights=False)
        out = self.norm(out)
        return self.to_c(out[:, 0]), self.to_d(out[:, 1])

    def forward(self, images):
        z_c, z_d = self.primitives(images)
        return {'crop': self.crop_head(z_c), 'disease': self.disease_head(z_d)}


def conditional_masks(taxonomy, train_seen, device):
    """valid[d, c]: crop c seen with disease d in training. Rows with < 2 options carry no signal."""
    n_crop, n_disease = taxonomy.size('crop'), taxonomy.size('disease')
    valid = torch.zeros(n_disease, n_crop, dtype=torch.bool)
    for k in np.flatnonzero(train_seen):
        valid[taxonomy.groups['disease'][k], taxonomy.groups['crop'][k]] = True
    return valid.to(device), valid.T.contiguous().to(device)


def masked_ce(logits, options, target=None):
    """Cross-entropy over the allowed options only (rows with < 2 options dropped). target=None: cross-entropy
    to the uniform distribution over the options minus log(#options), i.e. 0 when the adversary is confused."""
    keep = options.sum(1) >= 2
    if not keep.any():
        return logits.sum() * 0.0
    options = options[keep]
    log_p = F.log_softmax(logits[keep].float().masked_fill(~options, -1e4), -1)
    if target is not None:
        return F.nll_loss(log_p, target[keep])
    k = options.sum(1).float()
    return (-(log_p * options).sum(1) / k - k.log()).mean()


@torch.no_grad()
def tta(model):
    def forward(images):
        a, b = model(images), model(torch.flip(images, dims=[3]))
        return {k: F.log_softmax((F.log_softmax(a[k].float(), -1) + F.log_softmax(b[k].float(), -1)) / 2, -1)
                for k in a}
    return forward


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    taxonomy = Taxonomy(load_labels())
    stage = 'select' if args.stage == 'select' else 'final'
    data = make_datasets(stage, CLIP_MEAN, CLIP_STD)
    model = CPN(taxonomy, device, args.unfreeze, args.dim, args.cosine).to(device)
    model.token_drop = args.token_drop
    args.out.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': 'CPN', 'backbone_weights': BACKBONES['bioclip'], 'image_size': 224, 'normalisation': 'CLIP',
              'trainable_parameters': sum(p.numel() for p in trainable), 'train_images': len(data['train']),
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('trainable', config['trainable_parameters'], flush=True)

    crops_of_disease, diseases_of_crop = conditional_masks(taxonomy, seen_columns(taxonomy, stage), device)
    print('purify rows: diseases with >=2 crops', int((crops_of_disease.sum(1) >= 2).sum()),
          '| crops with >=2 diseases', int((diseases_of_crop.sum(1) >= 2).sum()), flush=True)

    sampler = balanced_sampler(data['train'].records, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(data['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
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
    parts = ('crop', 'disease', 'adv_c', 'adv_d', 'conf_c', 'conf_d')
    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow(['epoch', 'train_loss', *[f'loss_{p}' for p in parts], 'adv_scale',
                          'val_composition_top1_per_class', 'seconds'])
        for epoch in range(1, run_epochs + 1):
            start = time.perf_counter()
            model.train()
            sums, count = Counter(), 0
            for batch in train_loader:
                images = batch['image'].to(device, non_blocking=True)
                crop = batch['crop'].to(device, non_blocking=True)
                disease = batch['disease'].to(device, non_blocking=True)
                # DANN ramp 0 -> 1 over training, so the adversaries are informed before the encoder fights them
                scale = args.adv * (2.0 / (1.0 + np.exp(-10.0 * step / steps)) - 1.0)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    z_c, z_d = model.primitives(images)
                    terms = {'crop': loss_fn(model.crop_head(z_c), crop),
                             'disease': loss_fn(model.disease_head(z_d), disease)}
                    if args.adv > 0:
                        opt_c, opt_d = diseases_of_crop[crop], crops_of_disease[disease]
                        # adversaries learn on detached features; the encoder learns to confuse them
                        terms['adv_c'] = masked_ce(model.adv_c(z_c.detach()), opt_c, disease)
                        terms['adv_d'] = masked_ce(model.adv_d(z_d.detach()), opt_d, crop)
                        terms['conf_c'] = scale * masked_ce(frozen_mlp(model.adv_c, z_c), opt_c)
                        terms['conf_d'] = scale * masked_ce(frozen_mlp(model.adv_d, z_d), opt_d)
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
                                  'composition')
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save((ema or model).state_dict(), args.out / 'best.pth')
                else:
                    stale += 1
            seconds = time.perf_counter() - start
            means = {k: sums[k] / count for k in sums}
            history.writerow([epoch, means['total'], *[round(means.get(p, 0.0), 5) for p in parts],
                              round(scale, 4), score, round(seconds, 1)])
            file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {means["total"]:.4f} '
                  + ' '.join(f'{p} {means[p]:.3f}' for p in parts if p in means)
                  + f' adv_scale {scale:.2f} val {score} {seconds:.0f}s', flush=True)
            if args.stage == 'select' and stale >= args.patience:
                break
    model.eval()
    target = ema or model
    if args.stage == 'select':
        target.load_state_dict(torch.load(args.out / 'best.pth', map_location=device, weights_only=True))
        save_predictions(args.out / 'pseudo.npz', **predict(scorer(), eval_loader(data['pseudo'], args.batch_size,
                                                                                    args.workers), device))
        summary = {'best_epoch': best_epoch, 'best_val_composition_top1_per_class': best}
    else:
        torch.save(target.state_dict(), args.out / 'final.pth')
        for split in ('test_seen', 'test_unseen'):
            save_predictions(args.out / f'{split}.npz',
                             **predict(scorer(), eval_loader(data[split], args.batch_size, args.workers), device))
        summary = {'final_epochs': args.final_epochs}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--adv', type=float, default=0.5, help='Max confusion-loss weight (0 = no purification).')
    parser.add_argument('--token-drop', type=float, default=0.0, help='Share of patch tokens hidden in training.')
    parser.add_argument('--cosine', action='store_true', help='Cosine classifier heads instead of linear.')
    parser.add_argument('--ema', type=float, default=0.998, help='EMA decay for the evaluated weights (0 = off).')
    parser.add_argument('--tta', type=int, default=1, help='Average over horizontal flip at evaluation (1/0).')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--unfreeze', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--final-epochs', type=int)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help='Image encoder blocks.')
    parser.add_argument('--head-lr', type=float, default=1e-3, help='New modules: attention, MLPs, heads.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage == 'final' and not args.final_epochs:
        parser.error('--final-epochs is required for the final stage.')
    run(args)


if __name__ == '__main__':
    main()
