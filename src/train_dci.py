r"""DCI-Net: Disentangle, Compose, Imagine. A compositional model for unseen crop-disease pairs.

Built on the BioCLIP image encoder (last blocks fine-tuned, as in the heads control), with four additions:

  (2) dual-query attention  two learned queries attend over the 196 patch tokens, one gathering crop evidence
                            (z_c) and one gathering disease evidence (z_d); each feeds its own linear head.
  (5) disentangle           a cross-correlation penalty between batch-standardised z_c and z_d, so the two
                            branches do not carry the same information (e.g. "early blight looks like potato").
  (3) compose               a composer MLP turns (z_c, z_d) into a pair feature h. Each of the 333 pairs gets a
                            template generated from the frozen BioCLIP text embeddings of its crop and disease
                            names, so unseen pairs have templates without ever being trained on.
  (1) imagine               memory banks keep recent z_c per crop and z_d per disease. Every step, crop and
                            disease features from different images are composed into synthetic pair features
                            for pairs with no training images (and for rare seen pairs, so "synthetic" never
                            means "unseen" to the classifier), trained with the same pair loss.

Only training images and the names of the 333 pairs are used; no unseen image is touched. In the select stage
the 10 pseudo-unseen pairs are among the pairs being imagined, so gamma is chosen exactly as for every model.

Score for pair (c, d) = log P_pair(c, d) + log P(c) + log P(d); gamma calibration is applied afterwards by gzsl.py.

Ablations (see the flags):
  --pool cls --pair none --imagine 0 --decor 0   split projectors on the pooled vector (heads-model reference)
  --pair none --imagine 0 --decor 0               A1: dual-query attention only
  --imagine 0 --decor 0                           A2: + composer and text templates
  (defaults)                                      A3/full: + imagine + disentangle
  --decor 0                                       A4: full without disentangling
  --templates learned                             A5: full with random learned primitive embeddings, no text

  python src\train_dci.py --stage select --out runs\dci_select
  python src\train_dci.py --stage final --final-epochs 12 --out runs\dci_final
"""
import argparse
from collections import Counter
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
from train_clip_compositional import BACKBONES, words  # noqa: E402


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))


@torch.no_grad()
def primitive_text(clip, tokenizer, taxonomy, device):
    """Frozen, L2-normalised BioCLIP text embeddings for every crop and every disease name."""
    crops = [f'a photo of a {words(n)} leaf.' for n in taxonomy.names['crop']]
    diseases = ['a photo of a healthy leaf.' if n == 'healthy' else f'a photo of a leaf with {words(n)}.'
                for n in taxonomy.names['disease']]
    encode = lambda texts: F.normalize(clip.encode_text(tokenizer(texts).to(device)).float(), dim=-1)  # noqa: E731
    return encode(crops), encode(diseases)


class DCINet(nn.Module):
    def __init__(self, taxonomy, device, pool='dual', pair='text', templates='text', unfreeze=4, dim=256):
        super().__init__()
        arch, pretrained = BACKBONES['bioclip']
        clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        clip = clip.to(device).eval()
        text_c, text_d = primitive_text(clip, open_clip.get_tokenizer(arch), taxonomy, device)
        self.visual = clip.visual  # the text tower is only needed once, for the frozen primitive embeddings
        del clip
        self.visual.requires_grad_(False)
        for block in self.visual.transformer.resblocks[-unfreeze:]:
            block.requires_grad_(True)
        self.visual.ln_post.requires_grad_(True)
        self.visual.proj.requires_grad_(True)
        self.visual.output_tokens = True
        self.pool, self.pair = pool, pair
        width, out_dim = self.visual.ln_post.normalized_shape[0], self.visual.output_dim
        if pool == 'dual':
            self.queries = nn.Parameter(torch.randn(2, width) * 0.02)
            self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
            self.norm = nn.LayerNorm(width)
            self.to_c, self.to_d = mlp(width, width, dim), mlp(width, width, dim)
        else:
            self.to_c, self.to_d = mlp(out_dim, out_dim, dim), mlp(out_dim, out_dim, dim)
        self.crop_head = nn.Linear(dim, taxonomy.size('crop'))
        self.disease_head = nn.Linear(dim, taxonomy.size('disease'))
        self.register_buffer('pair_crop', torch.as_tensor(taxonomy.groups['crop']), persistent=False)
        self.register_buffer('pair_disease', torch.as_tensor(taxonomy.groups['disease']), persistent=False)
        if pair == 'text':
            self.composer = mlp(2 * dim, 2 * dim, dim)
            if templates == 'text':
                self.register_buffer('prim_c', text_c, persistent=False)
                self.register_buffer('prim_d', text_d, persistent=False)
            else:  # ablation: same template generator, random learned primitive embeddings instead of words
                self.prim_c = nn.Parameter(F.normalize(torch.randn_like(text_c), dim=-1))
                self.prim_d = nn.Parameter(F.normalize(torch.randn_like(text_d), dim=-1))
            self.template = mlp(2 * text_c.shape[1], 2 * dim, dim)
            self.log_scale = nn.Parameter(torch.tensor(np.log(16.0), dtype=torch.float32))

    def primitives(self, images):
        """Crop and disease features (z_c, z_d) and the attention weights over patches (dual pooling)."""
        pooled, tokens = self.visual(images)
        if self.pool == 'cls':
            return self.to_c(pooled), self.to_d(pooled), None
        q = self.queries.unsqueeze(0).expand(len(tokens), -1, -1).to(tokens.dtype)
        out, weights = self.attn(q, tokens, tokens, need_weights=True)
        out = self.norm(out)
        return self.to_c(out[:, 0]), self.to_d(out[:, 1]), weights

    def pair_logits(self, z_c, z_d):
        h = self.composer(torch.cat([z_c, z_d], -1)) + z_c + z_d  # learned fusion plus a residual sum
        templates = self.template(torch.cat([self.prim_c[self.pair_crop], self.prim_d[self.pair_disease]], -1))
        return self.log_scale.exp() * F.normalize(h, dim=-1) @ F.normalize(templates, dim=-1).T

    def forward(self, images):
        z_c, z_d, _ = self.primitives(images)
        out = {'crop': self.crop_head(z_c), 'disease': self.disease_head(z_d)}
        if self.pair == 'text':
            pair = F.log_softmax(self.pair_logits(z_c, z_d).float(), -1)
            out['composition'] = (pair + F.log_softmax(out['crop'].float(), -1)[:, self.pair_crop]
                                  + F.log_softmax(out['disease'].float(), -1)[:, self.pair_disease])
        return out, z_c, z_d


def cross_correlation(z_c, z_d):
    """Mean squared cross-correlation between batch-standardised crop and disease features."""
    z_c = (z_c - z_c.mean(0)) / (z_c.std(0) + 1e-5)
    z_d = (z_d - z_d.mean(0)) / (z_d.std(0) + 1e-5)
    return ((z_c.T @ z_d) / len(z_c)).pow(2).mean()


class Banks:
    """FIFO memory of recent (detached) crop features per crop and disease features per disease."""

    def __init__(self, n_crop, n_disease, dim, size, device):
        self.size = size
        self.c = torch.zeros(n_crop, size, dim, device=device)
        self.d = torch.zeros(n_disease, size, dim, device=device)
        self.nc = torch.zeros(n_crop, dtype=torch.long, device=device)
        self.nd = torch.zeros(n_disease, dtype=torch.long, device=device)

    @torch.no_grad()
    def push(self, z_c, crops, z_d, diseases):
        for bank, count, z, labels in ((self.c, self.nc, z_c, crops), (self.d, self.nd, z_d, diseases)):
            for feature, label in zip(z.detach().float(), labels.tolist()):
                bank[label, count[label] % self.size] = feature
                count[label] += 1

    def draw(self, crops, diseases, generator):
        """One stored feature per requested crop and disease; mask marks pairs whose banks are non-empty."""
        fc, fd = self.nc[crops].clamp(max=self.size), self.nd[diseases].clamp(max=self.size)
        ok = (fc > 0) & (fd > 0)
        u = torch.rand(len(crops), 2, device=crops.device, generator=generator)
        sc = (u[:, 0] * fc.clamp(min=1)).long().clamp(max=self.size - 1)
        sd = (u[:, 1] * fd.clamp(min=1)).long().clamp(max=self.size - 1)
        return self.c[crops, sc], self.d[diseases, sd], ok


def imagine_targets(taxonomy, train_seen, counts, device):
    """Pairs to synthesise: every pair without training images, and seen pairs weighted towards rare ones."""
    unseen = torch.as_tensor(np.flatnonzero(~train_seen), device=device)
    seen = np.flatnonzero(train_seen)
    weights = np.array([counts.get(int(k), 1) for k in seen], dtype=np.float64) ** -0.5
    return unseen, torch.as_tensor(seen, device=device), torch.as_tensor(weights / weights.sum(), device=device)


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    taxonomy = Taxonomy(load_labels())
    data = make_datasets('select' if args.stage == 'select' else 'final', CLIP_MEAN, CLIP_STD)
    model = DCINet(taxonomy, device, args.pool, args.pair, args.templates, args.unfreeze, args.dim).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    use_pair = args.pair == 'text'
    imagine = use_pair and args.imagine > 0
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': 'DCI-Net', 'backbone_weights': BACKBONES['bioclip'], 'image_size': 224,
              'normalisation': 'CLIP', 'trainable_parameters': sum(p.numel() for p in trainable),
              'train_images': len(data['train']), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('trainable', config['trainable_parameters'], flush=True)

    train_seen = seen_columns(taxonomy, 'select' if args.stage == 'select' else 'final')
    counts = Counter(r['composition_id'] for r in data['train'].records)
    unseen_ids, seen_ids, seen_weights = imagine_targets(taxonomy, train_seen, counts, device)
    banks = Banks(taxonomy.size('crop'), taxonomy.size('disease'), args.dim, args.bank, device)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    sampler = balanced_sampler(data['train'].records, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(data['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    encoder = {id(p) for p in model.visual.parameters()}
    no_decay = {id(model.log_scale)} if use_pair else set()
    groups = [{'params': [p for p in trainable if id(p) in encoder], 'lr': args.lr, 'weight_decay': 0.05},
              {'params': [p for p in trainable if id(p) not in encoder and id(p) not in no_decay],
               'lr': args.head_lr, 'weight_decay': 0.05},
              {'params': [p for p in trainable if id(p) in no_decay], 'lr': args.head_lr, 'weight_decay': 0.0}]
    optimiser = torch.optim.AdamW([g for g in groups if g['params']])
    run_epochs = args.final_epochs if args.stage == 'final' else args.epochs
    steps, warmup = run_epochs * len(train_loader), len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + np.cos(np.pi * min(s, steps) / steps)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    val_loader = eval_loader(data['val_seen'], args.batch_size, args.workers) if args.stage == 'select' else None
    outputs = lambda images: model(images)[0]  # noqa: E731
    best, best_epoch, stale = -1.0, 0, 0
    parts = ('crop', 'disease', 'pair', 'imagine', 'decor')
    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow(['epoch', 'train_loss', *[f'loss_{p}' for p in parts], 'val_composition_top1_per_class',
                          'seconds'])
        for epoch in range(1, run_epochs + 1):
            start = time.perf_counter()
            model.train()
            sums, count = Counter(), 0
            for batch in train_loader:
                images = batch['image'].to(device, non_blocking=True)
                crop = batch['crop'].to(device, non_blocking=True)
                disease = batch['disease'].to(device, non_blocking=True)
                composition = batch['composition'].to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    z_c, z_d, _ = model.primitives(images)
                    terms = {'crop': loss_fn(model.crop_head(z_c), crop),
                             'disease': loss_fn(model.disease_head(z_d), disease)}
                    if use_pair:
                        terms['pair'] = loss_fn(model.pair_logits(z_c, z_d), composition)
                    if args.decor:
                        terms['decor'] = args.decor * cross_correlation(z_c.float(), z_d.float())
                    if imagine and epoch > args.imagine_after:
                        pairs = torch.cat([
                            unseen_ids[torch.randint(len(unseen_ids), (args.syn_unseen,), device=device,
                                                     generator=rng)],
                            seen_ids[torch.multinomial(seen_weights, args.syn_seen, replacement=True,
                                                       generator=rng)]])
                        bank_c, bank_d, ok = banks.draw(model.pair_crop[pairs], model.pair_disease[pairs], rng)
                        if ok.any():
                            logits = model.pair_logits(bank_c[ok].to(z_c.dtype), bank_d[ok].to(z_d.dtype))
                            terms['imagine'] = args.imagine * loss_fn(logits, pairs[ok])
                    loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}; lower the learning rates.')
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                if imagine:
                    banks.push(z_c, crop, z_d, disease)
                n = len(images)
                sums['total'] += loss.item() * n
                for k, v in terms.items():
                    sums[k] += v.item() * n
                count += n
            score = None
            if args.stage == 'select':
                model.eval()
                score = val_score(predict(outputs, val_loader, device), data['val_seen'].records, taxonomy,
                                  'composition')
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save(model.state_dict(), args.out / 'best.pth')
                else:
                    stale += 1
            seconds = time.perf_counter() - start
            means = {k: sums[k] / count for k in sums}
            history.writerow([epoch, means['total'], *[round(means.get(p, 0.0), 5) for p in parts], score,
                              round(seconds, 1)])
            file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {means["total"]:.4f} '
                  + ' '.join(f'{p} {means[p]:.3f}' for p in parts if p in means)
                  + f' val {score} {seconds:.0f}s', flush=True)
            if args.stage == 'select' and stale >= args.patience:
                break
    model.eval()
    if args.stage == 'select':
        model.load_state_dict(torch.load(args.out / 'best.pth', map_location=device, weights_only=True))
        save_predictions(args.out / 'pseudo.npz', **predict(outputs, eval_loader(data['pseudo'], args.batch_size,
                                                                                   args.workers), device))
        summary = {'best_epoch': best_epoch, 'best_val_composition_top1_per_class': best}
    else:
        torch.save(model.state_dict(), args.out / 'final.pth')
        for split in ('test_seen', 'test_unseen'):
            save_predictions(args.out / f'{split}.npz',
                             **predict(outputs, eval_loader(data[split], args.batch_size, args.workers), device))
        summary = {'final_epochs': args.final_epochs}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pool', choices=('dual', 'cls'), default='dual', help='Dual-query attention or pooled.')
    parser.add_argument('--pair', choices=('text', 'none'), default='text', help='Composer + pair templates.')
    parser.add_argument('--templates', choices=('text', 'learned'), default='text')
    parser.add_argument('--imagine', type=float, default=1.0, help='Weight of the synthetic-pair loss (0 = off).')
    parser.add_argument('--imagine-after', type=int, default=1, help='Epochs before imagining (fills the banks).')
    parser.add_argument('--syn-unseen', type=int, default=32, help='Synthetic pairs per step without images.')
    parser.add_argument('--syn-seen', type=int, default=32, help='Synthetic seen pairs per step (rare-weighted).')
    parser.add_argument('--bank', type=int, default=8, help='Features kept per crop and per disease.')
    parser.add_argument('--decor', type=float, default=5.0, help='Weight of the z_c / z_d decorrelation loss.')
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
