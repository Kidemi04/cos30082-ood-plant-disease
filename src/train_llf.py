r"""LLF-Net: Leaf-Lesion Factorised Network. Crop from the whole leaf, disease from what is abnormal on it.

Built in steps; each flag adds one part so every part can be ablated (see the design page).

  backbone   BioCLIP ViT-B/16. With --adapters (default) every original weight is frozen and a small
             bottleneck adapter (768 -> 64 -> 768, zero-initialised output) is added after blocks 7-12.
             --adapters 0 falls back to CPN's recipe (fine-tune the last --unfreeze blocks).
  [A]        adapters: the layer probe showed fine-tuning pushed the right new-pair disease down the ranking
             (frozen Top-5 41.9 vs fine-tuned 35.4).
  taps       the disease branch reads tokens from blocks --taps (default 9,11,12): disease information
             peaks at blocks 9-11, while the crop shortcut keeps growing to block 12.
  [C]        --source: a crop query and a source query attend over block 12. The source head predicts which of
             the source datasets (filename prefix) the photo came from, giving photo style its own slot.
             Neither feeds the disease head. Source labels are a training aid only.
  losses     --con-d: cross-crop (+ cross-source) contrastive on z_d; --con-c: cross-disease contrastive on
             z_c; --decor: decorrelation of z_d against z_c and z_s. --factor-batch builds batches with
             bridges, crop decoys and style decoys for these losses.
  [B]        --lesion: each patch gets an abnormality score, 1 - its best cosine match among healthy-leaf patches
             of the same crop (true crop in training, predicted crop at test), taken at a frozen block
             (--score-block 6, before the first adapter). The standardised score is added, times a learned
             weight, to the disease query's attention logits, so the query leans towards abnormal patches.
             --lesion-layers 2 adds transformer layers (width 384) over the tap tokens first.
             Step-0 check (frozen probe): pooling the 25% most abnormal patches raised new-pair disease Top-1
             18.1 -> 19.6 at block 11; subtracting a healthy-leaf average instead made it worse (10.6), so
             subtraction was dropped.

Score for pair (c, d) = log P(c) + log P(d), renormalised over the 333 pairs; gamma calibration by gzsl.py.

  python src\train_llf.py --stage select --out runs\llf_s2_select
  set GZSL_PSEUDO_MANIFEST=dataset\metadata\pseudo_unseen_dev.csv   (development decisions)
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
from dataset_access import load_labels, load_split  # noqa: E402
from evaluate import Taxonomy, save_predictions  # noqa: E402
from gzsl import eval_loader, make_datasets, predict, seed_everything, val_score  # noqa: E402
from preprocessing import CLIP_MEAN, CLIP_STD, balanced_sampler  # noqa: E402
from train_clip_compositional import BACKBONES  # noqa: E402
from train_cpn import mlp, tta  # noqa: E402
from train_dci import cross_correlation  # noqa: E402

N_PATCH = 196


def source_of(filename):
    return str(filename).split('_')[0]


def source_index():
    """All source prefixes of the official training split, in a fixed order."""
    return {s: i for i, s in enumerate(sorted({source_of(r['filename']) for r in load_split('train')}))}


# ---------------------------------------------------------------- batches

class FactorBatches(Sampler):
    """n_random images (sqrt by disease) + groups x [2 bridges, 2 crop decoys, 2 style decoys].

    Bridges: one disease on two different crops, from different sources when possible. Crop decoy: same crop and
    (when possible) same source as its bridge, different disease. Style decoy: same source as its bridge,
    different crop and different disease.
    """

    def __init__(self, records, batch_size=32, groups=4, seed=42, power=0.5):
        self.rng = np.random.default_rng(seed)
        self.groups, self.n_random = groups, batch_size - 6 * groups
        if self.n_random < 0:
            raise ValueError('batch_size must be at least 6 x groups.')
        self.steps, self.n = len(records) // batch_size, len(records)
        self.crop = np.array([r['crop_id'] for r in records])
        self.dis = np.array([r['disease_id'] for r in records])
        self.src = np.array([source_of(r['filename']) for r in records])
        counts = Counter(self.dis.tolist())
        w = np.array([counts[d] ** -power for d in self.dis.tolist()])
        self.weights = w / w.sum()
        self.by_pair, self.by_crop, self.by_src = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(self.n):
            self.by_pair[(self.crop[i], self.dis[i])].append(i)
            self.by_crop[self.crop[i]].append(i)
            self.by_src[self.src[i]].append(i)
        crops_of = defaultdict(set)
        for c, d in self.by_pair:
            crops_of[d].add(c)
        self.crops_of = {d: sorted(cs) for d, cs in crops_of.items()}
        self.bridge = sorted(d for d, cs in self.crops_of.items() if len(cs) >= 2)

    def __len__(self):
        return self.steps

    def any(self):
        return int(self.rng.choice(self.n, p=self.weights))

    def pick(self, pool, ok):
        good = [i for i in pool if ok(i)]
        return int(self.rng.choice(good)) if good else None

    def __iter__(self):
        for _ in range(self.steps):
            batch = self.rng.choice(self.n, self.n_random, replace=True, p=self.weights).tolist()
            for d in self.rng.choice(self.bridge, self.groups, replace=False):
                c1, c2 = self.rng.choice(self.crops_of[d], 2, replace=False)
                b1 = int(self.rng.choice(self.by_pair[(c1, d)]))
                b2 = self.pick(self.by_pair[(c2, d)], lambda i: self.src[i] != self.src[b1])
                b2 = b2 if b2 is not None else int(self.rng.choice(self.by_pair[(c2, d)]))
                batch += [b1, b2]
                for b in (b1, b2):
                    cd = self.pick(self.by_crop[self.crop[b]], lambda i: self.dis[i] != d and self.src[i] == self.src[b])
                    if cd is None:
                        cd = self.pick(self.by_crop[self.crop[b]], lambda i: self.dis[i] != d)
                    sd = self.pick(self.by_src[self.src[b]], lambda i: self.dis[i] != d and self.crop[i] != self.crop[b])
                    batch += [cd if cd is not None else self.any(), sd if sd is not None else self.any()]
            yield batch


# ---------------------------------------------------------------- model

class Adapter(nn.Module):
    def __init__(self, width, dim):
        super().__init__()
        self.down, self.up = nn.Linear(width, dim), nn.Linear(dim, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


def tap_hook(module, inputs, output):
    """Runs the block's adapter (if any) and keeps its output for the taps. Generic, so deepcopy (EMA) is safe."""
    if hasattr(module, 'adapter'):
        output = module.adapter(output)
    if getattr(module, 'keep', False):
        module.kept = output
    return output


class LLFNet(nn.Module):
    def __init__(self, taxonomy, n_sources, device, args):
        super().__init__()
        arch, pretrained = BACKBONES['bioclip']
        clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = clip.visual.to(device)
        del clip
        self.visual.requires_grad_(False)
        self.visual.output_tokens = True
        blocks = self.visual.transformer.resblocks
        width = self.visual.ln_post.normalized_shape[0]
        if args.adapters:
            for i in range(args.adapt_from - 1, len(blocks)):
                blocks[i].add_module('adapter', Adapter(width, args.adapter_dim))
        else:
            for block in blocks[-args.unfreeze:]:
                block.requires_grad_(True)
            self.visual.ln_post.requires_grad_(True)
        self.taps = sorted(args.taps)
        for i, block in enumerate(blocks):
            block.keep = (i + 1) in self.taps
            block.register_forward_hook(tap_hook)
        self.use_source, self.lesion = args.source, args.lesion
        n_crop, n_disease = taxonomy.size('crop'), taxonomy.size('disease')
        # [C] factor queries over block 12 (crop, and source when enabled)
        self.q_factor = nn.Parameter(torch.randn(2 if args.source else 1, width) * 0.02)
        self.attn_f = nn.MultiheadAttention(width, 8, batch_first=True)
        self.norm_f = nn.LayerNorm(width)
        self.to_c = mlp(width, width, args.dim)
        self.crop_head = nn.Linear(args.dim, n_crop)
        if args.source:
            self.to_s = mlp(width, width, args.dim)
            self.source_head = nn.Linear(args.dim, n_sources)
        # disease branch over the taps; [B] adds an abnormality bias and optional extra layers
        self.tap_norm = nn.ModuleList(nn.LayerNorm(width) for _ in self.taps)
        self.tap_emb = nn.Parameter(torch.zeros(len(self.taps), 1, width))
        dd = width
        self.lesion_layers = None
        if args.lesion and args.lesion_layers > 0:
            dd = args.lesion_width
            self.lesion_in = nn.Linear(width, dd)
            layer = nn.TransformerEncoderLayer(dd, 8, 4 * dd, dropout=0.1, activation='gelu', batch_first=True,
                                               norm_first=True)
            self.lesion_layers = nn.TransformerEncoder(layer, args.lesion_layers)
        if args.lesion:
            self.score_block = args.score_block
            blocks[self.score_block - 1].keep = True
            self.lesion_bias = nn.Parameter(torch.tensor(float(args.lesion_bias)))
        self.bank_key = None
        self.q_d = nn.Parameter(torch.randn(1, 1, dd) * 0.02)
        self.attn_d = nn.MultiheadAttention(dd, 8, batch_first=True)
        self.norm_d = nn.LayerNorm(dd)
        self.to_d = mlp(dd, dd, args.dim)
        self.disease_head = nn.Linear(args.dim, n_disease)
        self.proj_d, self.proj_c = mlp(args.dim, args.dim, 128), mlp(args.dim, args.dim, 128)

    def backbone(self, images):
        _, tokens = self.visual(images)
        blocks = self.visual.transformer.resblocks
        kept = {}
        for i, block in enumerate(blocks):
            if block.keep:
                x = block.kept
                block.kept = None
                kept[i + 1] = (x if x.shape[0] == len(images) else x.transpose(0, 1))[:, 1:]  # patch tokens
        score = kept.get(self.score_block) if self.lesion else None
        return tokens, [kept[t] for t in self.taps], score

    def abnormality(self, score_tokens, crop_sel, bank_ids=None):
        """1 - max cosine similarity of each patch to the healthy-leaf bank of the selected crop (frozen block)."""
        bank = BANKS[self.bank_key]
        out = torch.zeros(score_tokens.shape[:2], device=score_tokens.device)
        has = torch.zeros(len(score_tokens), dtype=torch.bool, device=score_tokens.device)
        for j in range(len(score_tokens)):
            c = int(crop_sel[j])
            if c not in bank['feat']:
                continue
            sim = F.normalize(score_tokens[j].float(), dim=-1).half() @ bank['feat'][c].T
            if bank_ids is not None and int(bank_ids[j]) >= 0:
                sim[:, bank['owner'][c] == int(bank_ids[j])] = -1  # never match an image against itself
            out[j] = 1 - sim.float().max(1).values
            has[j] = True
        z = (out - out.mean(1, keepdim=True)) / (out.std(1, keepdim=True) + 1e-4)
        return out, z * has[:, None]

    def forward(self, images, crop_true=None, bank_ids=None, return_all=False):
        tokens, taps, score = self.backbone(images)
        q = self.q_factor.unsqueeze(0).expand(len(tokens), -1, -1).to(tokens.dtype)
        f, _ = self.attn_f(q, tokens, tokens, need_weights=False)
        f = self.norm_f(f)
        z_c = self.to_c(f[:, 0])
        crop_logits = self.crop_head(z_c)
        z_s = self.to_s(f[:, 1]) if self.use_source else None
        kv = torch.cat([self.tap_norm[k](x) + self.tap_emb[k] for k, x in enumerate(taps)], 1)
        if self.lesion_layers is not None:
            kv = self.lesion_layers(self.lesion_in(kv))
        mask, abn = None, None
        if self.lesion:
            crop_sel = crop_true if crop_true is not None else crop_logits.detach().argmax(1)
            with torch.no_grad():
                abn, z = self.abnormality(score, crop_sel, bank_ids)
            bias = (self.lesion_bias * z).repeat(1, len(taps))                       # (N, T*196)
            mask = bias[:, None, :].repeat_interleave(self.attn_d.num_heads, 0).to(kv.dtype)
        qd = self.q_d.expand(len(tokens), -1, -1).to(kv.dtype)
        d, _ = self.attn_d(qd, kv, kv, attn_mask=mask, need_weights=False)
        z_d = self.to_d(self.norm_d(d[:, 0]))
        out = {'crop': crop_logits, 'disease': self.disease_head(z_d)}
        if return_all:
            out.update(z_c=z_c, z_d=z_d, z_s=z_s, abn=abn,
                       source=self.source_head(z_s) if self.use_source else None)
        return out


BANKS = {}  # healthy-leaf banks live outside the model so the EMA copy shares them instead of duplicating them


@torch.no_grad()
def build_bank(model, records, healthy_id, cap, device, batch_size, workers, dataset, seed=42, key='main'):
    """Frozen score-block patch tokens of up to `cap` healthy training photos per crop."""
    rng = np.random.default_rng(seed)
    by_crop = defaultdict(list)
    for r in records:
        if r['disease_id'] == healthy_id:
            by_crop[r['crop_id']].append(r)
    chosen = []
    for c, rs in sorted(by_crop.items()):
        idx = rng.permutation(len(rs))[:cap]
        chosen += [rs[i] for i in sorted(idx)]
    index = {r['filename']: i for i, r in enumerate(chosen)}
    feats, owners = defaultdict(list), defaultdict(list)
    loader = DataLoader(dataset(chosen), batch_size=batch_size * 2, num_workers=workers, pin_memory=True)
    model.eval()
    for b in loader:
        with torch.amp.autocast('cuda'):
            _, _, score = model.backbone(b['image'].to(device, non_blocking=True))
        for j, fn in enumerate(b['filename']):
            c = int(b['crop'][j])
            feats[c].append(F.normalize(score[j].float(), dim=-1).half())
            owners[c].append(torch.full((score.shape[1],), index[fn], device=device))
    BANKS[key] = {'feat': {c: torch.cat(v) for c, v in feats.items()},
                  'owner': {c: torch.cat(v) for c, v in owners.items()}, 'index': index}
    model.bank_key = key
    return len(feats), len(chosen)


# ---------------------------------------------------------------- losses

def supcon(z, positive, exclude, temperature=0.1):
    """Supervised contrastive loss with explicit positive and excluded (neither side) masks."""
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = positive & ~eye
    keep = positive.any(1)
    if not keep.any():
        return sim.sum() * 0.0
    log_prob = sim - torch.logsumexp(sim.masked_fill(exclude | eye, float('-inf')), 1, keepdim=True)
    return (-(log_prob * positive).sum(1)[keep] / positive.sum(1)[keep]).mean()


def factor_losses(out, crop, disease, source, composition, args):
    same_d = disease[:, None] == disease[None]
    same_c = crop[:, None] == crop[None]
    same_s = source[:, None] == source[None]
    same_pair = composition[:, None] == composition[None]
    terms = {}
    if args.con_d > 0:
        pos = same_d & ~same_c & ~same_s
        fallback = same_d & ~same_c
        pos = torch.where(pos.any(1, keepdim=True), pos, fallback)
        terms['con_d'] = args.con_d * supcon(out['_proj_d'], pos, same_pair & ~pos, args.temperature)
    if args.con_c > 0:
        terms['con_c'] = args.con_c * supcon(out['_proj_c'], same_c & ~same_d, same_pair, args.temperature)
    if args.decor > 0:
        terms['decor'] = args.decor * (cross_correlation(out['z_d'].float(), out['z_c'].float()) +
                                       (cross_correlation(out['z_d'].float(), out['z_s'].float())
                                        if out['z_s'] is not None else 0.0))
    return terms


# ---------------------------------------------------------------- training

def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    taxonomy = Taxonomy(load_labels())
    healthy_id = taxonomy.names['disease'].index('healthy')
    stage = 'select' if args.stage == 'select' else 'final'
    data = make_datasets(stage, CLIP_MEAN, CLIP_STD)
    sources = source_index()
    model = LLFNet(taxonomy, len(sources), device, args).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_bank = None
    if args.lesion:
        from preprocessing import DPDDataset, eval_transform
        n_bank = build_bank(model, data['train'].records, healthy_id, args.bank_cap, device, args.batch_size,
                            args.workers, lambda recs: DPDDataset(recs, eval_transform(224, CLIP_MEAN, CLIP_STD), 224))
        print('healthy bank: %d crops, %d photos' % n_bank, flush=True)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': 'LLF-Net', 'backbone_weights': BACKBONES['bioclip'], 'sources': len(sources),
              'healthy_prototype_crops': n_bank, 'trainable_parameters': sum(p.numel() for p in trainable),
              'train_images': len(data['train']), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'pseudo_manifest': str(__import__('gzsl').PSEUDO_MANIFEST)}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('trainable', config['trainable_parameters'], flush=True)

    if args.factor_batch:
        sampler = FactorBatches(data['train'].records, args.batch_size, args.groups, args.seed)
        train_loader = DataLoader(data['train'], batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0)
    else:
        sampler = balanced_sampler(data['train'].records, generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(data['train'], batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                                  drop_last=True)
    encoder = {id(p) for n, p in model.visual.named_parameters() if 'adapter' not in n}
    groups = [{'params': [p for p in trainable if id(p) not in encoder], 'lr': args.head_lr, 'weight_decay': 0.05}]
    enc = [p for p in trainable if id(p) in encoder]
    if enc:
        groups.append({'params': enc, 'lr': args.lr, 'weight_decay': 0.05})
    optimiser = torch.optim.AdamW(groups)
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
    parts = ('crop', 'disease', 'source', 'con_d', 'con_c', 'decor')
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
                source = torch.tensor([sources[source_of(f)] for f in batch['filename']], device=device)
                bank_ids = (torch.tensor([BANKS[model.bank_key]['index'].get(f, -1) for f in batch['filename']])
                            if args.lesion else None)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(images, crop_true=crop, bank_ids=bank_ids, return_all=True)
                    terms = {'crop': loss_fn(out['crop'], crop), 'disease': loss_fn(out['disease'], disease)}
                    if args.source:
                        terms['source'] = args.source_w * loss_fn(out['source'], source)
                    out['_proj_d'] = model.proj_d(out['z_d']) if args.con_d > 0 else None
                    out['_proj_c'] = model.proj_c(out['z_c']) if args.con_c > 0 else None
                    terms.update(factor_losses(out, crop, disease, source, composition, args))
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
                        for eb, b in zip(ema.buffers(), model.buffers()):
                            eb.copy_(b)
                n = len(images)
                sums['total'] += loss.item() * n
                for k, v in terms.items():
                    sums[k] += float(v) * n
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
            history.writerow([epoch, means['total'], *[round(means.get(p, 0.0), 5) for p in parts], score,
                              round(seconds, 1)])
            file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {means["total"]:.4f} '
                  + ' '.join(f'{p} {means[p]:.3f}' for p in parts if p in means)
                  + f' val {score} {seconds:.0f}s', flush=True)
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
    parser.add_argument('--adapters', type=int, default=1, help='1: frozen BioCLIP + adapters. 0: fine-tune.')
    parser.add_argument('--adapt-from', type=int, default=7, help='First block (1-based) that gets an adapter.')
    parser.add_argument('--adapter-dim', type=int, default=64)
    parser.add_argument('--unfreeze', type=int, default=4, help='Blocks fine-tuned when --adapters 0.')
    parser.add_argument('--taps', type=lambda s: [int(x) for x in s.split(',')], default=[9, 11, 12])
    parser.add_argument('--source', type=int, default=0, help='[C] source slot (1/0).')
    parser.add_argument('--source-w', type=float, default=0.5)
    parser.add_argument('--con-d', type=float, default=0.0, help='Cross-crop + cross-source loss on z_d.')
    parser.add_argument('--con-c', type=float, default=0.0, help='Cross-disease loss on z_c.')
    parser.add_argument('--decor', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--factor-batch', type=int, default=0, help='Bridge/decoy batches (1/0).')
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--lesion', type=int, default=0, help='[B] abnormality bias on the disease query (1/0).')
    parser.add_argument('--lesion-layers', type=int, default=0, help='[B] extra transformer layers over the taps.')
    parser.add_argument('--lesion-width', type=int, default=384)
    parser.add_argument('--lesion-bias', type=float, default=2.0, help='Initial weight of the abnormality bias.')
    parser.add_argument('--score-block', type=int, default=6, help='Frozen block used to score abnormality.')
    parser.add_argument('--bank-cap', type=int, default=40, help='Healthy photos per crop in the bank.')
    parser.add_argument('--ema', type=float, default=0.998)
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--final-epochs', type=int)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help='Fine-tuned encoder blocks (--adapters 0).')
    parser.add_argument('--head-lr', type=float, default=1e-3, help='Adapters and all new modules.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage == 'final' and not args.final_epochs:
        parser.error('--final-epochs is required for the final stage.')
    first_trained = args.adapt_from if args.adapters else 13 - args.unfreeze
    if args.lesion and args.score_block >= first_trained:
        parser.error('--score-block must be a frozen block, before the first adapter or fine-tuned block.')
    run(args)


if __name__ == '__main__':
    main()
