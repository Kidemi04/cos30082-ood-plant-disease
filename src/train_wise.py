r"""WiSE-FT for compositional GZSL: zero-shot BioCLIP text classifiers + a fine-tuned image encoder, mixed in weight space.

The model starts as zero-shot BioCLIP. The crop and disease classifiers are the text embeddings of
"a photo of a <crop> leaf." and "a photo of a leaf with <disease>.", and the image encoder is the pretrained one.
Fine-tuning changes the last N image blocks (default 6), ln_post, proj and (with --score factor) the two
text-initialised classifiers. WiSE-FT (Wortsman et al., CVPR 2022) then mixes the two sets of weights:

    theta(alpha) = (1 - alpha) * theta_zero_shot + alpha * theta_fine_tuned

alpha = 0 is zero-shot BioCLIP and alpha = 1 is the plain fine-tuned model. Frozen weights are the same at both ends.
--backbone swaps the pretrained CLIP (BioCLIP, BioCLIP 2, OpenAI, LAION-2B, DataComp); everything else is unchanged.

  --score factor   pair score = log P(crop) + log P(disease) from the two text-initialised classifiers (trainable)
  --score pair     the 333 frozen pair prompts "a photo of a <crop> leaf with <disease>." score each composition;
                   crop-only and disease-only prompts add the primitive losses

Select stage: train without the pseudo-unseen pairs and pick the epoch by seen validation accuracy (as every other
model), then sweep alpha on the pseudo-unseen split; alpha (with gamma) maximises the pseudo H. pseudo.npz holds the
chosen alpha, so gzsl.py calibrate works unchanged.
Final stage: train on all 8,000 images and predict the test splits at --alpha (the select-stage choice). Predictions
at every alpha are also saved, for diagnostics only; they are never used to choose anything.

  python src\train_wise.py --stage select --out runs\wise_select
  python src\train_wise.py --stage final --final-epochs 9 --alpha 0.6 --out runs\wise_final
  python src\train_wise.py --stage final --final-epochs 9 --alpha 0.6 --out runs\wise_final --predict-only
"""
import argparse
import csv
import json
from pathlib import Path
import re
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
from evaluate import Taxonomy, accuracy, align, level_scores, save_predictions, top_k_hits  # noqa: E402
from gzsl import (composition_log_probs, eval_loader, make_datasets, predict, pseudo_protocol,  # noqa: E402
                  seed_everything, seen_columns, split_pseudo, sweep, val_score)
from preprocessing import CLIP_MEAN, CLIP_STD, balanced_sampler  # noqa: E402
from train_clip_compositional import CONTEXT, PromptBank  # noqa: E402

# --backbone: open_clip (architecture, pretrained tag). All use the OpenAI CLIP image mean/std and a standard text tower.
BACKBONES = {'bioclip': ('hf-hub:imageomics/bioclip', None),         # ViT-B/16, OpenAI CLIP tuned on TreeOfLife-10M
             'bioclip2': ('hf-hub:imageomics/bioclip-2', None),      # ViT-L/14, TreeOfLife-200M
             'openai': ('ViT-B-16', 'openai'),                       # BioCLIP's own starting point
             'laion2b': ('ViT-B-16', 'laion2b_s34b_b88k'),
             'datacomp': ('ViT-B-16', 'datacomp_xl_s13b_b90k')}
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.1), 1)
# --tune: which parameters of each unfrozen block train (ln_post and proj always train).
TUNE = {'all': ('',), 'attn': ('attn.',), 'mlp': ('mlp.',), 'norm': ('ln_1.', 'ln_2.')}


class WiseCLIP(nn.Module):
    def __init__(self, taxonomy, unfreeze=6, score='factor', tune='all', backbone='bioclip'):
        super().__init__()
        arch, pretrained = BACKBONES[backbone]
        self.clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.clip.requires_grad_(False)
        self.score = score
        self.bank = PromptBank(open_clip.get_tokenizer(arch), None, taxonomy)
        self.example = self.bank.example
        visual = self.clip.visual
        blocks = visual.transformer.resblocks
        if not 0 <= unfreeze <= len(blocks):
            raise ValueError(f'--unfreeze must be between 0 and {len(blocks)} for {backbone}.')
        for block in blocks[len(blocks) - unfreeze:]:
            for name, p in block.named_parameters():
                p.requires_grad_(name.startswith(TUNE[tune]))
        visual.ln_post.requires_grad_(True)
        visual.proj.requires_grad_(True)

    @torch.no_grad()
    def build_text(self, device, head_init='text'):
        """Encode the prompts once, then drop the text tower: the classifiers are all that is kept of it."""
        c = self.clip
        text = {}
        for name in ('composition', 'crop', 'disease'):
            s = {k: v.to(device) for k, v in self.bank.sets[name].items()}
            x = c.token_embedding(s['ids']) + c.positional_embedding[:CONTEXT]
            x = c.ln_final(c.transformer(x, attn_mask=c.attn_mask[:CONTEXT, :CONTEXT]))
            x = x[torch.arange(len(x), device=device), s['eot']]
            x = c.text_projection(x) if isinstance(c.text_projection, nn.Linear) else x @ c.text_projection
            text[name] = F.normalize(x.float(), dim=-1)
        if self.score == 'factor':
            if head_init == 'random':  # control: unit-norm random rows, same scale as the text embeddings
                text = {k: F.normalize(torch.randn_like(text[k]), dim=-1) for k in ('crop', 'disease')}
            self.crop_head = nn.Parameter(text['crop'].clone())
            self.disease_head = nn.Parameter(text['disease'].clone())
        else:
            for name, value in text.items():
                self.register_buffer(f'{name}_text', value)
        for name in ('transformer', 'token_embedding', 'ln_final'):
            setattr(c, name, None)
        del self.bank

    def forward(self, images, train=False):
        image = F.normalize(self.clip.encode_image(images), dim=-1)
        scale = self.clip.logit_scale.exp()
        if self.score == 'factor':
            return {'crop': scale * image @ self.crop_head.T.to(image.dtype),
                    'disease': scale * image @ self.disease_head.T.to(image.dtype)}
        out = {'composition': scale * image @ self.composition_text.T.to(image.dtype)}
        if train:
            out['crop'] = scale * image @ self.crop_text.T.to(image.dtype)
            out['disease'] = scale * image @ self.disease_text.T.to(image.dtype)
        return out


def trainable_state(model):
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def set_alpha(model, zero, tuned, alpha):
    for n, p in model.named_parameters():
        if n in zero:
            p.copy_(((1 - alpha) * zero[n] + alpha * tuned[n]).to(p.device, p.dtype))


def tta(model):
    def forward(images):
        a, b = model(images), model(torch.flip(images, dims=[3]))
        return {k: F.log_softmax((F.log_softmax(a[k].float(), -1) + F.log_softmax(b[k].float(), -1)) / 2, -1)
                for k in a}
    return forward


def per_class_top1(scores, records, level):
    targets = np.array([r[level + '_id'] for r in records])
    return 100 * accuracy(top_k_hits(scores, targets, 1), targets)['per_class']


def param_groups(model, args):
    """AdamW groups. Heads: --head-lr, no decay. Encoder block i of n: --lr * layer_decay ** (n - i); ln_post/proj: --lr."""
    n = len(model.clip.visual.transformer.resblocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('_head'):
            key, lr, decay = 'heads', args.head_lr, 0.0
        else:
            m = re.match(r'clip\.visual\.transformer\.resblocks\.(\d+)\.', name)
            depth = int(m.group(1)) + 1 if m else n
            key, lr, decay = f'block{depth}', args.lr * args.layer_decay ** (n - depth), args.weight_decay
        groups.setdefault(key, {'params': [], 'lr': lr, 'weight_decay': decay})['params'].append(p)
    return list(groups.values())


def alpha_sweep(model, zero, tuned, data, args, device, taxonomy):
    """Pseudo-unseen H, AUSUC and factor accuracy at every alpha. Returns rows and the predictions per alpha."""
    protocol = pseudo_protocol()
    seen = seen_columns(taxonomy, 'select')
    loader = eval_loader(data['pseudo'], args.batch_size, args.workers)
    rows, kept = [], {}
    for alpha in args.alphas:
        set_alpha(model, zero, tuned, float(alpha))
        model.eval()
        p = predict(tta(model), loader, device)
        kept[float(alpha)] = p
        parts = split_pseudo(p, protocol)
        s_lp = composition_log_probs(parts['val_seen'], protocol['val_seen'], taxonomy)
        u_lp = composition_log_probs(parts['pseudo_unseen'], protocol['pseudo_unseen'], taxonomy)
        curve, best, ausuc = sweep(s_lp, protocol['val_seen'], u_lp, protocol['pseudo_unseen'], seen)
        lv = level_scores(align(parts['pseudo_unseen'], protocol['pseudo_unseen']), taxonomy)
        row = {'alpha': float(alpha), 'gamma': best['gamma'], 'seen': 100 * best['seen'],
               'unseen': 100 * best['unseen'], 'H': 100 * best['harmonic'], 'H_gamma0': 100 * curve[0]['harmonic'],
               'ausuc': 100 * ausuc,
               'new_pair_crop_top1': per_class_top1(lv['crop'], protocol['pseudo_unseen'], 'crop'),
               'new_pair_disease_top1': per_class_top1(lv['disease'], protocol['pseudo_unseen'], 'disease')}
        rows.append(row)
        print(' '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()), flush=True)
    return rows, kept


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    taxonomy = Taxonomy(load_labels())
    data = make_datasets(args.stage, CLIP_MEAN, CLIP_STD)
    model = WiseCLIP(taxonomy, args.unfreeze, args.score, args.tune, args.backbone).to(device)
    model.build_text(device, args.head_init)
    zero = trainable_state(model)
    args.out.mkdir(parents=True, exist_ok=True)
    encoder = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith('_head')]
    heads = [p for n, p in model.named_parameters() if n.endswith('_head')]
    config = {**{k: str(v) if isinstance(v, Path) else v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in vars(args).items()},
              'backbone_weights': BACKBONES[args.backbone], 'normalisation': 'CLIP', 'prompt_example': model.example,
              'trainable_parameters': sum(p.numel() for p in encoder + heads),
              'train_images': len(data['train']), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    if not args.predict_only:  # keep the config of the run that trained final.pth
        (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('Prompts:', model.example, '| trainable', config['trainable_parameters'], flush=True)

    if args.predict_only:
        tuned = torch.load(args.out / 'final.pth', weights_only=True)
        print(f'Predicting from {args.out / "final.pth"} (training skipped)', flush=True)
    else:
        best, best_epoch = fit(model, data, args, device, taxonomy)
    # fit() has returned, so its persistent training workers are gone before the evaluation workers start.

    if args.stage == 'select':
        tuned = torch.load(args.out / 'best.pth', weights_only=True)
        rows, kept = alpha_sweep(model, zero, tuned, data, args, device, taxonomy)
        choice = max(rows, key=lambda r: (round(r['H'], 4), r['alpha']))
        for alpha, p in kept.items():
            save_predictions(args.out / f'pseudo_a{alpha:.1f}.npz', **p)
        save_predictions(args.out / 'pseudo.npz', **kept[choice['alpha']])
        summary = {'best_epoch': best_epoch, 'best_val_composition_top1_per_class': best,
                   'alpha': choice['alpha'], 'alpha_curve': rows}
    else:
        if not args.predict_only:
            tuned = trainable_state(model)
            torch.save(tuned, args.out / 'final.pth')
        loaders = {split: eval_loader(data[split], args.batch_size, args.workers)
                   for split in ('test_seen', 'test_unseen')}
        # The chosen alpha goes first: an interrupted run then still leaves the predictions that matter.
        for alpha in sorted(args.alphas, key=lambda a: abs(a - args.alpha) > 1e-6):
            set_alpha(model, zero, tuned, float(alpha))
            model.eval()
            for split, loader in loaders.items():
                p = predict(tta(model), loader, device)
                save_predictions(args.out / f'{split}_a{alpha:.1f}.npz', **p)
                if abs(alpha - args.alpha) < 1e-6:
                    save_predictions(args.out / f'{split}.npz', **p)
            print(f'predicted test splits at alpha {alpha:.1f}', flush=True)
        summary = {'final_epochs': args.final_epochs, 'alpha': args.alpha}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print({k: v for k, v in summary.items() if k != 'alpha_curve'})


def fit(model, data, args, device, taxonomy):
    """Train in place. Select stage: keeps the best seen-validation epoch in best.pth. Returns (best, best_epoch)."""
    train_seen = torch.tensor(np.flatnonzero(seen_columns(taxonomy, args.stage)), device=device)
    remap = torch.full((taxonomy.size('composition'),), -1, dtype=torch.long, device=device)
    remap[train_seen] = torch.arange(len(train_seen), device=device)
    sampler = balanced_sampler(data['train'].records, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(data['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    optimiser = torch.optim.AdamW(param_groups(model, args))
    run_epochs = args.final_epochs if args.stage == 'final' else args.epochs
    steps, warmup = run_epochs * len(train_loader), len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + np.cos(np.pi * min(s, steps) / steps)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    # Persistent workers: respawning them every epoch costs about as much as the validation itself on Windows.
    val_loader = DataLoader(data['val_seen'], batch_size=args.batch_size * 2, num_workers=args.workers,
                            pin_memory=True, persistent_workers=True) if args.stage == 'select' else None
    best, best_epoch, stale = -1.0, 0, 0
    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        history = csv.writer(file)
        history.writerow(['epoch', 'train_loss', 'val_composition_top1_per_class', 'seconds'])
        for epoch in range(1, run_epochs + 1):
            start = time.perf_counter()
            model.train()
            total, count = 0.0, 0
            for batch in train_loader:
                images = batch['image'].to(device, non_blocking=True)
                crop = batch['crop'].to(device, non_blocking=True)
                disease = batch['disease'].to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images, train=True)
                    loss = loss_fn(logits['crop'], crop) + args.disease_weight * loss_fn(logits['disease'], disease)
                    if args.score == 'pair':
                        target = remap[batch['composition'].to(device, non_blocking=True)]
                        loss = loss_fn(logits['composition'][:, train_seen], target) + args.primitive_weight * loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}; lower the learning rates.')
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_([p for g in optimiser.param_groups for p in g['params']], 1.0)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                total += loss.item() * len(images)
                count += len(images)
            score = None
            if args.stage == 'select':
                model.eval()
                score = val_score(predict(tta(model), val_loader, device), data['val_seen'].records, taxonomy,
                                  'composition')
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save(trainable_state(model), args.out / 'best.pth')
                else:
                    stale += 1
            seconds = time.perf_counter() - start
            history.writerow([epoch, total / count, score, round(seconds, 1)])
            file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {total / count:.4f} val {score} {seconds:.0f}s', flush=True)
            if args.stage == 'select' and stale >= args.patience:
                break
    return best, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--score', choices=('factor', 'pair'), default='factor')
    parser.add_argument('--backbone', choices=tuple(BACKBONES), default='bioclip', help='Pretrained CLIP weights.')
    parser.add_argument('--unfreeze', type=int, default=6, help='Final image-encoder blocks to fine-tune.')
    parser.add_argument('--tune', choices=tuple(TUNE), default='all',
                        help='Parts of each unfrozen block that train: all, attn, mlp or norm (LayerNorms only).')
    parser.add_argument('--head-init', choices=('text', 'random'), default='text', help='--score factor only.')
    parser.add_argument('--layer-decay', type=float, default=1.0,
                        help='Block i of n trains at lr * layer_decay ** (n - i); 1.0 = one learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Image encoder (heads use none).')
    parser.add_argument('--disease-weight', type=float, default=1.0, help='Loss = crop CE + this * disease CE.')
    parser.add_argument('--alphas', type=float, nargs='+', help='WiSE-FT alphas to evaluate (default 0.0 ... 1.0).')
    parser.add_argument('--primitive-weight', type=float, default=1.0, help='--score pair only.')
    parser.add_argument('--alpha', type=float, help='Final stage: WiSE-FT mixing weight chosen in the select stage.')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--final-epochs', type=int)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help='Image encoder.')
    parser.add_argument('--head-lr', type=float, default=1e-5, help='Text-initialised classifiers (--score factor).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--predict-only', action='store_true',
                        help='Final stage: skip training and predict from <out>/final.pth (resume an interrupted run).')
    args = parser.parse_args()
    if args.stage == 'final' and (not args.final_epochs or args.alpha is None):
        parser.error('--final-epochs and --alpha are required for the final stage.')
    if args.predict_only and (args.stage != 'final' or not (args.out / 'final.pth').is_file()):
        parser.error('--predict-only needs --stage final and an existing <out>/final.pth.')
    args.alphas = np.unique(np.round(np.array(args.alphas), 1)) if args.alphas else ALPHAS
    if args.alpha is not None and not np.isclose(args.alphas, args.alpha).any():
        parser.error(f'--alpha must be one of {args.alphas.tolist()}.')
    run(args)


if __name__ == '__main__':
    main()
