r"""Proposed model: compositional vision-language matching with learned crop and disease tokens.

Every composition is the sentence "a photo of a <crop> leaf with <disease>." and is scored by the cosine
similarity between its CLIP text embedding and the image embedding. Each crop and each disease owns a learned
offset that is added to the word embeddings of its name wherever the name appears, so the 10 unseen pairs are
new sentences built from offsets learned on other pairs. Offsets start at zero, so training starts exactly at
zero-shot CLIP; weight decay pulls them back towards the original words.

Variants (for the ablations):
  --tokens learned | fixed      learned offsets, or plain words (fixed prompts)
  --unfreeze N                  number of final image-encoder blocks to fine-tune (0 = frozen image encoder)
  --primitive-weight L          weight of the crop-only and disease-only prompt losses (0 = composition loss only)
  --heads                       control: same image encoder with linear crop/disease heads and no text
  --augment standard | strong   training augmentation strength (strong = field-robustness variant)

Stages:
  zeroshot  no training; predictions on the selection data (val_seen + pseudo_unseen)
  select    train without the pseudo-unseen pairs, pick the epoch by seen validation accuracy
  final     train on all 8,000 training images for --final-epochs and predict the test splits

  python src\train_clip_compositional.py --stage select --out runs\clip_select
"""
import argparse
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

BACKBONES = {'openai': ('ViT-B-16', 'openai'), 'bioclip': ('hf-hub:imageomics/bioclip', None)}
CONTEXT = 32  # prompts are short; a 32-token context saves memory against CLIP's 77
# Dataset names that read badly as English. Everything else only has underscores replaced by spaces.
# Only clear typos and awkward forms are changed; label IDs are untouched.
WORDS = {'pepper_bell': 'bell pepper', 'paddy': 'rice', 'aphis_spp': 'aphids', 'xamthomonas': 'xanthomonas',
         'crinckle': 'crinkle', 'verticulium_wilt': 'verticillium wilt', 'leafblast': 'leaf blast',
         'monillia_laxa': 'monilinia laxa', 'insects_damages': 'insect damage'}


def words(name):
    return WORDS.get(name, name.replace('_', ' '))


class PromptBank:
    """Token ids for every prompt, plus which positions belong to which crop or disease name."""

    def __init__(self, tokenizer, labels, taxonomy):
        self.tokenizer = tokenizer
        crops, diseases = taxonomy.names['crop'], taxonomy.names['disease']
        healthy = diseases.index('healthy') if 'healthy' in diseases else -1
        self.sets = {}
        comps = []
        for c in range(taxonomy.size('composition')):
            crop, disease = int(taxonomy.groups['crop'][c]), int(taxonomy.groups['disease'][c])
            if disease == healthy:
                comps.append(['a photo of a', ('d', disease, diseases), ('c', crop, crops), 'leaf.'])
            else:
                comps.append(['a photo of a', ('c', crop, crops), 'leaf with', ('d', disease, diseases), '.'])
        self.sets['composition'] = self.build(comps)
        self.sets['crop'] = self.build([['a photo of a', ('c', i, crops), 'leaf.'] for i in range(len(crops))])
        self.sets['disease'] = self.build([
            ['a photo of a leaf that is', ('d', i, diseases), '.'] if i == healthy
            else ['a photo of a leaf with', ('d', i, diseases), '.'] for i in range(len(diseases))])
        self.example = ' | '.join(self.decode('composition', i) for i in (0, 1))

    def build(self, prompts):
        t = self.tokenizer
        ids = torch.zeros(len(prompts), CONTEXT, dtype=torch.long)
        crop_at = torch.zeros(len(prompts), CONTEXT, dtype=torch.long)  # 0 = none, i + 1 = crop i
        disease_at = torch.zeros(len(prompts), CONTEXT, dtype=torch.long)
        eot = torch.zeros(len(prompts), dtype=torch.long)
        self._texts = getattr(self, '_texts', {})
        for row, pieces in enumerate(prompts):
            tokens, crops, diseases_ = [t.sot_token_id], [0], [0]
            for piece in pieces:
                if isinstance(piece, str):
                    new = t.encode(piece)
                    crops += [0] * len(new)
                    diseases_ += [0] * len(new)
                else:
                    kind, index, names = piece
                    new = t.encode(words(names[index]))
                    crops += [index + 1 if kind == 'c' else 0] * len(new)
                    diseases_ += [index + 1 if kind == 'd' else 0] * len(new)
                tokens += new
            tokens.append(t.eot_token_id)
            crops.append(0)
            diseases_.append(0)
            if len(tokens) > CONTEXT:
                raise ValueError(f'Prompt too long ({len(tokens)} tokens): {pieces}')
            ids[row, :len(tokens)] = torch.tensor(tokens)
            crop_at[row, :len(tokens)] = torch.tensor(crops)
            disease_at[row, :len(tokens)] = torch.tensor(diseases_)
            eot[row] = len(tokens) - 1
        return {'ids': ids, 'crop_at': crop_at, 'disease_at': disease_at, 'eot': eot}

    def decode(self, name, row):
        s = self.sets[name]
        return self.tokenizer.decode(s['ids'][row, 1:s['eot'][row]].tolist()).strip()


class CompositionalCLIP(nn.Module):
    def __init__(self, backbone, taxonomy, tokens='learned', unfreeze=4, heads=False):
        super().__init__()
        arch, pretrained = BACKBONES[backbone]
        self.clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.clip.requires_grad_(False)
        visual = self.clip.visual
        if unfreeze:
            for block in visual.transformer.resblocks[-unfreeze:]:
                block.requires_grad_(True)
            visual.ln_post.requires_grad_(True)
            visual.proj.requires_grad_(True)
        self.heads = heads
        width = self.clip.token_embedding.weight.shape[1]
        if heads:
            dim = visual.output_dim
            self.crop_head, self.disease_head = nn.Linear(dim, 55), nn.Linear(dim, 175)
            return
        bank = PromptBank(open_clip.get_tokenizer(arch), None, taxonomy)
        self.example = bank.example
        for name, tensors in bank.sets.items():
            for key, value in tensors.items():
                self.register_buffer(f'{name}_{key}', value, persistent=False)
        self.learned = tokens == 'learned'
        # Row 0 is the zero offset for positions that are not part of a name.
        self.crop_offset = nn.Parameter(torch.zeros(56, width), requires_grad=self.learned)
        self.disease_offset = nn.Parameter(torch.zeros(176, width), requires_grad=self.learned)
        self.cache = None

    def prompt_parameters(self):
        return [self.crop_offset, self.disease_offset] if not self.heads and self.learned else []

    def encode_prompts(self, name):
        c = self.clip
        ids = getattr(self, f'{name}_ids')
        x = c.token_embedding(ids)
        x = x + self.crop_offset[getattr(self, f'{name}_crop_at')] + self.disease_offset[getattr(self, f'{name}_disease_at')]
        x = x + c.positional_embedding[:CONTEXT]
        x = c.transformer(x, attn_mask=c.attn_mask[:CONTEXT, :CONTEXT])
        x = c.ln_final(x)
        x = x[torch.arange(len(x), device=x.device), getattr(self, f'{name}_eot')]
        x = c.text_projection(x) if isinstance(c.text_projection, nn.Linear) else x @ c.text_projection
        return F.normalize(x, dim=-1)

    def text_features(self):
        if self.learned and self.training:
            return {n: self.encode_prompts(n) for n in ('composition', 'crop', 'disease')}
        if self.cache is None:  # fixed prompts, or evaluation: encode once
            with torch.no_grad():
                self.cache = {n: self.encode_prompts(n) for n in ('composition', 'crop', 'disease')}
        return self.cache

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, 'learned', False):
            self.cache = None
        return self

    def forward(self, images):
        image = self.clip.encode_image(images)
        if self.heads:
            return {'crop': self.crop_head(image), 'disease': self.disease_head(image)}
        image = F.normalize(image, dim=-1)
        scale = self.clip.logit_scale.exp()
        text = self.text_features()
        return {name: scale * image @ feats.T.to(image.dtype) for name, feats in text.items()}


def run(args):
    seed_everything(args.seed)
    device = torch.device('cuda')
    taxonomy = Taxonomy(load_labels())
    data = make_datasets('select' if args.stage in ('zeroshot', 'select') else 'final', CLIP_MEAN, CLIP_STD,
                         augment=args.augment)
    model = CompositionalCLIP(args.backbone, taxonomy, args.tokens, args.unfreeze, args.heads).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'backbone_weights': BACKBONES[args.backbone], 'image_size': 224, 'normalisation': 'CLIP',
              'context_length': CONTEXT, 'name_words': WORDS, 'prompt_example': getattr(model, 'example', None),
              'trainable_parameters': sum(p.numel() for p in trainable),
              'train_images': len(data['train']), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('Example prompts:', config['prompt_example'], '| trainable', config['trainable_parameters'], flush=True)

    def outputs(images):
        logits = model(images)
        return {'composition': logits['composition']} if 'composition' in logits else logits

    level = 'composition'
    if args.stage == 'zeroshot':
        model.eval()
        save_predictions(args.out / 'pseudo.npz', **predict(outputs, eval_loader(data['pseudo'], args.batch_size,
                                                                                   args.workers), device))
        print('Saved zero-shot predictions.')
        return

    # Composition loss only over compositions present in training (pseudo-unseen pairs excluded when selecting).
    train_seen = torch.tensor(np.flatnonzero(seen_columns(taxonomy, args.stage)), device=device)
    remap = torch.full((taxonomy.size('composition'),), -1, dtype=torch.long, device=device)
    remap[train_seen] = torch.arange(len(train_seen), device=device)
    sampler = balanced_sampler(data['train'].records, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(data['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    new_head = lambda n: n.startswith(('crop_head', 'disease_head'))  # noqa: E731
    groups = [{'params': model.prompt_parameters(), 'lr': args.token_lr, 'weight_decay': args.token_wd},
              {'params': [p for n, p in model.named_parameters() if new_head(n)], 'lr': 1e-3, 'weight_decay': 0.05},
              {'params': [p for n, p in model.named_parameters()
                          if p.requires_grad and 'offset' not in n and not new_head(n)],
               'lr': args.lr, 'weight_decay': 0.05}]
    optimiser = torch.optim.AdamW([g for g in groups if g['params']])
    steps = args.epochs * len(train_loader)
    warmup = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + np.cos(np.pi * min(s, steps) / steps)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    val_loader = eval_loader(data['val_seen'], args.batch_size, args.workers) if args.stage == 'select' else None
    run_epochs = args.final_epochs if args.stage == 'final' else args.epochs
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
                    logits = model(images)
                    if args.heads:
                        loss = loss_fn(logits['crop'], crop) + loss_fn(logits['disease'], disease)
                    else:
                        target = remap[batch['composition'].to(device, non_blocking=True)]
                        loss = loss_fn(logits['composition'][:, train_seen], target)
                        if args.primitive_weight:
                            loss = loss + args.primitive_weight * (loss_fn(logits['crop'], crop)
                                                                   + loss_fn(logits['disease'], disease))
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
                score = val_score(predict(outputs, val_loader, device), data['val_seen'].records, taxonomy, level)
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save({k: v for k, v in model.state_dict().items()}, args.out / 'best.pth')
                else:
                    stale += 1
            seconds = time.perf_counter() - start
            history.writerow([epoch, total / count, score, round(seconds, 1)])
            file.flush()
            print(f'epoch {epoch}/{run_epochs} loss {total / count:.4f} val {score} {seconds:.0f}s', flush=True)
            if args.stage == 'select' and stale >= args.patience:
                break
    model.eval()
    if args.stage == 'select':
        model.load_state_dict(torch.load(args.out / 'best.pth', map_location=device, weights_only=True))
        model.cache = None
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
    parser.add_argument('--stage', choices=('zeroshot', 'select', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backbone', choices=BACKBONES, default='openai')
    parser.add_argument('--tokens', choices=('learned', 'fixed'), default='learned')
    parser.add_argument('--unfreeze', type=int, default=4)
    parser.add_argument('--primitive-weight', type=float, default=1.0)
    parser.add_argument('--heads', action='store_true', help='Control: linear crop/disease heads, no text.')
    parser.add_argument('--augment', choices=('standard', 'strong'), default='standard',
                        help='Training augmentation strength (strong = field-robustness variant).')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--final-epochs', type=int)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help='Image encoder (and linear heads).')
    parser.add_argument('--token-lr', type=float, default=5e-5)
    parser.add_argument('--token-wd', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage == 'final' and not args.final_epochs:
        parser.error('--final-epochs is required for the final stage.')
    run(args)


if __name__ == '__main__':
    main()
