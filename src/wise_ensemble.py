r"""Best model as one checkpoint: WiSE-FT on BioCLIP 2, 3-seed ensemble (test H 65.23, AUSUC 55.97).

export  reads runs\wise_b2_tuned{,_s1,_s2}_final\final.pth (fine-tuned weights, alpha 1), mixes each with the
        zero-shot BioCLIP 2 weights at the alpha its select stage chose (1.0, 0.7, 1.0), and writes one file with
        the three mixed weight sets, gamma, the label map and the preprocessing settings. Only the trainable part is
        stored (last 6 image blocks, ln_post, proj, crop and disease heads); the frozen rest is the pretrained
        hf-hub:imageomics/bioclip-2 image tower, which open_clip downloads on first use.
test    predicts both test splits from the checkpoint, applies its gamma and prints seen / unseen / H. With
        --reference it also compares the scores with the original ensemble's prediction files.
predict top-k crop + disease pairs for image files (the scoring the UI should use).

Each member runs with flip test-time augmentation; the ensemble averages the members' crop and disease
log-probabilities; a pair scores log P(crop) + log P(disease), renormalised over the 333 valid pairs, and gamma is
subtracted from the 323 seen pairs.

  python src\wise_ensemble.py export --out checkpoints\wise_bioclip2_ens.pth
  python src\wise_ensemble.py test --checkpoint checkpoints\wise_bioclip2_ens.pth --out runs\wise_b2_ckpt_final ^
      --reference runs\wise_b2_ens_final
  python src\wise_ensemble.py predict --checkpoint checkpoints\wise_bioclip2_ens.pth leaf1.jpg leaf2.jpg
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import PROJECT, load_labels, load_split  # noqa: E402
from evaluate import Taxonomy, harmonic_mean, level_scores, save_predictions  # noqa: E402
from gzsl import calibrated, composition_log_probs, eval_loader, predict, split_accuracy  # noqa: E402
from preprocessing import CLIP_MEAN, CLIP_STD, DPDDataset, eval_transform, load_image  # noqa: E402
from train_wise import BACKBONES, WiseCLIP, trainable_state, tta  # noqa: E402

FORMAT = 'wise-clip-ensemble/1'
MEMBERS = ('wise_b2_tuned', 'wise_b2_tuned_s1', 'wise_b2_tuned_s2')
ENSEMBLE = 'wise_b2_ens'
IMAGE_SIZE = 224


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def combine(member_predictions):
    """Average the members' per-factor log-probabilities (float64), exactly as the original ensemble did."""
    return {k: np.mean([lsm(p[k].astype(np.float64)) for p in member_predictions], 0).astype(np.float32)
            for k in ('crop', 'disease')}


def export(args):
    runs = PROJECT / 'runs'
    configs = [json.loads((runs / f'{m}_final/config.json').read_text(encoding='utf-8')) for m in MEMBERS]
    for key in ('backbone', 'unfreeze', 'score', 'tune', 'head_init'):
        if len({str(c[key]) for c in configs}) != 1:
            raise ValueError(f'Members differ in {key}; they must share one architecture.')
    base = configs[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = load_labels()
    model = WiseCLIP(Taxonomy(labels), base['unfreeze'], base['score'], base['tune'], base['backbone']).to(device)
    model.build_text(device, base['head_init'])
    zero = trainable_state(model)  # zero-shot weights: pretrained blocks + text-embedding heads
    members = []
    for name, config in zip(MEMBERS, configs):
        tuned = torch.load(runs / f'{name}_final/final.pth', weights_only=True)
        if tuned.keys() != zero.keys():
            raise ValueError(f'{name}: final.pth does not match the model parameters.')
        alpha = float(config['alpha'])
        mixed = {n: (1 - alpha) * zero[n] + alpha * tuned[n] for n in zero}  # train_wise.set_alpha arithmetic
        members.append({'run': name, 'seed': config['seed'], 'alpha': alpha, 'final_epochs': config['final_epochs'],
                        'state': mixed})
        print(f'{name}: seed {config["seed"]}, alpha {alpha}, {config["final_epochs"]} epochs')
    gamma = json.loads((runs / f'{ENSEMBLE}_final/calibration.json').read_text(encoding='utf-8'))['gamma']
    metrics = json.loads((PROJECT / f'results/{ENSEMBLE}_calibrated/metrics.json').read_text(encoding='utf-8'))
    test = {s: metrics['splits'][f'test_{s}']['composition']['top1']['per_class'] for s in ('seen', 'unseen')}
    test['harmonic'] = harmonic_mean(test['seen'], test['unseen'])
    bundle = {'format': FORMAT, 'name': ENSEMBLE, 'backbone': base['backbone'],
              'open_clip': list(BACKBONES[base['backbone']]), 'unfreeze': base['unfreeze'], 'score': base['score'],
              'tune': base['tune'], 'image_size': IMAGE_SIZE, 'mean': list(CLIP_MEAN), 'std': list(CLIP_STD),
              'tta': 'horizontal flip', 'combine': 'mean of per-factor log-softmax', 'gamma': float(gamma),
              'labels': labels, 'test_reference': test, 'members': members}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    size = args.out.stat().st_size / 2 ** 20
    print(f'gamma {gamma}; reference test seen {100 * test["seen"]:.2f} / unseen {100 * test["unseen"]:.2f} / '
          f'H {100 * test["harmonic"]:.2f}')
    print(f'Wrote {args.out} ({size:.0f} MB)')


class Ensemble:
    """One BioCLIP 2 image tower; the members' trainable weights are swapped in turn (kept on the device)."""

    def __init__(self, path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if bundle.get('format') != FORMAT:
            raise ValueError(f'{path} is not a {FORMAT} checkpoint.')
        self.info = {k: v for k, v in bundle.items() if k != 'members'}
        self.taxonomy = Taxonomy(bundle['labels'])
        self.gamma = bundle['gamma']
        self.transform = eval_transform(bundle['image_size'], bundle['mean'], bundle['std'])
        self.min_size = round(bundle['image_size'] / 0.875)
        model = WiseCLIP(self.taxonomy, bundle['unfreeze'], bundle['score'], bundle['tune'], bundle['backbone'])
        clip = model.clip  # the heads come from the checkpoint, so the text tower is never run
        for name in ('transformer', 'token_embedding', 'ln_final'):
            setattr(clip, name, None)
        del model.bank
        first = bundle['members'][0]['state']
        model.crop_head = nn.Parameter(torch.zeros_like(first['crop_head']))
        model.disease_head = nn.Parameter(torch.zeros_like(first['disease_head']))
        self.model = model.to(self.device).eval()
        self.params = {n: p for n, p in self.model.named_parameters() if n in first}
        self.members = [{**{k: v for k, v in m.items() if k != 'state'},
                         'state': {n: t.to(self.device) for n, t in m['state'].items()}} for m in bundle['members']]
        self.seen = np.zeros(self.taxonomy.size('composition'), dtype=bool)
        self.seen[self.taxonomy.seen] = True

    @torch.no_grad()
    def use(self, member):
        for n, p in self.params.items():
            p.copy_(member['state'][n])

    def predict_loader(self, loader):
        """Per-member and combined predictions for a DataLoader of DPDDataset items."""
        per_member = []
        for member in self.members:
            self.use(member)
            per_member.append(predict(tta(self.model), loader, self.device))
            print(f'  {member["run"]} done', flush=True)
        return per_member, {'filenames': per_member[0]['filenames'], **combine(per_member)}

    def composition_scores(self, predictions):
        """Calibrated pair scores (log P(crop) + log P(disease) over the 333 pairs, minus gamma on seen pairs)."""
        log_p = level_scores(predictions, self.taxonomy)
        return calibrated(log_p['composition'], self.seen, self.gamma), log_p

    @torch.no_grad()
    def predict_images(self, images, k=5):
        """images: paths or PIL images. Returns one dict per image with the top-k pairs."""
        batch = torch.stack([self.transform(load_image(i, self.min_size) if isinstance(i, (str, Path))
                                            else i.convert('RGB')) for i in images])
        outputs = []
        for member in self.members:
            self.use(member)
            with torch.amp.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                out = tta(self.model)(batch.to(self.device))
            outputs.append({key: value.float().cpu().numpy() for key, value in out.items()})
        scores, log_p = self.composition_scores(combine(outputs))
        confidence = np.exp(lsm(scores))  # softmax of the calibrated scores, for display only
        names, groups = self.taxonomy.names, self.taxonomy.groups
        results = []
        for row in range(len(batch)):
            top = np.argsort(-scores[row])[:k]
            results.append({
                'crop': names['crop'][int(log_p['crop'][row].argmax())],
                'disease': names['disease'][int(log_p['disease'][row].argmax())],
                'top': [{'composition_id': int(c), 'composition': names['composition'][c],
                         'crop': names['crop'][groups['crop'][c]], 'disease': names['disease'][groups['disease'][c]],
                         'seen_in_training': bool(self.seen[c]), 'confidence': float(confidence[row, c])}
                        for c in top]})
        return results


def test(args):
    ensemble = Ensemble(args.checkpoint)
    print('Members:', ', '.join(f'{m["run"]} (alpha {m["alpha"]})' for m in ensemble.members), '| gamma',
          ensemble.gamma)
    args.out.mkdir(parents=True, exist_ok=True)
    scores = {}
    for split in ('test_seen', 'test_unseen'):
        records = load_split(split)
        dataset = DPDDataset(records, ensemble.transform, ensemble.info['image_size'])
        print(f'{split}: {len(records)} images', flush=True)
        per_member, combined = ensemble.predict_loader(eval_loader(dataset, args.batch_size, args.workers))
        save_predictions(args.out / f'{split}.npz', **combined)
        log_p = calibrated(composition_log_probs(combined, records, ensemble.taxonomy), ensemble.seen,
                           ensemble.gamma)
        save_predictions(args.out / f'{split}_calibrated.npz', [r['filename'] for r in records], composition=log_p)
        scores[split] = split_accuracy(log_p, records)
        if args.reference:
            ref = dict(np.load(args.reference / f'{split}.npz'))
            same = np.array_equal(ref['filenames'], combined['filenames'])
            gap = max(float(np.abs(ref[k] - combined[k]).max()) for k in ('crop', 'disease')) if same else None
            agree = (float((ref['disease'].argmax(1) == combined['disease'].argmax(1)).mean()) if same else None)
            print(f'  vs reference: same order {same}, max |diff| {gap}, disease top-1 agreement {agree}')
    h = harmonic_mean(scores['test_seen'], scores['test_unseen'])
    ref = ensemble.info['test_reference']
    print(f'Checkpoint test (per-class Top-1, gamma {ensemble.gamma}): seen {100 * scores["test_seen"]:.2f} / '
          f'unseen {100 * scores["test_unseen"]:.2f} / H {100 * h:.2f}')
    print(f'Reference:                                   seen {100 * ref["seen"]:.2f} / '
          f'unseen {100 * ref["unseen"]:.2f} / H {100 * ref["harmonic"]:.2f}')


def predict_cli(args):
    ensemble = Ensemble(args.checkpoint)
    for path, result in zip(args.images, ensemble.predict_images(args.images, args.top_k)):
        print(f'{path}: crop {result["crop"]}, disease {result["disease"]}')
        for i, pair in enumerate(result['top'], 1):
            tag = '' if pair['seen_in_training'] else '  [unseen pair]'
            print(f'  {i}. {pair["crop"]} / {pair["disease"]}  {100 * pair["confidence"]:.1f}%{tag}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('export', help='Bundle the three seeds into one checkpoint.')
    e.add_argument('--out', type=Path, default=PROJECT / 'checkpoints/wise_bioclip2_ens.pth')
    t = commands.add_parser('test', help='Predict the test splits from the checkpoint and score them.')
    t.add_argument('--checkpoint', type=Path, default=PROJECT / 'checkpoints/wise_bioclip2_ens.pth')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--reference', type=Path, help='Run folder with the original ensemble test_*.npz.')
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--workers', type=int, default=4)
    p = commands.add_parser('predict', help='Top-k pairs for image files.')
    p.add_argument('--checkpoint', type=Path, default=PROJECT / 'checkpoints/wise_bioclip2_ens.pth')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('images', nargs='+')
    args = parser.parse_args()
    {'export': export, 'test': test, 'predict': predict_cli}[args.command](args)


if __name__ == '__main__':
    main()
