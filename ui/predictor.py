"""Model loading and prediction for the plant disease UI.

Every model gives crop and disease scores; a composition (crop-disease pair) is scored as
P(crop) x P(disease), renormalised over the 333 valid pairs, exactly as evaluate.level_scores does.
The calibration constant gamma (from the run's calibration.json, or the number in ui/config.json) is
then subtracted from every seen pair's log-probability, as in gzsl.calibrated, so the UI ranks pairs
the same way the reported results do.

Model choices live in ui/config.json, so switching the proposed run is a one-line change there.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny

UI = Path(__file__).resolve().parent
ROOT = UI.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from evaluate import Taxonomy, level_scores  # noqa: E402
from preprocessing import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, eval_transform  # noqa: E402

ASSETS = UI / 'assets'
# Dataset names that read badly; the same fixes the proposed model's prompts use.
WORDS = {'pepper_bell': 'bell pepper', 'paddy': 'rice', 'aphis_spp': 'aphids', 'xamthomonas': 'xanthomonas',
         'crinckle': 'crinkle', 'verticulium_wilt': 'verticillium wilt', 'leafblast': 'leaf blast',
         'monillia_laxa': 'monilinia laxa', 'insects_damages': 'insect damage'}


def pretty(name):
    text = WORDS.get(name, name.replace('_', ' '))
    return text[:1].upper() + text[1:]


def load_config():
    return json.loads((UI / 'config.json').read_text(encoding='utf-8'))


def load_label_map():
    """Prefer the copy in ui/assets so the demo laptop does not need the dataset folder."""
    for path in (ASSETS / 'label_map.json', ROOT / 'dataset/prepared/label_map.json'):
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    raise FileNotFoundError('label_map.json not found: run ui/build_index.py on a machine with the dataset.')


def pick_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).replace('NVIDIA ', '').replace('GeForce ', '')
        return torch.device('cuda'), f'GPU · {name}'
    return torch.device('cpu'), 'CPU'


class Labels:
    def __init__(self):
        self.taxonomy = Taxonomy(load_label_map())
        t = self.taxonomy
        self.crop_names = [pretty(n) for n in t.names['crop']]
        self.disease_names = [pretty(n) for n in t.names['disease']]
        self.comp_crop = t.groups['crop']
        self.comp_disease = t.groups['disease']
        self.seen_mask = np.zeros(t.size('composition'), dtype=bool)
        self.seen_mask[t.seen] = True

    def pair_title(self, comp):
        return f'{self.crop_names[self.comp_crop[comp]]} · {self.disease_names[self.comp_disease[comp]]}'


# ---------------------------------------------------------------- the three model families

class IndependentCNNs(nn.Module):
    """Baseline 1: a crop ConvNeXt and a disease ConvNeXt, trained separately."""

    def __init__(self, crop_path, disease_path):
        super().__init__()
        self.crop_net = self._load(crop_path, 55)
        self.disease_net = self._load(disease_path, 175)

    @staticmethod
    def _load(path, classes):
        net = convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, classes)
        state = torch.load(path, map_location='cpu', weights_only=False)
        net.load_state_dict(state.get('model', state), strict=True)
        return net

    def forward(self, x):
        return {'crop': self.crop_net(x), 'disease': self.disease_net(x)}


class MultiTaskCNN(nn.Module):
    """Baseline 2: one ConvNeXt backbone with a crop head and a disease head."""

    def __init__(self, path):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        width = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.crop = nn.Linear(width, 55)
        self.disease = nn.Linear(width, 175)
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state.get('model', state), strict=True)

    def forward(self, x):
        features = self.backbone(x)
        return {'crop': self.crop(features), 'disease': self.disease(features)}


class DisentangledModel(nn.Module):
    """Disentangled ConvNeXt-Tiny v2 (Kelvin): crop and disease branches on a shared ConvNeXt backbone."""

    def __init__(self, run):
        super().__init__()
        from torch.torch_version import TorchVersion
        from train_disentangled_v2 import DisentangledNetV2
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        self.net = DisentangledNetV2(embed_dim=int(cfg.get('embed_dim', 512)), head_type=str(cfg.get('head_type', 'linear')),
                                     use_grl=float(cfg.get('grl_lambda', 0.02)) > 0.0,
                                     detach_grl_backbone=not bool(cfg.get('no_detach_grl', False)),
                                     use_mixstyle=bool(cfg.get('mixstyle', False)), dropout=float(cfg.get('dropout', 0.1)),
                                     pretrained=False)
        # The checkpoint stores a torch version stamp; allow just that, keep the safe loader for the rest.
        torch.serialization.add_safe_globals([TorchVersion])
        state = torch.load(run / 'final.pth', map_location='cpu', weights_only=True)
        self.net.load_state_dict(state.get('model', state), strict=True)

    def forward(self, x):
        out = self.net(x)
        return {'crop': out['crop'], 'disease': out['disease']}


class CSPModel(nn.Module):
    """CSP + WiSE-FT (Nigel): OpenAI CLIP ViT-B/16 with learned crop and disease words, scoring all 333 pairs.

    Same steps as src/csp_model.load_csp, but the label names come from the UI's own copy, so the demo
    laptop needs no dataset folder. The 333 text features never change, so they are computed once.
    """

    def __init__(self, path, taxonomy):
        super().__init__()
        from csp_model import CSP, image_normalisation, load_clip
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        clip_model, tokenizer = load_clip('cpu', ckpt['clip_model'], ckpt['clip_weights'])
        if 'clip_state' in ckpt:  # fine-tuned (WiSE-FT blended) CLIP weights
            unexpected = clip_model.load_state_dict(ckpt['clip_state'], strict=False).unexpected_keys
            if unexpected:
                raise ValueError(f'Unexpected CLIP weights in {path}: {unexpected[:3]}')
        self.csp = CSP(clip_model, tokenizer, taxonomy=taxonomy, dropout=ckpt['dropout'])
        with torch.no_grad():
            self.csp.disease_embeddings.copy_(ckpt['disease_embeddings'])
            self.csp.crop_embeddings.copy_(ckpt['crop_embeddings'])
        self.csp.eval()
        self.mean, self.std = image_normalisation(clip_model)
        self._text = None

    def forward(self, x):
        feats = self.csp.clip.encode_image(x).float()
        with torch.autocast(x.device.type, enabled=False):  # text side in full precision, as predict_csp does
            if self._text is None:
                self._text = self.csp.text_features(self.csp.all_pairs).float()
            logits = self.csp.clip.logit_scale.exp() * F.normalize(feats, dim=-1) @ self._text.t()
        return {'composition': logits}


class BioCLIP2Ensemble(nn.Module):
    """Final model (Neng yi): BioCLIP 2 ViT-L/14 + WiSE-FT, an ensemble of 3 seeds with horizontal-flip TTA.

    The checkpoint (format 'wise-clip-ensemble/1') stores, per seed, the fine-tuned top image blocks and the
    crop / disease heads; the frozen lower blocks come from the public BioCLIP 2 weights. Scores follow the
    checkpoint's own rule: logits = logit_scale * normalise(image feature) . head, then the mean of each factor's
    log-softmax over seeds and flips. Checked against the checkpoint's test reference (H 65.24 vs 65.23).
    """

    def __init__(self, path):
        super().__init__()
        import open_clip
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        name, pretrained = ckpt['open_clip']
        clip = open_clip.create_model(name, pretrained=pretrained)
        self.visual = clip.visual.requires_grad_(False)  # inference only; the text tower is not needed
        self.scale = float(clip.logit_scale.detach().exp())
        self.mean, self.std = tuple(ckpt['mean']), tuple(ckpt['std'])
        self.flip = ckpt.get('tta') == 'horizontal flip'
        own = dict(self.visual.named_parameters())
        self.member_maps = []
        for i, member in enumerate(ckpt['members']):
            mapping = []
            for key, value in member['state'].items():
                if key.startswith('clip.visual.'):
                    pname = key[len('clip.visual.'):]
                    if pname not in own:
                        raise ValueError(f'Unexpected weight in {path}: {key}')
                    bname = f'm{i}__' + pname.replace('.', '__')
                    self.register_buffer(bname, value, persistent=False)
                    mapping.append((pname, bname))
            self.register_buffer(f'm{i}__crop_head', member['state']['crop_head'], persistent=False)
            self.register_buffer(f'm{i}__disease_head', member['state']['disease_head'], persistent=False)
            self.member_maps.append(mapping)

    def forward(self, x):
        batch = x.shape[0]
        views = torch.cat([x, torch.flip(x, dims=[3])]) if self.flip else x
        n_views = views.shape[0] // batch
        own = dict(self.visual.named_parameters())
        crop_lp = disease_lp = 0
        for i, mapping in enumerate(self.member_maps):
            with torch.no_grad():
                for pname, bname in mapping:  # swap in this seed's fine-tuned blocks
                    own[pname].copy_(getattr(self, bname))
            # Autocast caches half-precision copies of weights for the whole region; drop them after the swap,
            # or later seeds silently reuse the first seed's cached blocks.
            torch.clear_autocast_cache()
            feats = F.normalize(self.visual(views).float(), dim=-1)
            # Heads in full precision: with logit scale 100, half precision rounds logits to ~0.06 and flips
            # close seen-vs-unseen calls near gamma.
            with torch.autocast(x.device.type, enabled=False):
                for head, total in (('crop', 'c'), ('disease', 'd')):
                    logits = self.scale * feats @ getattr(self, f'm{i}__{head}_head').float().t()
                    lp = F.log_softmax(logits, dim=-1).view(n_views, batch, -1).mean(0)
                    if total == 'c':
                        crop_lp = crop_lp + lp
                    else:
                        disease_lp = disease_lp + lp
        k = len(self.member_maps)
        return {'crop': crop_lp / k, 'disease': disease_lp / k}


class ClipModel(nn.Module):
    """The proposed model (any variant of src/train_clip_compositional.py), plus image embeddings."""

    def __init__(self, run, taxonomy):
        super().__init__()
        from train_clip_compositional import CompositionalCLIP
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        self.model = CompositionalCLIP(config['backbone'], taxonomy, config.get('tokens', 'learned'),
                                       config.get('unfreeze', 4), config.get('heads', False))
        state = torch.load(run / 'final.pth', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)

    def forward(self, x):
        m = self.model
        if m.heads:
            features = m.clip.encode_image(x)
            out = {'crop': m.crop_head(features), 'disease': m.disease_head(features)}
        else:
            out = m(x)
            features = m.clip.encode_image(x)
        out['embedding'] = F.normalize(features.float(), dim=-1)
        return out


# ---------------------------------------------------------------- predictor

class Predictor:
    def __init__(self):
        self.config = load_config()
        self.labels = Labels()
        self.device, self.device_label = pick_device()
        self.models, self.meta = {}, {}
        for key, spec in self.config['models'].items():
            self.meta[key] = {'label': spec['label'], 'short': spec['short'], 'arch': spec['arch'],
                              'group': spec.get('group', 'proposed'), 'results': spec.get('results'),
                              'gamma': self._gamma(spec), 'available': self._available(spec)}

    @staticmethod
    def _path(rel):
        return ROOT / rel

    def _gamma(self, spec):
        g = spec.get('gamma')
        if isinstance(g, str):
            path = self._path(spec['run']) / g
            return float(json.loads(path.read_text(encoding='utf-8'))['gamma']) if path.is_file() else None
        return None if g is None else float(g)

    def _available(self, spec):
        files = {'clip': [str(Path(spec.get('run', '')) / 'final.pth')],
                 'independent': [spec.get('crop_weights', ''), spec.get('disease_weights', '')],
                 'disentangled': [str(Path(spec.get('run', '')) / 'final.pth')],
                 'csp': [spec.get('checkpoint', '')],
                 'bioclip2': [spec.get('checkpoint', '')],
                 'multitask': [spec.get('weights', '')]}[spec['kind']]
        return all(f and self._path(f).is_file() for f in files)

    def choices(self):
        return [(m['label'], k) for k, m in self.meta.items() if m['available']]

    def load(self, key):
        """Load a model on first use and keep it in memory."""
        if key in self.models:
            return self.models[key]
        spec = self.config['models'][key]
        if spec['kind'] == 'clip':
            net, mean, std = ClipModel(self._path(spec['run']), self.labels.taxonomy), CLIP_MEAN, CLIP_STD
        elif spec['kind'] == 'bioclip2':
            net = BioCLIP2Ensemble(self._path(spec['checkpoint']))
            mean, std = net.mean, net.std
        elif spec['kind'] == 'csp':
            net = CSPModel(self._path(spec['checkpoint']), self.labels.taxonomy)
            mean, std = net.mean, net.std
        elif spec['kind'] == 'disentangled':
            net, mean, std = DisentangledModel(self._path(spec['run'])), IMAGENET_MEAN, IMAGENET_STD
        elif spec['kind'] == 'independent':
            net = IndependentCNNs(self._path(spec['crop_weights']), self._path(spec['disease_weights']))
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            net, mean, std = MultiTaskCNN(self._path(spec['weights'])), IMAGENET_MEAN, IMAGENET_STD
        net = net.to(self.device).eval()
        self.models[key] = (net, eval_transform(224, mean, std))
        return self.models[key]

    @torch.no_grad()
    def raw(self, key, image):
        """Model outputs for one PIL image: numpy crop / disease (or composition) logits, optional embedding."""
        net, transform = self.load(key)
        x = transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            out = net(x)
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    def predict(self, key, image, k=5):
        start = time.perf_counter()
        out = self.raw(key, image)
        seconds = time.perf_counter() - start
        L = self.labels
        scores = {name: out[name] for name in ('crop', 'disease', 'composition') if name in out}
        log_p = level_scores(scores, L.taxonomy)
        gamma = self.meta[key]['gamma'] or 0.0
        comp = log_p['composition'][0].copy()
        comp[L.seen_mask] -= gamma
        comp_p = np.exp(comp - comp.max())
        comp_p /= comp_p.sum()
        # Crop and disease confidence = calibrated pair probabilities summed per crop / per disease, so the
        # three cards agree with the ranked pair (the raw heads ignore calibration and can disagree).
        crop_p = np.bincount(L.comp_crop, weights=comp_p, minlength=len(L.crop_names))
        disease_p = np.bincount(L.comp_disease, weights=comp_p, minlength=len(L.disease_names))

        order = np.argsort(-comp_p)
        top = [{'id': int(c), 'title': L.pair_title(c), 'p': float(comp_p[c]),
                'unseen': not bool(L.seen_mask[c]), 'crop': int(L.comp_crop[c]),
                'disease': int(L.comp_disease[c])} for c in order[:k]]
        best = top[0]

        def runner_up(probs, chosen, names):
            idx = next(int(i) for i in np.argsort(-probs) if int(i) != chosen)
            return names[idx], float(probs[idx])

        return {
            'model': key, 'meta': self.meta[key], 'seconds': seconds, 'device': self.device_label,
            'calibrated': self.meta[key]['gamma'] is not None, 'top': top, 'best': best,
            'crop': {'name': L.crop_names[best['crop']], 'p': float(crop_p[best['crop']]),
                     'next': runner_up(crop_p, best['crop'], L.crop_names)},
            'disease': {'name': L.disease_names[best['disease']], 'p': float(disease_p[best['disease']]),
                        'next': runner_up(disease_p, best['disease'], L.disease_names)},
            'rank_of': {int(c): r + 1 for r, c in enumerate(order)},
            'embedding': out['embedding'][0] if 'embedding' in out else None,
        }

    def embed(self, image):
        """Image embedding from the index model, for the similar-images search."""
        return self.raw(self.config['index_model'], image)['embedding'][0]


class SimilarIndex:
    """Nearest training images by cosine similarity of the index model's image embeddings."""

    def __init__(self):
        path = ASSETS / 'index.npz'
        self.ok = path.is_file()
        if not self.ok:
            return
        data = np.load(path, allow_pickle=False)
        self.emb = data['embeddings'].astype(np.float32)
        self.comp, self.crop, self.disease = data['composition'], data['crop'], data['disease']
        self.files = [str(f) for f in data['thumbs']]

    def _top(self, sims, mask, n):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-sims[idx])][:n]
        return [(self.files[i], int(self.comp[i]), float(sims[i])) for i in idx]

    def search(self, embedding, best, n=8):
        """Seen pair: closest images of that pair. Unseen pair: closest same-crop and same-disease images."""
        if not self.ok or embedding is None:
            return {'mode': 'none'}
        sims = self.emb @ np.asarray(embedding, dtype=np.float32)
        if not best['unseen']:
            same = self._top(sims, self.comp == best['id'], n)
            if len(same) < n:
                used = {f for f, _, _ in same}
                rest = [r for r in self._top(sims, np.ones(len(sims), dtype=bool), n * 2) if r[0] not in used]
                same += rest[:n - len(same)]
            return {'mode': 'seen', 'images': same}
        half = n // 2
        return {'mode': 'unseen',
                'crop': self._top(sims, (self.crop == best['crop']) & (self.disease != best['disease']), half),
                'disease': self._top(sims, (self.disease == best['disease']) & (self.crop != best['crop']), half)}
