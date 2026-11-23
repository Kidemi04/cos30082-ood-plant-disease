"""Compositional Soft Prompting (CSP; Nayak, Yu and Bach, ICLR 2023) on a frozen CLIP for DPD-Small.

Every composition is scored with the prompt "a photo of [disease] [crop]". The [disease] and [crop] words are
learnable token embeddings, one per disease (175 x 512) and one per crop (55 x 512), shared by every pair that
contains them. Both CLIP encoders stay frozen, so an unseen pair such as tomato_early_blight is built from
embeddings learned on other pairs (tomato_late_blight, potato_early_blight, ...).

Needs open_clip; install it without replacing the CUDA build of torch, e.g. with a constraints file that pins
the installed torch and torchvision: pip install open_clip_torch -c <constraints.txt>
"""
from pathlib import Path
import sys

import open_clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_access import load_labels  # noqa: E402
from evaluate import Taxonomy  # noqa: E402

# (open_clip model name, pretrained tag). OpenAI's weights were trained with QuickGELU, so not plain ViT-B-16;
# BioCLIP (ViT-B/16 trained on the TreeOfLife-10M biology images) loads its weights from the Hugging Face hub.
BACKBONES = {'openai': ('ViT-B-16-quickgelu', 'openai'), 'bioclip': ('hf-hub:imageomics/bioclip', None)}
CLIP_MODEL, CLIP_WEIGHTS = BACKBONES['openai']
TEMPLATE = 'a photo of x x'  # the two placeholder words become the disease and crop embeddings


def load_clip(device, model_name=CLIP_MODEL, weights=CLIP_WEIGHTS):
    """Return the frozen CLIP model (eval mode, no gradients) and its tokenizer."""
    model = open_clip.create_model(model_name, pretrained=weights)
    model.eval().requires_grad_(False).to(device)
    return model, open_clip.get_tokenizer(model_name)


def backbone_slug(model_name=CLIP_MODEL, weights=CLIP_WEIGHTS):
    """A file-name-safe name for a backbone, e.g. ViT-B-16-quickgelu_openai or imageomics-bioclip_hub."""
    return f'{model_name.replace("hf-hub:", "").replace("/", "-")}_{weights or "hub"}'


def image_normalisation(clip_model):
    """CLIP's own image mean and std, for the shared transforms in preprocessing.py."""
    visual = clip_model.visual
    return (tuple(getattr(visual, 'image_mean', None) or OPENAI_DATASET_MEAN),
            tuple(getattr(visual, 'image_std', None) or OPENAI_DATASET_STD))


@torch.no_grad()
def name_embeddings(clip_model, tokenizer, names):
    """Initial embedding of each name: the mean CLIP token embedding of its words."""
    tokens = tokenizer([name.replace('_', ' ') for name in names]).to(clip_model.token_embedding.weight.device)
    embeddings = clip_model.token_embedding(tokens)
    ends = tokens.argmax(1).tolist()  # end-of-text is the highest token ID
    return torch.stack([embeddings[i, 1:end].mean(0) for i, end in enumerate(ends)])


def encode_token_embeddings(clip_model, tokens):
    """CLIP text features from token embeddings (N, L, D) whose last position is the end-of-text token."""
    length = tokens.shape[1]
    x = tokens + clip_model.positional_embedding[:length]
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask[:length, :length])
    x = clip_model.ln_final(x[:, -1])
    projection = clip_model.text_projection
    return projection(x) if isinstance(projection, nn.Linear) else x @ projection


class CSP(nn.Module):
    """Learnable disease and crop embeddings, turned into composition classifiers by the frozen CLIP."""

    def __init__(self, clip_model, tokenizer, taxonomy=None, dropout=0.2):
        super().__init__()
        taxonomy = taxonomy or Taxonomy(load_labels())
        device = clip_model.token_embedding.weight.device
        self.clip = clip_model
        self.dropout = nn.Dropout(dropout)
        template = tokenizer([TEMPLATE]).to(device)
        end = int(template.argmax())
        # The text transformer is causal, so tokens after end-of-text never reach its feature; dropping them
        # gives the same features as the full 77-token sequence at a fraction of the cost.
        with torch.no_grad():
            self.register_buffer('template', clip_model.token_embedding(template[:, :end + 1]))
        self.disease_slot, self.crop_slot = end - 2, end - 1
        self.disease_embeddings = nn.Parameter(name_embeddings(clip_model, tokenizer, taxonomy.names['disease']))
        self.crop_embeddings = nn.Parameter(name_embeddings(clip_model, tokenizer, taxonomy.names['crop']))
        self.register_buffer('pair_disease', torch.as_tensor(taxonomy.groups['disease'], device=device))
        self.register_buffer('pair_crop', torch.as_tensor(taxonomy.groups['crop'], device=device))
        self.register_buffer('all_pairs', torch.arange(len(taxonomy.names['composition']), device=device))

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()  # only the embedding dropout changes between training and evaluation
        return self

    def text_features(self, pairs):
        """L2-normalised text features of the given composition IDs."""
        diseases = self.dropout(self.disease_embeddings)[self.pair_disease[pairs]]
        crops = self.dropout(self.crop_embeddings)[self.pair_crop[pairs]]
        template = self.template.expand(len(pairs), -1, -1)
        tokens = torch.cat([template[:, :self.disease_slot], diseases[:, None], crops[:, None],
                            template[:, self.crop_slot + 1:]], 1)
        return F.normalize(encode_token_embeddings(self.clip, tokens), dim=-1)

    def forward(self, image_features, pairs):
        """Logits (images, pairs): CLIP's scaled cosine similarity between images and composition prompts."""
        text = self.text_features(pairs)
        return self.clip.logit_scale.exp() * F.normalize(image_features.float(), dim=-1) @ text.t()

    @torch.no_grad()
    def all_logits(self, image_features):
        """Logits for all 333 compositions, in composition ID order. Call eval() first."""
        return self(image_features, self.all_pairs)

    def soft_state(self):
        return {'disease_embeddings': self.disease_embeddings.detach().cpu(),
                'crop_embeddings': self.crop_embeddings.detach().cpu()}


def load_csp(path, device):
    """Rebuild a trained CSP model, in eval mode, from a checkpoint saved by train_csp.py."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    clip_model, tokenizer = load_clip(device, checkpoint['clip_model'], checkpoint['clip_weights'])
    if 'clip_state' in checkpoint:  # fine-tuned CLIP weights from finetune_csp.py
        unexpected = clip_model.load_state_dict(checkpoint['clip_state'], strict=False).unexpected_keys
        if unexpected:
            raise ValueError(f'Unexpected CLIP weights in {path}: {unexpected[:3]}')
    model = CSP(clip_model, tokenizer, dropout=checkpoint['dropout'])
    with torch.no_grad():
        model.disease_embeddings.copy_(checkpoint['disease_embeddings'])
        model.crop_embeddings.copy_(checkpoint['crop_embeddings'])
    model.backbone = (checkpoint['clip_model'], checkpoint['clip_weights'])
    model.finetuned = 'clip_state' in checkpoint  # its image features differ from the frozen-CLIP cache
    return model.eval()
