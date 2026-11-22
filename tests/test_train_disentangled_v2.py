"""Unit tests for src/train_disentangled_v2.py (CPU-only, fast, zero dataset dependency)."""
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from train_disentangled_v2 import (  # noqa: E402
    DisentangledNetV2,
    MixStyle,
    ModelEMA,
    NUM_CROPS,
    NUM_DISEASES,
    compute_effective_log_priors,
    grl_schedule,
    orthogonality_loss,
)


def test_backbone_protected_grl_zero_backbone_grad():
    """Adversarial crop loss must update disease_proj & adv_crop_head with ZERO gradient into backbone."""
    model = DisentangledNetV2(embed_dim=512, head_type='linear', use_grl=True, detach_grl_backbone=True, pretrained=False)
    model.train()
    x = torch.randn(4, 3, 224, 224)
    crops = torch.tensor([0, 1, 2, 3])
    out = model(x, return_features=True, grl_lambda=0.05)

    assert out['crop'].shape == (4, NUM_CROPS)
    assert out['disease'].shape == (4, NUM_DISEASES)
    assert out['z_crop'].shape == (4, 512)
    assert out['z_disease'].shape == (4, 512)
    assert out['adv_crop'].shape == (4, NUM_CROPS)

    loss_adv = torch.nn.functional.cross_entropy(out['adv_crop'], crops)
    loss_adv.backward()

    backbone_grads = [p.grad.abs().max().item() for p in model.backbone.parameters() if p.grad is not None]
    assert len(backbone_grads) == 0 or max(backbone_grads) == 0.0, 'GRL leaked into shared backbone!'

    proj_grads = [p.grad.abs().max().item() for p in model.disease_proj.parameters() if p.grad is not None]
    assert len(proj_grads) > 0 and max(proj_grads) > 0.0, 'GRL did not reach disease_proj!'

    adv_grads = [p.grad.abs().max().item() for p in model.adv_crop_head.parameters() if p.grad is not None]
    assert len(adv_grads) > 0 and max(adv_grads) > 0.0, 'adv_crop_head received no gradient!'


def test_mixstyle_eval_is_identity():
    ms = MixStyle(p=1.0, alpha=0.1)
    x = torch.randn(4, 96, 14, 14)
    ms.eval()
    assert torch.equal(ms(x), x)
    ms.train()
    y = ms(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not torch.equal(y, x)


def test_cosine_heads_and_ema_and_ortho():
    model = DisentangledNetV2(embed_dim=512, head_type='cosine', use_grl=True, use_mixstyle=True, pretrained=False)
    ema = ModelEMA(model, decay=0.995)
    x = torch.randn(2, 3, 224, 224)
    model.train()
    out = model(x, return_features=True, grl_lambda=0.02)
    loss = out['crop'].sum() + out['disease'].sum() + orthogonality_loss(out['z_crop'], out['z_disease'])
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(-1e-3 * p.grad)
    ema.update(model)
    ema.ema.eval()
    out_eval = ema.ema(x)
    assert out_eval['crop'].shape == (2, NUM_CROPS) and torch.isfinite(out_eval['crop']).all()


def test_effective_log_priors_and_grl_schedule():
    records = [
        {'composition_id': 0, 'crop_id': 0, 'disease_id': 0},
        {'composition_id': 0, 'crop_id': 0, 'disease_id': 0},
        {'composition_id': 0, 'crop_id': 0, 'disease_id': 0},
        {'composition_id': 0, 'crop_id': 0, 'disease_id': 0},
        {'composition_id': 1, 'crop_id': 1, 'disease_id': 1},
    ]
    c_lp, d_lp = compute_effective_log_priors(records, power=0.5)
    assert c_lp.shape == (NUM_CROPS,) and d_lp.shape == (NUM_DISEASES,)
    assert torch.isfinite(c_lp).all() and torch.isfinite(d_lp).all()
    # comp 0 has mass sqrt(4)=2, comp 1 has mass sqrt(1)=1 -> log(2/3) vs log(1/3), diff = log(2)
    assert abs((d_lp[0] - d_lp[1]).item() - torch.log(torch.tensor(2.0)).item()) < 1e-5
    assert grl_schedule(0.0, 0.02) == 0.0
    assert abs(grl_schedule(0.5, 0.02) - 0.02 * (2.0 / (1.0 + torch.exp(torch.tensor(-5.0)).item()) - 1.0)) < 1e-7
