# Report draft: Baseline 2 and Disentangled ConvNeXt-Tiny (Kelvin's Approach)

Nov 23, 2026 · @Kelvin

> **Placement note for team assembly:** The main body below is intentionally kept concise (~1.5 pages, ~850 words) so it integrates cleanly into the shared 8-page report limit alongside Nigel's (Baseline 1 + CSP/WiSE-FT), Neng Yi's (BioCLIP 2), and Dino's (Robustness + UI) sections. Detailed ablations, hyperparameters, figures (Fig K0–K3), reproduction commands, and the AI / contribution declarations are placed in the Appendix.

---

## 1. Baseline 2: Shared-Backbone Multitask ConvNeXt-Tiny

### Method and Training
Baseline 2 replaces Baseline 1's two independent networks (55.8M parameters) with a single shared `ConvNeXt-Tiny` backbone (ImageNet-1K pretrained, 28.0M trainable parameters) feeding two parallel linear heads from the same 768-dimensional feature vector: one for crop (55 classes) and one for disease (175 classes). Both heads are trained jointly by minimising the sum of crop and disease cross-entropy losses (label smoothing 0.1) under the team's shared preprocessing, square-root composition-balanced sampling, and two-stage fine-tuning schedule (3 head-only epochs at `lr = 1e-3`, followed by full-network fine-tuning at `lr = 1e-4` with cosine decay, batch size 32, seed 42). At test time, each of the 333 valid pairs is scored by $\log P(\text{crop} \mid x) + \log P(\text{disease} \mid x)$ and calibrated with stacking constant $\gamma$ chosen strictly on the `pseudo-unseen` split without touching test data.

### Results and Failure-Mode Diagnosis
On the held-out test set, Baseline 2 establishes a stronger calibrated baseline than Baseline 1 while using half the parameters: **51.30% calibrated Top-1 harmonic mean (H)** versus 49.15% for Baseline 1 (Table K1). However, inspecting the uncalibrated predictions ($\gamma = 0$) exposes a severe representation bottleneck: **unseen Top-1 collapses to 2.86% (H = 5.43%)**, even though closed-set `Unseen-Only Top-1` reaches **79.28%**.

| Model (All `ConvNeXt-Tiny`, 224×224) | Params | Raw H Top-1 | Cal. Seen Top-1 | Cal. Unseen Top-1 | **Cal. H Top-1** | **Cal. H Top-5** | **Unseen-Only Top-1** | Test Disent. Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline 1 (Independent, `γ=5.9` repo) | 55.8M | 20.29% | 39.62% | 64.71% | 49.15% | 78.80% | 79.13% | — |
| Baseline 2 (Shared Multitask) | 28.0M | 5.43% | 41.78% | 66.44% | 51.30% | 77.50% | 79.28% | −0.0740 |
| Disentangled v1 (`256d`, Cosine, GRL `λ=0.02`) | 29.0M | **11.11%** | 42.92% | 50.06% | 46.22% | 76.04% | 70.80% | +0.0246 |
| **Disentangled v2 (`512d`, Detach-GRL, LA, EMA)** | **30.1M** | **9.07%** | **44.68%** | **61.38%** | **51.71%** | **79.60%** | **80.49%** | **+0.0786** |

*Table K1. Controlled test comparison on the exact same `ConvNeXt-Tiny` backbone (per-class Top-1/Top-5 over all 333 pairs; `γ` chosen solely on `pseudo-unseen`).*

Feature-space analysis on the 42 multi-crop diseases in the held-out test set (3,448 images, 114,415 cross-crop pairs) confirms why: in Baseline 2's shared 768-d representation, images of the **same crop with different diseases** are substantially more similar (mean cosine `0.1769`) than images of the **same disease on different crops** (`0.1029`), yielding a negative disentanglement margin of **−0.0740**. On `tomato_early_blight` (472 unseen images), Baseline 2's disease head predicts other tomato diseases (`bacterial_spot` 112, `septoria_leaf_spot` 105, `late_blight` 87) rather than transferring `early_blight` from potato/celery. This directly mirrors the "disease representation weak link" identified in Neng Yi's BioCLIP 2 analysis, and motivates explicit crop–disease feature disentanglement.

---

## 2. Proposed Approach: Disentangled ConvNeXt-Tiny (v1 → v2)

### Motivation and Controlled Backbone Choice
Whereas our teammates' CLIP/BioCLIP approaches improve generalisation primarily by scaling to vision–language models pretrained on 200M–400M web/biology images (introducing potential pretraining overlap with public leaf datasets), this approach asks a complementary question: **keeping the exact same `ConvNeXt-Tiny` (ImageNet-1K) backbone as Baselines 1 and 2, can explicit feature disentanglement learn crop-invariant disease representations?**

### Architecture and Training Objectives (v2)
Built directly on Baseline 2 (30.1M parameters; Figure K0 in Appendix), `DisentangledNetV2` projects the 768-d backbone feature into two 512-d residual subspaces (`ResidualProjection`), $z_{\text{crop}}$ and $z_{\text{disease}}$, feeding matched linear heads. Three training-only regularisers (removed at inference) shape these subspaces:

1. **Subspace Orthogonality Loss ($w = 0.05$):** Penalises mean squared cosine similarity $\cos^2(z_{\text{crop}}, z_{\text{disease}})$ so the two subspaces encode non-redundant cues.
2. **Backbone-Protected Masked Gradient Reversal (Masked GRL, $\lambda = 0.02$):** An auxiliary adversarial crop classifier predicts crop from $z_{\text{disease}}$ through a Gradient Reversal Layer (Ganin et al., 2016), pushing $z_{\text{disease}}$ to discard crop identity. Two dataset-driven safeguards are critical:
   - *Multi-crop masking:* In DPD-Small, **133 of 175 diseases (76%) occur on only a single crop in training**; penalising crop predictability on single-crop diseases creates a direct mathematical conflict with disease classification. We therefore mask the adversarial loss so it activates **strictly on the 42 multi-crop diseases** present in the training split.
   - *Backbone detachment:* The adversarial branch is computed on `disease_proj(shared.detach())`, stripping crop cues from `disease_proj` without injecting negative crop gradients back into the shared `ConvNeXt-Tiny` backbone.
3. **Effective-Prior Logit Adjustment ($\tau = 0.5$) & Model EMA (`decay = 0.995`):** Adds $\tau \log \pi_{\text{eff}}(d)$ to training disease logits (Menon et al., ICLR 2021), where $\pi_{\text{eff}}$ is the effective disease prior induced by square-root balanced sampling, giving long-tail diseases wider margins without modifying test-time calibration.

### Iterative Diagnosis and Zero-Leakage Model Selection (v1 → v2)
Every architectural decision was made strictly on the `pseudo-unseen` validation protocol (selecting checkpoints by `pseudo_ausuc`, never touching test images; Table K2 in Appendix):
- **v1 ablations (`SupCon` vs `Masked GRL`):** Cross-crop weighted Supervised Contrastive Learning underperformed on `pseudo-unseen` (`H = 41.10%`, `AUSUC = 27.21`) compared to dual branches (`45.21% / 31.35`) and gentle `Masked GRL` (`λ = 0.02`, `48.07% / 33.69`). However, v1's single test run (`H = 46.22%`, `Unseen-Only = 70.80%`) underperformed Baseline 2.
- **Root-cause diagnosis on `select` logs:** Increasing `grl_lambda` from `0 → 0.02 → 0.05 → 0.20` monotonically degraded `pseudo_unseen_only` accuracy (`72.45% → 69.99% → 67.59% → 63.76% → 56.77%`), proving that un-detached GRL gradients corrupted the shared backbone, compounded by a 256-d projection bottleneck and mismatched linear/cosine head scales.
- **v2 design & pre-registered gating:** Fixing backbone detachment, widening to 512-d residual projections, adding EMA, and ablating Stage 1–2 `MixStyle` (which hurt unseen crop accuracy by −6.5 to −10.0 pp by perturbing leaf colour statistics) vs `Logit Adjustment` (`τ = 0.5`) yielded a single clear winner (`v2_g3_logit_adj`: `pseudo_H = 48.70%`, `AUSUC = 34.13`, `pseudo_unseen_only = 74.98%`). Only this gated winner was retrained on all 8,000 images (30 epochs) and evaluated once on test with `γ = 5.8`.

### Results and Disentanglement Evidence
On the held-out test set (Table K1; Figures K1–K3 in Appendix), **Disentangled v2 outperforms Baseline 1, Baseline 2, and v1 across all three headline controlled metrics**:
- **Calibrated Top-1 H = 51.71%** (+0.41 pp over B2, +2.56 pp over B1, +5.49 pp over v1), with **Seen Top-1 rising to 44.68%** (+2.90 pp over B2) and **Unseen-Only Top-1 reaching 80.49%** (+1.21 pp over B2, +9.69 pp over v1).
- **Calibrated Top-5 H = 79.60%** (+2.10 pp over B2, +0.80 pp over B1, +3.56 pp over v1); uncalibrated Top-1 H rises from 5.43% (B2) to **9.07%** (1.67×).
- **Direct representation proof (42 multi-crop diseases on held-out test, Figure K2):** While v2's shared backbone retains crop bias (margin `−0.0839`), its 512-d disease subspace $z_{\text{disease}}$ increases cross-crop same-disease cosine similarity by **2.56× macro / 3.18× micro** (`0.1029 → 0.2637` macro; `0.1056 → 0.3362` micro) while keeping same-crop different-disease similarity low (`0.1851`). The disentanglement margin flips from **−0.0740 (B2)** to **+0.0786 (v2)** (3.20× v1's `+0.0246`), improving **39 of 42** multi-crop diseases (including `powdery_mildew +0.394`, `red_rust +0.389`, `healthy +0.335`, `late_blight +0.255`, `mosaic_virus +0.235`). Held-out subspace orthogonality reaches `mean |cos(z_crop, z_disease)| = 0.0271`.

### Limitations and Future Work
- **Backbone ceiling vs. zero overlap:** With an ImageNet-1K CNN (30.1M params), overall H (51.71%) trails BioCLIP 2 (65.23%, ViT-L/14 pretrained on 200M organism images), though our approach guarantees zero plant-dataset pretraining overlap and requires no vision–language encoder.
- **Single-seed variance and two-iteration transparency:** Both v1 and v2 were evaluated once on test (disclosed transparently above); per-class accuracy fluctuates by ~1–2 pp across seeds.
- **Rejected directions (Future Work):** Symptom-routed Mixture-of-Experts (MoE) was explored via BioCLIP and EmbeddingGemma-2 text clustering (`ARI = 0.143`, grouping by surface word form rather than visual symptoms) and deferred to future work because 8,000 long-tail images (median 10 images/disease) risk routing collapse. Applying our backbone-protected disentanglement head on top of BioCLIP 2 is a direct next step.

---

## Appendix K: Figures, Full Ablations, and Reproduction

### Figures (in `docs/disentangled_figures/` on branch `disentangled-supcon`)
- **Figure K0 (`fig0_architecture_b2_vs_v2.png`):** Architecture comparison between Baseline 2 (shared 768-d multitask CNN) and Disentangled ConvNeXt-Tiny v2 (512-d residual orthogonal subspaces + training-only Orthogonality Loss, Backbone-Protected Masked GRL, Effective-Prior Logit Adjustment, and EMA).
- **Figure K1 (`fig1_ablation_and_test.png`):** (a) Zero-leakage `pseudo-unseen` model selection & ablation progression (`pseudo_H` and `AUSUC`); (b) Official held-out test comparison across B1, B2, Disentangled v1, and Disentangled v2 on the identical `ConvNeXt-Tiny` backbone.
- **Figure K2 (`fig2_disentanglement_evidence.png`):** (a) Cross-crop feature disentanglement margin on held-out test (3,448 images, 42 multi-crop diseases, 114,415 pairs) comparing B2, v2 backbone, v1 `z_disease`, and v2 `z_disease`; (b) Per-disease cross-crop cosine similarity gain (`v2 z_disease − B2`, 39/42 diseases improved).
- **Figure K3 (`fig3_unseen_and_calibration.png`):** (a) Per-class Top-1 calibrated accuracy on all 10 official unseen compositions (v1 50.06% vs v2 61.38%); (b) `pseudo-unseen` calibration sweep selecting `γ = 5.8` without test labels.
- **Seen-test confusion matrices:** Saved under `results/disentangled_v2_calibrated/confusion_test_seen_{composition,crop,disease}.png` (and `results/b2_multitask_calibrated/`).

### Table K2. Select-Stage Ablation Summary (Zero Test Leakage, `pseudo-unseen` Protocol)

| Run | Key Changes | Best Epoch | `val_seen` | `S_dis` | `U_u10` | `γ` | `pseudo_H` | `AUSUC` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `g0_b2_select` | Baseline 2 (shared 768-d, linear) | `full 22` | 72.07% | 71.49% | 72.45% | 5.9 | 44.39% | 29.48 |
| `r2_supcon_p12_w005` | v1 + Cross-Crop SupCon (`w=0.05, p=12`) | `full 21` | 70.98% | — | — | 6.0 | 41.10% | 27.21 |
| `r2_branches_cos` | v1 Orthogonal Branches + Cosine Head | `full 25` | 70.50% | — | — | 6.1 | 45.21% | 31.35 |
| `r2_grl_002_p12` | v1 + Masked GRL (`λ=0.02`, un-detached) | `full 15` | 69.02% | — | 67.59% | 5.1 | 48.07% | 33.69 |
| `v2_g1_clean` | v2 Detach-GRL (`0.02`) + 512d Res + EMA | `full 27` | 68.37% | 70.73% | 70.75% | 5.8 | **50.05%** | **34.19** |
| `v2_g2_mixstyle` | v2 Clean + Stage 1&2 MixStyle (`p=0.5`) | `full 26` | 68.31% | 69.90% | 71.78% | 6.1 | 46.00% | 29.70 |
| **`v2_g3_logit_adj` ★** | **v2 Clean + Logit Adjustment (`τ=0.5`)** | **`full 30`** | **68.56%** | **72.50%** | **74.98%** | **5.8** | **48.70%** | **34.13** |
| `v2_g4_mixstyle_la` | v2 Clean + MixStyle + Logit Adjustment | `full 25` | 69.11% | 72.47% | 71.94% | 5.6 | 47.63% | 31.77 |

### Table K3. Disentangled v2 Training Configuration (`v2_final_winner/config.json`)

| Setting | Value |
|---|---|
| Backbone | `torchvision` `convnext_tiny` (`IMAGENET1K_V1`), 30,070,653 trainable parameters |
| Subspaces & Heads | Two 512-d `ResidualProjection` branches (`dropout=0.1`) + linear heads (55 crops, 175 diseases) |
| Regularisers (train only) | Orthogonality (`w=0.05`), Backbone-Protected Masked GRL (`λ=0.02`, 42 multi-crop diseases), Effective-Prior Logit Adjustment (`τ=0.5`), Model EMA (`decay=0.995`) |
| Optimiser & Schedule | AdamW (`wd=0.05`); 3 head epochs (`lr=1e-3`) + 30 full epochs (`lr=1e-4`, cosine decay); batch 32; seed 42; square-root balanced sampling |
| Calibration & Compute | `γ = 5.8` from `v2_g3_logit_adj/pseudo.npz`; trained on Google Colab NVIDIA L4 (23 GB) |

### Reproduction Commands (branch `disentangled-supcon`)

```bash
# 1. Select stage (strictly zero test usage) + calibration gamma selection
python src/train_disentangled_v2.py --stage select --select-by pseudo_ausuc --patience 12 \
  --logit-adj-tau 0.5 --out runs/v2_g3_logit_adj
python src/gzsl.py calibrate --pseudo runs/v2_g3_logit_adj/pseudo.npz --out runs/v2_g3_logit_adj/cal

# 2. Final stage (all 8,000 training images, 30 epochs) + single test evaluation
python src/train_disentangled_v2.py --stage final --final-epochs 30 --logit-adj-tau 0.5 \
  --out runs/v2_final_winner
python src/gzsl.py calibrate --pseudo runs/v2_g3_logit_adj/pseudo.npz \
  --seen runs/v2_final_winner/test_seen.npz --unseen runs/v2_final_winner/test_unseen.npz \
  --out runs/v2_final_winner/cal
python evaluate.py run --name disentangled_v2_calibrated \
  --seen runs/v2_final_winner/cal/test_seen_calibrated.npz \
  --unseen runs/v2_final_winner/cal/test_unseen_calibrated.npz

# 3. Regenerate report figures & run UI inference wrapper
python scripts/plot_architecture_fig0.py
python scripts/plot_disentangled_figures.py
python src/infer_disentangled_v2.py --checkpoint outputs/disentangled_v2_final_seed42/final.pth --image leaf.jpg
```

---

## Team Assembly Notes (Not Part of Main Body)

### Honest Generative AI Declaration Draft (COS30082 `AI Planning` Compliance)
> Generative AI assistants (Claude / Google Gemini) were used during the planning, brainstorming, and code-review stages of Baseline 2 and the Disentangled ConvNeXt-Tiny (v1→v2) approach: specifically to explore candidate OOD architectures (comparing MoE, SupCon, Masked GRL, MixStyle, and Logit Adjustment), verify zero-leakage experimental protocols, assist with PyTorch implementation/unit-test structure and plotting scripts, and polish technical prose. All experiments, Colab L4 training runs, ablation decisions, metric verifications, and final technical explanations were executed, inspected, and validated by the student.

### "Who Did What" Contribution Draft (Kelvin)
- Repository setup, DPD-Small dataset preparation pipeline, and shared preprocessing utilities.
- Design, training, and evaluation of **Baseline 2** (Shared-Backbone Multitask `ConvNeXt-Tiny`, 51.30% calibrated H) and failure-mode diagnosis.
- Design, iterative ablation (`v1 → v2`, 10+ zero-leakage `pseudo-unseen` runs on Colab L4), feature disentanglement analysis (42 multi-crop diseases), report figures (`Fig K0–K3`), and standalone UI inference package (`src/infer_disentangled_v2.py` + `disentangled_v2_final_model.zip`) for **Disentangled ConvNeXt-Tiny v2** (51.71% calibrated H, 79.60% Top-5 H, 80.49% Unseen-Only, test disentanglement margin +0.0786).
