# Baselines B1 and B2: setup, results and differences

Sources: `src/train_baseline.py`, `src/gzsl.py`, `runs/b1_*`, `runs/b2_multitask_*`, `results/b1_*`, `results/b2_multitask_*`, `results/comparison.md`. Per-unseen-class numbers below were recomputed from `runs/*_final/test_unseen*.npz`.

## 1. The task

DPD-Small: 11,448 leaf images labelled with a crop (55), a disease (175) and their composition (333 crop-disease pairs).

| Split | Images | Compositions |
|---|---|---|
| train | 8,000 | 323 seen |
| test_seen | 2,000 | seen |
| test_unseen | 1,448 | 10 unseen (crop and disease each seen in training, never together) |

Evaluation is **generalised zero-shot (GZSL)**: every test image is ranked against all 333 compositions, and the model is not told whether the image is seen or unseen. Primary metric is per-class mean Top-1 accuracy, and the headline number is the harmonic mean H = 2·S·U / (S + U) of seen (S) and unseen (U) composition accuracy.

Both baselines output a crop score and a disease score. The composition score is P(crop) × P(disease), renormalised over the 333 valid pairs (`evaluate.py`). Because a product of two independent heads can give a high score to a pair it never saw in training, both baselines can predict the unseen compositions at all.

## 2. Shared recipe (identical for B1 and B2)

| Item | Value |
|---|---|
| Backbone | torchvision ConvNeXt-Tiny, ImageNet-1K V1 weights (~28M parameters per backbone) |
| Head | Linear layer on the 768-d pooled feature |
| Input | 224 px, shared `preprocessing.py` (RandomResizedCrop 0.5–1.0, h/v flips, mild colour jitter, ImageNet normalisation) |
| Loss | Cross-entropy, label smoothing 0.1 |
| Optimiser | AdamW, weight decay 0.05, gradient clip 1.0, AMP, batch 32 |
| Schedule | 3 head-only epochs at lr 1e-3 (backbone frozen), then up to 30 full-model epochs at lr 1e-4 with cosine decay, early-stopping patience 5 |
| Sampler | Square-root frequency sampling (what it balances differs, see §3) |
| Seed | 42, single run |
| Hardware | RTX 5060 Laptop GPU, torch 2.11.0+cu128 |

### Protocol (pseudo-unseen, `src/gzsl.py`)

1. **Select stage.** Train on the shared validation-split train records minus 10 *pseudo-unseen* seen compositions (6,493 images). Choose the best epoch by seen validation per-class Top-1 (1,169 images). The 10 held-out pairs (apple_scab, blackgram_anthracnose, cherry_powdery_mildew, coffee_rust, corn_spot, grape_leaf_blight, paddy_bacterial_blight, papaya_curl, rose_downy_mildew, tomato_brown_spot; 338 images) act as fake unseen data.
2. **Calibrate.** On the select-stage predictions, sweep γ from 0 to 30 (step 0.1). γ is subtracted from the log-probability of every seen composition. Pick the γ that maximises pseudo-unseen H. **Test data is never used to choose γ.**
3. **Final stage.** Retrain on all 8,000 training images for the chosen number of epochs, predict both test splits, report raw (γ = 0) and calibrated (chosen γ) results.

## 3. What is different between B1 and B2

| | **B1: independent** | **B2: multi-task** |
|---|---|---|
| Networks | Two separate ConvNeXt-Tiny models: a crop network and a disease network | One ConvNeXt-Tiny with two heads (crop 55, disease 175) |
| Parameters | ~56M (2 backbones) | ~28M (1 shared backbone) |
| Loss | crop CE (crop net); disease CE (disease net) | crop CE + disease CE, summed |
| Sampler | crop net: square-root by `crop_id`; disease net: square-root by `disease_id` | square-root by `composition_id` |
| Epoch selected by | crop net: val crop Top-1; disease net: val disease Top-1 | val **composition** Top-1 |
| Best epoch (full stage) | crop 11 (val 76.78%), disease 8 (val 72.36%) | 22 (val 72.42%) |
| Final training time | ~8.6 min + ~7.2 min ≈ 16 min | ≈ 15.5 min |
| Predictions | Two files merged with `gzsl.py merge` | One file |
| Calibration γ | 5.7 | 5.9 |
| Pseudo-unseen H at γ / at 0 | 49.08 / 3.47 | 44.57 / 0.90 |
| Pseudo-unseen AUSUC | 32.47 | 30.01 |

The key conceptual difference: in **B1 the disease network never sees crop labels and its features are not shared with the crop network**, so its disease prediction is less tied to which crop is in the photo. In **B2 one shared feature vector feeds both heads**, and training samples balanced by composition, so the heads learn which crops go with which diseases. That helps seen pairs and hurts new pairs.

## 4. Test results (per-class mean %, ranked against all 333 compositions)

| Model | Seen Top-1 | Unseen Top-1 | **H Top-1** | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 per-image | Unseen-only (closed set) Top-1 |
|---|---|---|---|---|---|---|---|---|
| B1 raw | 51.27 | 12.65 | 20.29 | 74.82 | 38.97 | 51.25 | 17.89 | 79.11 |
| B2 raw | **54.74** | 2.86 | 5.43 | 73.32 | 24.77 | 37.03 | 4.15 | 79.28 |
| B1 calibrated | 39.62 | 64.71 | 49.15 | 67.95 | 93.78 | **78.80** | **61.04** | 79.11 |
| B2 calibrated | 41.78 | **66.44** | **51.30** | 65.36 | **95.18** | 77.50 | 59.78 | 79.28 |

Crop and disease level (Top-1 per-class):

| Model | Seen crop | Seen disease | Unseen crop | Unseen disease |
|---|---|---|---|---|
| B1 raw | 67.06 | 50.35 | 67.00 | 19.00 |
| B2 raw | 67.63 | 55.26 | 61.19 | 18.06 |
| B1 calibrated | 59.22 | 41.14 | 79.02 | 64.96 |
| B2 calibrated | 58.09 | 43.17 | 80.05 | 65.70 |

### Per unseen composition (Top-1 %, test_unseen)

| Composition | Images | B1 raw | B2 raw | B1 cal | B2 cal |
|---|---|---|---|---|---|
| tomato_early_blight | 472 | 2.1 | 0.2 | 66.9 | 63.1 |
| grape_black_rot | 271 | 0.7 | 0.0 | 28.4 | 24.0 |
| pepper_bell_bacterial_spot | 220 | 45.5 | 7.7 | **72.3** | 59.1 |
| cotton_bacterial_blight | 125 | 1.6 | 0.0 | **67.2** | 56.8 |
| mango_anthracnose | 95 | 0.0 | 0.0 | **96.8** | 81.1 |
| sugarcane_rust | 89 | 16.9 | 11.2 | 62.9 | **84.3** |
| cherry_aphis_spp | 72 | 0.0 | 0.0 | 51.4 | 56.9 |
| cucumber_powdery_mildew | 37 | 16.2 | 0.0 | 89.2 | 86.5 |
| corn_brown_spot | 35 | 2.9 | 0.0 | 40.0 | **71.4** |
| cucumber_downy_mildew | 32 | 40.6 | 9.4 | 71.9 | 81.2 |

### Where predictions go

| Model | Unseen images predicted as a *seen* pair | Seen images predicted as an *unseen* pair |
|---|---|---|
| B1 raw | 89.6% | 0.4% |
| B2 raw | 97.7% | 0.1% |
| B1 calibrated | 20.4% | 19.6% |
| B2 calibrated | 21.5% | 20.6% |

## 5. Interpretation

**1. Without calibration, both baselines are biased towards seen pairs, B2 much more.** B2 sends 97.7% of unseen images to a seen composition. Its unseen composition accuracy is 2.86%, against 12.65% for B1. This is the main difference between the two baselines.

**2. Why B2 collapses: entangled heads.** Unseen *disease* accuracy is about the same for both (B1 19.0%, B2 18.1% per-class; 11.0% and 10.2% per-image). But per-image unseen *composition* accuracy is 10.3% for B1 and 2.1% for B2. So when B1 gets the disease right it nearly always gets the crop right too. When B2 gets the disease right it usually picks a crop that has been seen with that disease. The shared backbone and composition-balanced sampling teach B2 the co-occurrence of crop and disease, so its two outputs are not independent and the product P(crop)·P(disease) favours known pairs. B1's separately trained networks stay closer to the independence that the product assumes.

**3. The same entanglement is why raw B2 is better on seen data.** Seen composition Top-1: B2 54.74% vs B1 51.27%; seen disease 55.26% vs 50.35%. Knowing which diseases go with which crops helps when the pair has been seen. B2 does this with half the parameters.

**4. Both models can tell the unseen pairs apart.** Closed-set accuracy over only the 10 unseen compositions is 79.11% (B1) and 79.28% (B2). The low raw numbers come from seen-versus-unseen bias, not from failing to recognise the components.

**5. Calibration fixes most of the bias and closes the gap.** Subtracting γ ≈ 6 from seen log-probabilities raises H Top-1 from 20.29 to 49.15 (B1) and from 5.43 to 51.30 (B2). The cost is seen accuracy: −11.7 points for B1, −13.0 for B2, because about 20% of seen images are now pulled into an unseen pair. Typical pulls: seen tomato diseases → tomato_early_blight (B1: 74% of tomato_bacterial_spot), cassava_bacterial_blight → cotton_bacterial_blight and apple_rust → sugarcane_rust (B2). These are seen images moving to unseen pairs that share a crop or disease.

**6. After calibration B1 and B2 are effectively tied.** B2 leads on H Top-1 per-class (51.30 vs 49.15). B1 leads on H Top-5 (78.80 vs 77.50) and H Top-1 per-image (61.04 vs 59.78). The per-class wins are split: B1 is better on mango, cotton, pepper and grape; B2 on corn_brown_spot, sugarcane_rust, cherry_aphis_spp and cucumber_downy_mildew. On pseudo-unseen validation B1 was ahead (H 49.08 vs 44.57, AUSUC 32.47 vs 30.01), but on test B2 is ahead on H Top-1. The ranking changes between validation and test, so a ~2-point difference from one seed should not be reported as a real win either way.

**7. grape_black_rot is hard for both** (28% and 24% calibrated, with 271 images, the second largest unseen class).

### Context

In `results/comparison.md`, calibrated B2 (H 51.30) beats every CLIP/BioCLIP variant except one (fixed 45.56, frozen 42.45, full 49.84, noprim 49.76). Calibrated B1 (49.15) beats fixed and frozen and is level with full and noprim. The best model is `clip_bioclip_heads_calibrated` at H 55.60 (seen 47.40, unseen 67.23). A proposed model therefore needs to beat **calibrated** baselines, not the raw ones; comparing against raw B2 (H 5.43) would overstate the gain.

## 6. Caveats to mention in the report

- Single seed (42) per model, so there are no error bars. Differences of 1–2 points are within run-to-run noise.
- γ is tuned on pseudo-unseen compositions from the training data, not on test. The pseudo-unseen pairs are smaller and differ from the real unseen ones, so the γ is only approximately right for test.
- The final models are retrained on all 8,000 images for the epoch count chosen in the select stage. There is no validation check of the final weights.
- B1 uses about twice the parameters of B2 and two forward passes at inference.
- The `b1_crop_select` and `b1_disease_select` pseudo predictions were regenerated from `best.pth` (see `summary.json` notes and commit ff16beb, which fixed saving single-head predictions for the merge).
