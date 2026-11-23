# Baseline 2 experiment notes

## Method and selection

- One ImageNet-pretrained ConvNeXt-Tiny backbone, crop head (55 classes), disease head (175 classes); 27,996,998 total parameters.
- Shared 224 px preprocessing and fixed train/validation split. The validation run used 6,779 training images and 1,221 validation images.
- Square-root sampling by composition; AdamW, weight decay 0.05, batch size 32, label smoothing 0.1 on both equally weighted losses, seed 42, mixed precision.
- Three head-only epochs at initial learning rate 0.001; then up to 30 full-model epochs at initial learning rate 0.0001 with cosine schedule and patience 5.
- Checkpoint selection: validation composition per-class Top-1 against all 333 candidates. The best was full epoch 15 (71.326%); early stopping followed full epoch 20. This validation split contains seen compositions, so it does not measure unseen-pair generalisation.
- The final model was retrained from pretrained weights using all 8,000 training images, three head-only epochs and 15 full-model epochs, with the original learning-rate schedules. Test splits were evaluated after selection and retraining.
- Hardware/software: NVIDIA GeForce RTX 4060 Ti, 8 GB; PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121. Epoch times in the final retrain sum to 481.4 seconds, including data-loader startup but excluding test prediction. Times depend on hardware and system load.
- Final checkpoint SHA-256: `c360cd8f23ad27df33082d213545b977d0604d63599642d31054277a8c7bbc89`. Weights are not included in Git.

## Comparison with the committed Baseline 1 result

Both result summaries rank each image against all 333 valid compositions and report per-class mean accuracy. Baseline 1 numbers are taken from `baseline1-independent/results/convnext_tiny_independent/summary.md` at commit `2f549d2fcbcbc29648ddf9b2367a767931db0ff3`; its result file does not identify the exact checkpoint hash, so confirm checkpoint provenance with its owner.

| Composition metric | Independent (Baseline 1) | Multi-task (Baseline 2) |
|---|---:|---:|
| Seen Top-1 | 50.18% | 52.71% |
| Unseen Top-1 | 9.36% | 3.40% |
| Harmonic mean Top-1 | 15.78% | 6.40% |
| Seen Top-5 | 73.28% | 74.35% |
| Unseen Top-5 | 36.87% | 27.32% |
| Harmonic mean Top-5 | 49.06% | 39.96% |
| Parameters | 55,817,126 across two models | 27,996,998 |

This single run is not a controlled architecture-only comparison. Baseline 1 samples by crop or disease for its separate models and selects each using its own target metric; Baseline 2 samples by composition and selects by composition accuracy. The training runs also used different machines; do not interpret their elapsed times as a fair speed comparison.

The multi-task model's additional `composition_unseen_only` Top-1 of 80.44% ranks only the 10 unseen candidates. It is not comparable to the primary 3.40%, which ranks all 333 candidates; the difference describes the effect of the candidate set and does not by itself establish a cause.

## Seen-test examples for discussion

The seen-test confusion matrix includes 15 healthy paddy images predicted as paddy blast (46.88% of healthy paddy examples), and 9 tomato bacterial spot images predicted as tomato septoria leaf spot (33.33% of that true class). Counts are descriptive and should not be generalised to field performance without additional evidence. The full crop, disease and composition matrices and machine-readable metrics are alongside this note.
