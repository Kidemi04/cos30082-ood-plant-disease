# convnext_tiny_independent: calibrated stacking on pseudo_val_seen + pseudo_val_unseen

gamma is subtracted from the log-probability of each of the 313 trained compositions before ranking all 333. Accuracy is per-class mean (%) unless marked per-image.

| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 (per-image) |
|---|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.000 | 69.49 | 2.84 | 5.46 | 87.21 | 32.35 | 47.20 | 5.90 |
| Calibrated (best pseudo-unseen validation harmonic mean) | 6.081 | 48.22 | 47.19 | 47.70 | 77.44 | 88.36 | 82.54 | 58.67 |

AUC of the seen-unseen Top-1 curve (x100): 33.62 (per-class), 47.31 (per-image).
