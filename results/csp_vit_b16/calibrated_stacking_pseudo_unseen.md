# csp_vit_b16: calibrated stacking on pseudo_val_seen + pseudo_val_unseen

gamma is subtracted from the log-probability of each of the 313 trained compositions before ranking all 333. Accuracy is per-class mean (%) unless marked per-image.

| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 (per-image) |
|---|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.000 | 50.05 | 7.67 | 13.30 | 77.51 | 45.98 | 57.72 | 17.89 |
| Calibrated (best pseudo-unseen validation harmonic mean) | 2.738 | 40.29 | 39.51 | 39.89 | 72.42 | 77.94 | 75.08 | 49.18 |

AUC of the seen-unseen Top-1 curve (x100): 22.06 (per-class), 36.40 (per-image).
