# csp_vit_b16: calibrated stacking on test_seen + test_unseen

gamma is subtracted from the log-probability of each of the 323 trained compositions before ranking all 333. Accuracy is per-class mean (%) unless marked per-image.

| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 (per-image) |
|---|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.000 | 41.23 | 14.72 | 21.70 | 67.86 | 60.54 | 63.99 | 20.67 |
| Calibrated (best test harmonic mean) | 2.718 | 36.29 | 53.52 | 43.25 | 64.79 | 89.66 | 75.23 | 54.12 |

AUC of the seen-unseen Top-1 curve (x100): 28.04 (per-class), 42.46 (per-image).

This gamma was chosen with test labels; report it only as a best case, beside the uncalibrated result.
