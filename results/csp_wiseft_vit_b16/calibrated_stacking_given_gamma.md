# csp_wiseft_vit_b16: calibrated stacking on test_seen + test_unseen

gamma is subtracted from the log-probability of each of the 323 trained compositions before ranking all 333. Accuracy is per-class mean (%) unless marked per-image.

| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 (per-image) |
|---|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.000 | 52.81 | 19.51 | 28.50 | 76.30 | 64.90 | 70.14 | 29.75 |
| Calibrated (given gamma) | 5.346 | 43.47 | 67.39 | 52.85 | 71.48 | 92.20 | 80.53 | 63.78 |

AUC of the seen-unseen Top-1 curve (x100): 40.92 (per-class), 54.58 (per-image).
