# convnext_tiny_independent: calibrated stacking on test_seen + test_unseen

gamma is subtracted from the log-probability of each of the 323 trained compositions before ranking all 333. Accuracy is per-class mean (%) unless marked per-image.

| | gamma | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | H Top-1 (per-image) |
|---|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.000 | 50.18 | 9.36 | 15.78 | 73.28 | 36.87 | 49.06 | 10.56 |
| Calibrated (given gamma) | 6.081 | 36.36 | 63.66 | 46.28 | 66.04 | 93.62 | 77.44 | 59.31 |

AUC of the seen-unseen Top-1 curve (x100): 33.43 (per-class), 46.40 (per-image).
