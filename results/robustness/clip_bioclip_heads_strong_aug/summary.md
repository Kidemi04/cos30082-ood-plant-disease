# Robustness: clip_bioclip_heads_strong_aug

Model run: `runs/clip_bioclip_heads_strong_final`. Calibration gamma = 6.1. Per-class mean accuracy (%), every image ranked against all 333 compositions; H = harmonic mean of seen and unseen accuracy.

## Headline

| | H Top-1 | Relative to clean |
|---|---|---|
| Clean | 54.40 | 100% |
| Mean over all distortions and severities | 42.03 | 77% |
| Mean over held-out distortions only | 36.11 | 66% |

## By distortion (H Top-1 per severity)

| Distortion | Simulates | Held out from strong aug. | Sev. 1 | Sev. 2 | Sev. 3 | Mean | Drop vs clean |
|---|---|---|---|---|---|---|---|
| clean | original test image | | | | | 54.40 | |
| dark | low light, shade |  | 55.44 | 54.28 | 48.67 | 52.79 | 1.61 pts |
| bright | direct sun, overexposure |  | 52.43 | 44.28 | 32.24 | 42.98 | 11.42 pts |
| low_contrast | haze, flat lighting |  | 54.25 | 51.29 | 45.47 | 50.33 | 4.07 pts |
| gaussian_blur | out of focus |  | 53.61 | 46.42 | 28.77 | 42.94 | 11.46 pts |
| motion_blur | camera shake, moving leaf | yes | 46.04 | 32.41 | 20.88 | 33.11 | 21.29 pts |
| gaussian_noise | low-light sensor noise | yes | 42.49 | 25.47 | 8.22 | 25.39 | 29.01 pts |
| jpeg | heavy compression (messaging apps) | yes | 42.70 | 30.90 | 9.49 | 27.70 | 26.70 pts |
| low_resolution | small or distant photo | yes | 51.90 | 44.80 | 31.21 | 42.64 | 11.76 pts |
| occlusion | fingers, other leaves, debris |  | 52.77 | 52.00 | 47.47 | 50.75 | 3.65 pts |
| rotation | tilted camera, viewpoint | yes | 53.09 | 51.67 | 50.37 | 51.71 | 2.69 pts |

## All conditions

| Distortion | Sev. | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 |
|---|---|---|---|---|---|---|---|
| clean |  | 45.45 | 67.75 | 54.40 | 76.13 | 97.04 | 85.33 |
| dark | 1 | 46.17 | 69.35 | 55.44 | 76.41 | 95.48 | 84.89 |
| dark | 2 | 44.00 | 70.84 | 54.28 | 75.14 | 96.00 | 84.30 |
| dark | 3 | 37.38 | 69.70 | 48.67 | 71.54 | 95.06 | 81.64 |
| bright | 1 | 44.28 | 64.25 | 52.43 | 74.53 | 97.51 | 84.48 |
| bright | 2 | 35.52 | 58.78 | 44.28 | 68.33 | 96.32 | 79.94 |
| bright | 3 | 23.01 | 53.80 | 32.24 | 56.11 | 92.95 | 69.98 |
| low_contrast | 1 | 45.04 | 68.18 | 54.25 | 76.16 | 95.36 | 84.68 |
| low_contrast | 2 | 42.52 | 64.60 | 51.29 | 74.45 | 95.16 | 83.54 |
| low_contrast | 3 | 35.60 | 62.89 | 45.47 | 69.63 | 95.34 | 80.48 |
| gaussian_blur | 1 | 45.40 | 65.44 | 53.61 | 73.63 | 96.85 | 83.66 |
| gaussian_blur | 2 | 36.14 | 64.88 | 46.42 | 68.33 | 95.62 | 79.71 |
| gaussian_blur | 3 | 19.12 | 58.08 | 28.77 | 51.78 | 89.60 | 65.63 |
| motion_blur | 1 | 36.63 | 61.98 | 46.04 | 68.78 | 94.47 | 79.60 |
| motion_blur | 2 | 23.08 | 54.38 | 32.41 | 49.20 | 86.42 | 62.70 |
| motion_blur | 3 | 13.48 | 46.37 | 20.88 | 35.49 | 82.08 | 49.55 |
| gaussian_noise | 1 | 32.06 | 62.98 | 42.49 | 66.93 | 94.81 | 78.47 |
| gaussian_noise | 2 | 17.35 | 47.83 | 25.47 | 42.64 | 86.43 | 57.10 |
| gaussian_noise | 3 | 4.75 | 30.73 | 8.22 | 15.83 | 62.60 | 25.27 |
| jpeg | 1 | 32.37 | 62.71 | 42.70 | 61.82 | 95.55 | 75.07 |
| jpeg | 2 | 22.08 | 51.46 | 30.90 | 48.58 | 89.02 | 62.86 |
| jpeg | 3 | 5.50 | 34.58 | 9.49 | 21.08 | 65.64 | 31.92 |
| low_resolution | 1 | 42.90 | 65.67 | 51.90 | 72.79 | 96.56 | 83.01 |
| low_resolution | 2 | 33.84 | 66.26 | 44.80 | 65.63 | 95.08 | 77.66 |
| low_resolution | 3 | 20.77 | 62.72 | 31.21 | 52.94 | 90.08 | 66.69 |
| occlusion | 1 | 42.77 | 68.89 | 52.77 | 75.36 | 97.21 | 84.90 |
| occlusion | 2 | 42.33 | 67.40 | 52.00 | 74.76 | 97.32 | 84.57 |
| occlusion | 3 | 37.33 | 65.18 | 47.47 | 69.97 | 96.73 | 81.20 |
| rotation | 1 | 44.54 | 65.69 | 53.09 | 73.23 | 95.87 | 83.03 |
| rotation | 2 | 43.84 | 62.89 | 51.67 | 73.49 | 94.36 | 82.63 |
| rotation | 3 | 42.74 | 61.31 | 50.37 | 72.18 | 94.76 | 81.94 |
