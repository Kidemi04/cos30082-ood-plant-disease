# Robustness: clip_bioclip_heads_standard_aug

Model run: `runs/clip_bioclip_heads_final`. Calibration gamma = 5.4. Per-class mean accuracy (%), every image ranked against all 333 compositions; H = harmonic mean of seen and unseen accuracy.

## Headline

| | H Top-1 | Relative to clean |
|---|---|---|
| Clean | 55.60 | 100% |
| Mean over all distortions and severities | 36.71 | 66% |
| Mean over held-out distortions only | 29.70 | 53% |

## By distortion (H Top-1 per severity)

| Distortion | Simulates | Held out from strong aug. | Sev. 1 | Sev. 2 | Sev. 3 | Mean | Drop vs clean |
|---|---|---|---|---|---|---|---|
| clean | original test image | | | | | 55.60 | |
| dark | low light, shade |  | 55.21 | 50.29 | 42.90 | 49.47 | 6.13 pts |
| bright | direct sun, overexposure |  | 52.52 | 42.92 | 28.11 | 41.18 | 14.42 pts |
| low_contrast | haze, flat lighting |  | 54.58 | 48.86 | 40.04 | 47.83 | 7.77 pts |
| gaussian_blur | out of focus |  | 51.15 | 34.77 | 13.55 | 33.16 | 22.44 pts |
| motion_blur | camera shake, moving leaf | yes | 40.64 | 22.50 | 12.79 | 25.31 | 30.29 pts |
| gaussian_noise | low-light sensor noise | yes | 38.46 | 19.98 | 5.39 | 21.28 | 34.32 pts |
| jpeg | heavy compression (messaging apps) | yes | 37.37 | 25.63 | 7.13 | 23.38 | 32.23 pts |
| low_resolution | small or distant photo | yes | 47.66 | 34.10 | 19.97 | 33.91 | 21.69 pts |
| occlusion | fingers, other leaves, debris |  | 50.46 | 48.04 | 42.36 | 46.95 | 8.65 pts |
| rotation | tilted camera, viewpoint | yes | 47.95 | 43.03 | 42.91 | 44.63 | 10.97 pts |

## All conditions

| Distortion | Sev. | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 |
|---|---|---|---|---|---|---|---|
| clean |  | 47.40 | 67.23 | 55.60 | 75.44 | 97.89 | 85.21 |
| dark | 1 | 47.44 | 66.04 | 55.21 | 73.36 | 97.59 | 83.76 |
| dark | 2 | 41.38 | 64.10 | 50.29 | 70.01 | 95.86 | 80.92 |
| dark | 3 | 33.46 | 59.75 | 42.90 | 62.29 | 94.17 | 74.98 |
| bright | 1 | 44.92 | 63.20 | 52.52 | 72.61 | 97.20 | 83.12 |
| bright | 2 | 35.22 | 54.93 | 42.92 | 64.79 | 95.37 | 77.16 |
| bright | 3 | 19.56 | 49.89 | 28.11 | 47.29 | 91.33 | 62.31 |
| low_contrast | 1 | 46.41 | 66.25 | 54.58 | 74.12 | 97.05 | 84.05 |
| low_contrast | 2 | 39.78 | 63.33 | 48.86 | 71.09 | 96.06 | 81.71 |
| low_contrast | 3 | 30.08 | 59.87 | 40.04 | 60.10 | 93.75 | 73.24 |
| gaussian_blur | 1 | 44.45 | 60.24 | 51.15 | 69.05 | 97.35 | 80.79 |
| gaussian_blur | 2 | 25.22 | 55.94 | 34.77 | 55.84 | 92.58 | 69.66 |
| gaussian_blur | 3 | 7.90 | 47.75 | 13.55 | 29.49 | 83.18 | 43.54 |
| motion_blur | 1 | 31.28 | 58.00 | 40.64 | 61.60 | 93.06 | 74.13 |
| motion_blur | 2 | 14.72 | 47.68 | 22.50 | 38.79 | 84.44 | 53.16 |
| motion_blur | 3 | 7.72 | 37.23 | 12.79 | 26.15 | 76.23 | 38.94 |
| gaussian_noise | 1 | 28.48 | 59.22 | 38.46 | 58.58 | 92.69 | 71.79 |
| gaussian_noise | 2 | 12.76 | 45.96 | 19.98 | 33.55 | 80.96 | 47.44 |
| gaussian_noise | 3 | 2.99 | 27.40 | 5.39 | 9.12 | 63.40 | 15.95 |
| jpeg | 1 | 28.76 | 53.33 | 37.37 | 55.40 | 92.71 | 69.36 |
| jpeg | 2 | 17.94 | 44.88 | 25.63 | 40.64 | 82.42 | 54.43 |
| jpeg | 3 | 4.01 | 31.84 | 7.13 | 14.32 | 65.37 | 23.49 |
| low_resolution | 1 | 39.38 | 60.34 | 47.66 | 65.59 | 96.12 | 77.97 |
| low_resolution | 2 | 23.95 | 59.18 | 34.10 | 54.50 | 93.52 | 68.86 |
| low_resolution | 3 | 12.19 | 55.17 | 19.97 | 34.56 | 86.15 | 49.33 |
| occlusion | 1 | 40.18 | 67.80 | 50.46 | 72.08 | 96.32 | 82.45 |
| occlusion | 2 | 37.35 | 67.29 | 48.04 | 68.67 | 96.81 | 80.35 |
| occlusion | 3 | 31.71 | 63.79 | 42.36 | 62.25 | 93.98 | 74.90 |
| rotation | 1 | 39.03 | 62.14 | 47.95 | 68.29 | 95.57 | 79.66 |
| rotation | 2 | 35.52 | 54.55 | 43.03 | 67.61 | 92.33 | 78.06 |
| rotation | 3 | 34.97 | 55.51 | 42.91 | 66.58 | 90.70 | 76.79 |
