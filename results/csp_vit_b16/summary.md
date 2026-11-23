# csp_vit_b16

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 41.23 | 67.86 | 59.65 | 86.75 |
| crop | 66.38 | 87.08 | 87.75 | 97.10 |
| disease | 43.40 | 71.79 | 62.75 | 89.55 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 14.72 | 60.54 | 12.50 | 57.25 |
| crop | 62.13 | 93.92 | 67.13 | 93.85 |
| disease | 26.56 | 68.29 | 18.99 | 63.19 |
| composition_unseen_only | 77.38 | 99.79 | 78.18 | 99.45 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 21.70 | 63.99 |
| per_image | 20.67 | 68.98 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_brown_spot | 26 | 81.25 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 15 | 65.22 |
| paddy_hispa | paddy_brown_spot | 11 | 91.67 |
| cassava_healthy | cassava_brown_streak_disease | 11 | 39.29 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| cassava_green_mottle | cassava_brown_streak_disease | 8 | 40.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| tomato_bacterial_spot | tomato_late_blight | 8 | 29.63 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| cassava_green_mottle | cassava_mosaic | 7 | 35.00 |
| cassava_brown_streak_disease | cassava_mosaic | 7 | 24.14 |
| tomato_septoria_leaf_spot | tomato_early_blight | 7 | 17.95 |
| cassava_brown_spot | cassava_bacterial_blight | 6 | 31.58 |
| paddy_tungro | paddy_brown_spot | 6 | 35.29 |
| paddy_blast | paddy_brown_spot | 6 | 27.27 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| mango | cassava | 9 | 16.98 |
| apple | peach | 6 | 2.52 |
| guava | cassava | 6 | 46.15 |
| cherry | apple | 6 | 42.86 |
| apple | tomato | 5 | 2.10 |
| paddy | corn | 4 | 2.53 |
| cotton | tomato | 4 | 21.05 |
| potato | tomato | 4 | 7.55 |
| grape | tomato | 3 | 7.32 |
| apple | cotton | 3 | 1.26 |
| wheat | paddy | 3 | 7.14 |
| mango | coffee | 3 | 5.66 |
| blackgram | soybean | 3 | 20.00 |
| cassava | coffee | 3 | 1.12 |
| coffee | cassava | 3 | 7.69 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | brown_spot | 28 | 7.93 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| hispa | brown_spot | 11 | 91.67 |
| healthy | brown_streak_disease | 10 | 2.83 |
| green_mottle | brown_streak_disease | 8 | 40.00 |
| bacterial_spot | late_blight | 8 | 15.38 |
| septoria_leaf_spot | early_blight | 8 | 19.51 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| healthy | caterpillar | 8 | 2.27 |
| blast | brown_spot | 7 | 31.82 |
| green_mottle | mosaic | 7 | 35.00 |
| bacterial_spot | septoria_leaf_spot | 7 | 13.46 |
| brown_streak_disease | mosaic | 7 | 24.14 |
| rust | healthy | 6 | 7.14 |
| brown_spot | bacterial_blight | 6 | 8.33 |

