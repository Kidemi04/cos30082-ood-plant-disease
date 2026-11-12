# b1_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 39.62 | 67.95 | 60.55 | 85.40 |
| crop | 59.22 | 75.88 | 87.80 | 94.75 |
| disease | 41.14 | 74.71 | 64.50 | 89.70 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 64.71 | 93.78 | 61.53 | 91.78 |
| crop | 79.02 | 95.40 | 78.87 | 93.23 |
| disease | 64.96 | 94.78 | 61.53 | 92.40 |
| composition_unseen_only | 79.11 | 97.16 | 78.31 | 95.10 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 49.15 | 78.80 |
| per_image | 61.04 | 88.48 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 20 | 74.07 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_blast | 17 | 53.12 |
| tomato_septoria_leaf_spot | tomato_early_blight | 16 | 41.03 |
| tomato_healthy | tomato_early_blight | 16 | 50.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 11 | 37.93 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| tomato_late_blight | tomato_early_blight | 8 | 27.59 |
| tomato_target_spot | tomato_early_blight | 7 | 46.67 |
| potato_early_blight | tomato_early_blight | 7 | 29.17 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| tomato_leaf_blight | tomato_early_blight | 6 | 60.00 |
| paddy_blast | paddy_tungro | 6 | 27.27 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 9 | 16.98 |
| apple | cucumber | 8 | 3.36 |
| apple | sugarcane | 7 | 2.94 |
| cotton | cucumber | 5 | 26.32 |
| basil | tomato | 5 | 45.45 |
| cherry | apple | 5 | 35.71 |
| apple | tomato | 5 | 2.10 |
| broccoli | cucumber | 4 | 80.00 |
| paddy | palm | 4 | 2.53 |
| paddy | corn | 3 | 1.90 |
| soybean | cucumber | 3 | 4.17 |
| squash | cucumber | 3 | 100.00 |
| corn | sugarcane | 3 | 1.46 |
| blueberry | mango | 3 | 17.65 |
| apple | mango | 3 | 1.26 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 19 | 36.54 |
| healthy | early_blight | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | blast | 17 | 4.82 |
| septoria_leaf_spot | early_blight | 16 | 39.02 |
| healthy | diabrotica_speciosa | 11 | 3.12 |
| late_blight | early_blight | 11 | 22.00 |
| mosaic | healthy | 11 | 8.46 |
| healthy | brown_spot | 10 | 2.83 |
| hispa | blast | 9 | 75.00 |
| healthy | bacterial_blight | 9 | 2.55 |
| target_spot | early_blight | 8 | 38.10 |
| stripe_rust | rust | 8 | 80.00 |
| healthy | leafblast | 7 | 1.98 |
| leaf_blight | early_blight | 6 | 15.38 |

