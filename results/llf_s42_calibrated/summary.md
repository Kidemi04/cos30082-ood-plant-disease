# llf_s42_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 45.09 | 72.51 | 61.90 | 88.55 |
| crop | 61.67 | 80.04 | 87.30 | 95.95 |
| disease | 51.19 | 79.44 | 68.55 | 92.30 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 69.84 | 97.58 | 69.48 | 97.86 |
| crop | 90.78 | 99.09 | 88.47 | 98.48 |
| disease | 69.16 | 97.60 | 67.47 | 98.00 |
| composition_unseen_only | 87.85 | 99.28 | 86.19 | 98.69 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 54.80 | 83.20 |
| per_image | 65.47 | 92.97 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_septoria_leaf_spot | tomato_early_blight | 24 | 61.54 |
| potato_early_blight | tomato_early_blight | 18 | 75.00 |
| tomato_healthy | tomato_early_blight | 16 | 50.00 |
| peach_bacterial_spot | pepper_bell_bacterial_spot | 14 | 73.68 |
| tomato_bacterial_spot | tomato_early_blight | 13 | 48.15 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| paddy_healthy | paddy_leafblast | 12 | 37.50 |
| tomato_late_blight | tomato_early_blight | 12 | 41.38 |
| apple_rust | sugarcane_rust | 12 | 34.29 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 10 | 43.48 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_target_spot | tomato_early_blight | 8 | 53.33 |
| tomato_leaf_mold | tomato_early_blight | 8 | 33.33 |
| corn_leaf_blight | corn_brown_spot | 7 | 43.75 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 22 | 41.51 |
| apple | sugarcane | 15 | 6.30 |
| peach | pepper_bell | 14 | 35.00 |
| paddy | corn | 14 | 8.86 |
| apple | cucumber | 7 | 2.94 |
| apple | tomato | 6 | 2.52 |
| wheat | sugarcane | 5 | 11.90 |
| cotton | cucumber | 4 | 21.05 |
| apple | grape | 4 | 1.68 |
| blackgram | cucumber | 4 | 26.67 |
| cherry | apple | 4 | 28.57 |
| broccoli | cucumber | 3 | 60.00 |
| plum | cherry | 3 | 30.00 |
| palm | corn | 3 | 23.08 |
| squash | cucumber | 3 | 100.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 23 | 56.10 |
| late_blight | early_blight | 14 | 28.00 |
| healthy | early_blight | 14 | 3.97 |
| healthy | leafblast | 12 | 3.40 |
| healthy | blast | 12 | 3.40 |
| bacterial_spot | early_blight | 11 | 21.15 |
| caterpillar | diabrotica_speciosa | 10 | 43.48 |
| healthy | caterpillar | 9 | 2.55 |
| mosaic_virus | early_blight | 9 | 16.67 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| leaf_mold | early_blight | 8 | 33.33 |
| target_spot | early_blight | 7 | 33.33 |
| leaf_blight | brown_spot | 7 | 17.95 |
| mosaic | healthy | 7 | 5.38 |

