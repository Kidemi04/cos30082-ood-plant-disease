# dci_a1_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 46.39 | 74.53 | 64.20 | 88.10 |
| crop | 63.45 | 82.34 | 88.85 | 96.35 |
| disease | 50.90 | 81.03 | 68.90 | 91.45 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 68.56 | 96.58 | 67.89 | 96.82 |
| crop | 86.62 | 98.58 | 86.26 | 97.93 |
| disease | 68.21 | 96.69 | 66.30 | 97.17 |
| composition_unseen_only | 84.87 | 99.06 | 84.88 | 98.62 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 55.34 | 84.13 |
| per_image | 65.99 | 92.26 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 17 | 53.12 |
| tomato_septoria_leaf_spot | tomato_early_blight | 13 | 33.33 |
| tomato_bacterial_spot | tomato_early_blight | 12 | 44.44 |
| paddy_healthy | paddy_blast | 12 | 37.50 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| cassava_healthy | cassava_brown_streak_disease | 10 | 35.71 |
| tomato_healthy | tomato_early_blight | 10 | 31.25 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| tomato_late_blight | tomato_early_blight | 8 | 27.59 |
| cassava_brown_spot | cassava_bacterial_blight | 8 | 42.11 |
| cassava_mosaic | cassava_green_mottle | 8 | 6.45 |
| potato_early_blight | tomato_early_blight | 7 | 29.17 |
| tomato_leaf_mold | tomato_early_blight | 6 | 25.00 |
| peach_bacterial_spot | pepper_bell_bacterial_spot | 6 | 31.58 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 10 | 18.87 |
| paddy | corn | 8 | 5.06 |
| apple | cucumber | 8 | 3.36 |
| wheat | sugarcane | 7 | 16.67 |
| peach | pepper_bell | 6 | 15.00 |
| apple | mango | 6 | 2.52 |
| cherry | apple | 6 | 42.86 |
| apple | sugarcane | 5 | 2.10 |
| basil | tomato | 5 | 45.45 |
| soybean | sugarcane | 4 | 5.56 |
| apple | tomato | 4 | 1.68 |
| grape | cucumber | 4 | 9.76 |
| banana | corn | 4 | 12.50 |
| broccoli | cucumber | 3 | 60.00 |
| palm | corn | 3 | 23.08 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 17 | 4.82 |
| septoria_leaf_spot | early_blight | 13 | 31.71 |
| caterpillar | diabrotica_speciosa | 12 | 52.17 |
| healthy | early_blight | 11 | 3.12 |
| healthy | blast | 11 | 3.12 |
| healthy | caterpillar | 10 | 2.83 |
| healthy | brown_streak_disease | 10 | 2.83 |
| hispa | leafblast | 9 | 75.00 |
| bacterial_spot | early_blight | 9 | 17.31 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| stripe_rust | rust | 9 | 90.00 |
| late_blight | early_blight | 8 | 16.00 |
| mosaic | green_mottle | 8 | 6.15 |
| leaf_mold | early_blight | 6 | 25.00 |
| blast | brown_spot | 6 | 27.27 |

