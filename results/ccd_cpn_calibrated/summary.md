# ccd_cpn_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 46.48 | 73.39 | 62.90 | 86.40 |
| crop | 64.05 | 80.81 | 89.40 | 96.05 |
| disease | 51.24 | 80.31 | 67.40 | 90.40 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 67.94 | 95.83 | 69.06 | 97.17 |
| crop | 85.69 | 98.64 | 86.05 | 98.27 |
| disease | 67.94 | 96.14 | 68.65 | 97.51 |
| composition_unseen_only | 83.75 | 99.33 | 84.94 | 98.83 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 55.20 | 83.12 |
| per_image | 65.84 | 91.47 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 17 | 53.12 |
| tomato_septoria_leaf_spot | tomato_early_blight | 16 | 41.03 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| tomato_healthy | tomato_early_blight | 14 | 43.75 |
| tomato_late_blight | tomato_early_blight | 13 | 44.83 |
| tomato_bacterial_spot | tomato_early_blight | 12 | 44.44 |
| cassava_mosaic | cassava_green_mottle | 10 | 8.06 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| tomato_target_spot | tomato_early_blight | 9 | 60.00 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| paddy_healthy | paddy_blast | 9 | 28.12 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| corn_leaf_blight | corn_brown_spot | 7 | 43.75 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | tomato | 8 | 3.36 |
| wheat | sugarcane | 7 | 16.67 |
| potato | tomato | 7 | 13.21 |
| paddy | corn | 6 | 3.80 |
| apple | cucumber | 6 | 2.52 |
| cherry | apple | 5 | 35.71 |
| peach | pepper_bell | 5 | 12.50 |
| soybean | cucumber | 4 | 5.56 |
| cotton | cucumber | 4 | 21.05 |
| pumpkin | cucumber | 4 | 25.00 |
| apple | mango | 4 | 1.68 |
| blackgram | cucumber | 4 | 26.67 |
| banana | corn | 4 | 12.50 |
| broccoli | cucumber | 3 | 60.00 |
| soybean | sugarcane | 3 | 4.17 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 17 | 4.82 |
| septoria_leaf_spot | early_blight | 16 | 39.02 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| late_blight | early_blight | 13 | 26.00 |
| healthy | early_blight | 12 | 3.40 |
| bacterial_spot | early_blight | 12 | 23.08 |
| mosaic | green_mottle | 10 | 7.69 |
| hispa | leafblast | 9 | 75.00 |
| target_spot | early_blight | 9 | 42.86 |
| leaf_mold | early_blight | 9 | 37.50 |
| healthy | blast | 9 | 2.55 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| stripe_rust | rust | 8 | 80.00 |

