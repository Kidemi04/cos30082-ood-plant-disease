# ccd_nocon_cpn_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 45.07 | 72.06 | 60.10 | 86.45 |
| crop | 62.23 | 80.07 | 87.40 | 95.95 |
| disease | 50.03 | 79.09 | 66.40 | 90.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 66.11 | 95.20 | 62.78 | 95.51 |
| crop | 83.47 | 98.05 | 83.29 | 96.75 |
| disease | 65.40 | 95.53 | 61.19 | 95.93 |
| composition_unseen_only | 82.69 | 98.62 | 80.87 | 97.44 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 53.60 | 82.03 |
| per_image | 61.41 | 90.75 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_septoria_leaf_spot | tomato_early_blight | 17 | 43.59 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| tomato_bacterial_spot | tomato_early_blight | 15 | 55.56 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| tomato_late_blight | tomato_early_blight | 12 | 41.38 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| peach_bacterial_spot | pepper_bell_bacterial_spot | 10 | 52.63 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| tomato_healthy | tomato_early_blight | 9 | 28.12 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| tomato_leaf_mold | tomato_early_blight | 8 | 33.33 |
| tomato_target_spot | tomato_early_blight | 8 | 53.33 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 10 | 4.20 |
| peach | pepper_bell | 10 | 25.00 |
| wheat | sugarcane | 10 | 23.81 |
| paddy | corn | 8 | 5.06 |
| apple | sugarcane | 8 | 3.36 |
| pumpkin | cucumber | 7 | 43.75 |
| potato | tomato | 7 | 13.21 |
| apple | tomato | 7 | 2.94 |
| blackgram | cucumber | 6 | 40.00 |
| tomato | corn | 5 | 1.72 |
| banana | corn | 5 | 15.62 |
| soybean | cucumber | 4 | 5.56 |
| cotton | cucumber | 4 | 21.05 |
| coffee | sugarcane | 4 | 10.26 |
| cherry | apple | 4 | 28.57 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 16 | 39.02 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| bacterial_spot | early_blight | 14 | 26.92 |
| late_blight | early_blight | 13 | 26.00 |
| healthy | blast | 13 | 3.68 |
| healthy | leafblast | 11 | 3.12 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| hispa | leafblast | 9 | 75.00 |
| healthy | brown_spot | 9 | 2.55 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| leaf_mold | early_blight | 8 | 33.33 |
| healthy | early_blight | 8 | 2.27 |
| target_spot | early_blight | 8 | 38.10 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |

