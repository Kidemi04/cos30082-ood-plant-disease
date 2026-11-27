# wise_f6_tuned_s2_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 46.84 | 74.36 | 66.85 | 88.85 |
| crop | 64.73 | 82.60 | 90.00 | 96.35 |
| disease | 48.54 | 79.92 | 70.65 | 91.90 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 79.32 | 99.62 | 74.86 | 99.52 |
| crop | 89.02 | 99.83 | 88.74 | 99.72 |
| disease | 78.48 | 99.76 | 73.62 | 99.59 |
| composition_unseen_only | 89.06 | 99.90 | 89.23 | 99.79 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 58.90 | 85.16 |
| per_image | 70.63 | 93.88 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 23 | 71.88 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 13 | 56.52 |
| tomato_septoria_leaf_spot | tomato_early_blight | 11 | 28.21 |
| paddy_hispa | paddy_blast | 11 | 91.67 |
| corn_leaf_blight | corn_brown_spot | 11 | 68.75 |
| tomato_bacterial_spot | tomato_early_blight | 11 | 40.74 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_healthy | tomato_early_blight | 9 | 28.12 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| tomato_late_blight | tomato_early_blight | 6 | 20.69 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 6 | 13.64 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 10 | 6.33 |
| potato | tomato | 7 | 13.21 |
| apple | sugarcane | 6 | 2.52 |
| apple | mango | 6 | 2.52 |
| broccoli | cucumber | 5 | 100.00 |
| basil | tomato | 5 | 45.45 |
| wheat | sugarcane | 5 | 11.90 |
| cotton | cucumber | 4 | 21.05 |
| apple | cucumber | 4 | 1.68 |
| mango | cucumber | 4 | 7.55 |
| tomato | pepper_bell | 4 | 1.37 |
| apple | cotton | 4 | 1.68 |
| wheat | corn | 4 | 9.52 |
| plum | cherry | 3 | 30.00 |
| palm | corn | 3 | 23.08 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 22 | 6.23 |
| caterpillar | diabrotica_speciosa | 13 | 56.52 |
| septoria_leaf_spot | early_blight | 11 | 26.83 |
| hispa | blast | 11 | 91.67 |
| leaf_blight | brown_spot | 11 | 28.21 |
| bacterial_spot | early_blight | 11 | 21.15 |
| healthy | caterpillar | 9 | 2.55 |
| healthy | early_blight | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| healthy | brown_spot | 7 | 1.98 |
| brown_spot | bacterial_blight | 7 | 9.72 |
| mosaic | healthy | 7 | 5.38 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| blast | brown_spot | 6 | 27.27 |

