# cpn_p1_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 46.16 | 71.39 | 64.05 | 84.75 |
| crop | 64.97 | 80.63 | 89.80 | 95.85 |
| disease | 48.03 | 76.57 | 67.95 | 87.95 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 69.35 | 97.44 | 71.82 | 96.75 |
| crop | 82.24 | 97.36 | 84.32 | 96.69 |
| disease | 68.17 | 97.23 | 70.51 | 96.48 |
| composition_unseen_only | 81.71 | 98.29 | 85.29 | 97.58 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 55.42 | 82.41 |
| per_image | 67.71 | 90.36 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 23 | 85.19 |
| tomato_late_blight | tomato_early_blight | 17 | 58.62 |
| tomato_septoria_leaf_spot | tomato_early_blight | 15 | 38.46 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| corn_leaf_blight | corn_brown_spot | 13 | 81.25 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| tomato_leaf_mold | tomato_early_blight | 10 | 41.67 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_target_spot | tomato_early_blight | 9 | 60.00 |
| corn_northern_leaf_blight | corn_brown_spot | 9 | 50.00 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 9 | 20.45 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| tomato_healthy | tomato_early_blight | 7 | 21.88 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 10 | 6.33 |
| wheat | sugarcane | 10 | 23.81 |
| apple | cucumber | 8 | 3.36 |
| potato | tomato | 7 | 13.21 |
| apple | tomato | 7 | 2.94 |
| cherry | apple | 5 | 35.71 |
| basil | tomato | 4 | 36.36 |
| cotton | cucumber | 3 | 15.79 |
| soybean | sugarcane | 3 | 4.17 |
| plum | cherry | 3 | 30.00 |
| plum | apple | 3 | 30.00 |
| apple | mango | 3 | 1.26 |
| cassava | cotton | 3 | 1.12 |
| corn | sugarcane | 3 | 1.46 |
| banana | corn | 3 | 9.38 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 21 | 40.38 |
| late_blight | early_blight | 19 | 38.00 |
| septoria_leaf_spot | early_blight | 15 | 36.59 |
| healthy | leafblast | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| leaf_blight | brown_spot | 13 | 33.33 |
| healthy | blast | 13 | 3.68 |
| leaf_mold | early_blight | 10 | 41.67 |
| healthy | caterpillar | 9 | 2.55 |
| target_spot | early_blight | 9 | 42.86 |
| northern_leaf_blight | brown_spot | 9 | 50.00 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |

