# wise_b2_tuned_s1_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 48.08 | 82.90 | 62.70 | 90.95 |
| crop | 74.41 | 93.05 | 92.20 | 98.40 |
| disease | 51.75 | 86.49 | 66.50 | 93.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 85.83 | 99.46 | 84.94 | 99.24 |
| crop | 92.43 | 99.67 | 93.30 | 99.52 |
| disease | 84.97 | 99.53 | 84.46 | 99.38 |
| composition_unseen_only | 91.43 | 99.73 | 92.75 | 99.59 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 61.63 | 90.42 |
| per_image | 72.15 | 94.91 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_healthy | tomato_early_blight | 27 | 84.38 |
| tomato_septoria_leaf_spot | tomato_early_blight | 25 | 64.10 |
| tomato_bacterial_spot | tomato_early_blight | 22 | 81.48 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_leafblast | 17 | 53.12 |
| potato_early_blight | tomato_early_blight | 14 | 58.33 |
| tomato_late_blight | tomato_early_blight | 13 | 44.83 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| tomato_leaf_mold | tomato_early_blight | 12 | 50.00 |
| cassava_brown_spot | cassava_bacterial_blight | 11 | 57.89 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 9 | 20.45 |
| paddy_healthy | paddy_blast | 8 | 25.00 |
| paddy_hispa | paddy_leafblast | 7 | 58.33 |
| tomato_target_spot | tomato_early_blight | 7 | 46.67 |
| cassava_green_mottle | cassava_brown_streak_disease | 6 | 30.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 15 | 28.30 |
| apple | tomato | 7 | 2.94 |
| paddy | corn | 6 | 3.80 |
| wheat | sugarcane | 6 | 14.29 |
| cotton | cucumber | 5 | 26.32 |
| plum | cherry | 4 | 40.00 |
| paddy | sugarcane | 4 | 2.53 |
| squash | cucumber | 3 | 100.00 |
| corn | sugarcane | 3 | 1.46 |
| cabbage | grape | 3 | 75.00 |
| apple | grape | 3 | 1.26 |
| eggplant | tomato | 3 | 60.00 |
| zucchini | cucumber | 3 | 75.00 |
| palm | corn | 2 | 15.38 |
| apple | peach | 2 | 0.84 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 25 | 60.98 |
| healthy | early_blight | 25 | 7.08 |
| bacterial_spot | early_blight | 22 | 42.31 |
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | leafblast | 17 | 4.82 |
| late_blight | early_blight | 13 | 26.00 |
| brown_spot | bacterial_blight | 13 | 18.06 |
| healthy | caterpillar | 12 | 3.40 |
| leaf_mold | early_blight | 12 | 50.00 |
| healthy | bacterial_blight | 10 | 2.83 |
| yellow_leaf_curl_virus | early_blight | 9 | 20.45 |
| target_spot | early_blight | 8 | 38.10 |
| hispa | leafblast | 7 | 58.33 |
| healthy | blast | 7 | 1.98 |
| brown_spot | rust | 7 | 9.72 |

