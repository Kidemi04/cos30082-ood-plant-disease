# clip_bioclip_frozen_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 30.51 | 62.74 | 45.95 | 78.00 |
| crop | 55.85 | 81.28 | 82.05 | 95.30 |
| disease | 33.42 | 65.74 | 49.65 | 83.00 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 69.78 | 94.39 | 67.68 | 92.82 |
| crop | 77.38 | 96.59 | 79.56 | 96.62 |
| disease | 69.79 | 94.79 | 66.16 | 93.02 |
| composition_unseen_only | 78.43 | 98.42 | 77.49 | 97.65 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 42.45 | 75.38 |
| per_image | 54.74 | 84.77 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_septoria_leaf_spot | tomato_early_blight | 29 | 74.36 |
| corn_streak_virus | corn_brown_spot | 16 | 29.09 |
| soybean_healthy | soybean_caterpillar | 14 | 48.28 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| tomato_bacterial_spot | tomato_early_blight | 13 | 48.15 |
| corn_lethal_necrosis | corn_brown_spot | 13 | 41.94 |
| corn_healthy | corn_brown_spot | 11 | 26.83 |
| tomato_leaf_mold | tomato_early_blight | 10 | 41.67 |
| tomato_late_blight | tomato_early_blight | 10 | 34.48 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| tomato_healthy | tomato_early_blight | 8 | 25.00 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 8 | 18.18 |
| cassava_green_mottle | cassava_brown_streak_disease | 8 | 40.00 |
| apple_marssonina_blotch | cucumber_downy_mildew | 8 | 57.14 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 16 | 6.72 |
| cassava | cotton | 16 | 5.95 |
| tomato | cucumber | 14 | 4.81 |
| paddy | corn | 10 | 6.33 |
| cassava | cucumber | 7 | 2.60 |
| sugarcane | corn | 7 | 15.91 |
| apple | cherry | 6 | 2.52 |
| grape | cucumber | 6 | 14.63 |
| potato | tomato | 6 | 11.32 |
| cotton | cucumber | 5 | 26.32 |
| plum | apple | 4 | 40.00 |
| wheat | corn | 4 | 9.52 |
| apple | cotton | 4 | 1.68 |
| paddy | sugarcane | 4 | 2.53 |
| cassava | mango | 4 | 1.49 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 27 | 65.85 |
| healthy | brown_spot | 24 | 6.80 |
| streak_virus | brown_spot | 16 | 29.09 |
| healthy | caterpillar | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| healthy | leafblast | 14 | 3.97 |
| healthy | anthracnose | 13 | 3.68 |
| bacterial_spot | early_blight | 13 | 25.00 |
| lethal_necrosis | brown_spot | 13 | 41.94 |
| healthy | bacterial_blight | 12 | 3.40 |
| late_blight | early_blight | 11 | 22.00 |
| leaf_mold | early_blight | 10 | 41.67 |
| mosaic | healthy | 10 | 7.69 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| marssonina_blotch | downy_mildew | 9 | 64.29 |

