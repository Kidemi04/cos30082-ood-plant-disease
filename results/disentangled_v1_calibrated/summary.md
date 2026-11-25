# final_winner_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 42.92 | 67.51 | 63.55 | 84.45 |
| crop | 59.07 | 74.55 | 87.05 | 94.40 |
| disease | 45.35 | 73.28 | 66.90 | 88.55 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 50.06 | 87.05 | 55.73 | 88.67 |
| crop | 75.07 | 91.33 | 76.10 | 91.51 |
| disease | 49.81 | 88.35 | 54.83 | 89.78 |
| composition_unseen_only | 70.80 | 94.58 | 74.52 | 94.27 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 46.22 | 76.04 |
| per_image | 59.38 | 86.51 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 17 | 62.96 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 15 | 65.22 |
| tomato_late_blight | tomato_early_blight | 13 | 44.83 |
| tomato_septoria_leaf_spot | tomato_early_blight | 11 | 28.21 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| tomato_healthy | tomato_early_blight | 9 | 28.12 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| cassava_healthy | cassava_brown_streak_disease | 8 | 28.57 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| cassava_green_mottle | cassava_mosaic | 6 | 30.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 6 | 15.38 |
| soybean_healthy | soybean_caterpillar | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 8 | 15.09 |
| paddy | corn | 6 | 3.80 |
| apple | cucumber | 6 | 2.52 |
| apple | mango | 6 | 2.52 |
| basil | tomato | 6 | 54.55 |
| cotton | cucumber | 5 | 26.32 |
| apple | sugarcane | 5 | 2.10 |
| wheat | sugarcane | 5 | 11.90 |
| cherry | apple | 5 | 35.71 |
| soybean | cucumber | 4 | 5.56 |
| guava | cucumber | 4 | 30.77 |
| paddy | sugarcane | 4 | 2.53 |
| mango | cucumber | 3 | 5.66 |
| squash | cucumber | 3 | 100.00 |
| blueberry | mango | 3 | 17.65 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 17 | 32.69 |
| healthy | blast | 16 | 4.53 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| late_blight | early_blight | 13 | 26.00 |
| mosaic | healthy | 11 | 8.46 |
| septoria_leaf_spot | early_blight | 10 | 24.39 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| hispa | blast | 9 | 75.00 |
| leaf_mold | early_blight | 9 | 37.50 |
| healthy | early_blight | 9 | 2.55 |
| healthy | brown_streak_disease | 8 | 2.27 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| stripe_rust | rust | 7 | 70.00 |
| green_mottle | mosaic | 6 | 30.00 |
| brown_spot | rust | 6 | 8.33 |

