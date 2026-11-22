# final_winner_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 50.83 | 73.44 | 68.50 | 87.35 |
| crop | 64.65 | 82.97 | 90.60 | 96.15 |
| disease | 51.19 | 73.58 | 69.55 | 85.35 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 6.23 | 33.99 | 6.84 | 44.34 |
| crop | 61.63 | 78.96 | 67.54 | 83.43 |
| disease | 17.33 | 32.62 | 11.40 | 30.18 |
| composition_unseen_only | 70.80 | 94.58 | 74.52 | 94.27 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 11.11 | 46.47 |
| per_image | 12.43 | 58.82 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| cassava_healthy | cassava_brown_streak_disease | 8 | 28.57 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| cassava_green_mottle | cassava_mosaic | 6 | 30.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 6 | 22.22 |
| tomato_target_spot | tomato_healthy | 6 | 40.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 6 | 15.38 |
| soybean_healthy | soybean_caterpillar | 5 | 17.24 |
| coffee_rust | coffee_spider_mites | 5 | 25.00 |
| corn_leaf_blight | corn_northern_leaf_blight | 5 | 31.25 |
| peanut_healthy | peanut_early_leaf_spot | 5 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| basil | tomato | 7 | 63.64 |
| cherry | apple | 6 | 42.86 |
| apple | peach | 5 | 2.10 |
| potato | tomato | 5 | 9.43 |
| guava | cassava | 5 | 38.46 |
| raspberry | tomato | 3 | 27.27 |
| plum | apple | 3 | 30.00 |
| apple | guava | 3 | 1.26 |
| paddy | palm | 3 | 1.90 |
| apple | cassava | 3 | 1.26 |
| pear | apple | 3 | 12.00 |
| apple | rose | 2 | 0.84 |
| grape | tomato | 2 | 4.88 |
| wheat | corn | 2 | 4.76 |
| potato | apple | 2 | 3.77 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| healthy | blast | 16 | 4.53 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| mosaic | healthy | 11 | 8.46 |
| hispa | blast | 9 | 75.00 |
| healthy | brown_streak_disease | 8 | 2.27 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| green_mottle | mosaic | 6 | 30.00 |
| bacterial_spot | septoria_leaf_spot | 6 | 11.54 |
| rust | healthy | 6 | 7.14 |
| target_spot | healthy | 6 | 28.57 |
| septoria_leaf_spot | verticulium_wilt | 6 | 14.63 |
| healthy | caterpillar | 5 | 1.42 |
| leaf_blight | northern_leaf_blight | 5 | 12.82 |
| healthy | early_leaf_spot | 5 | 1.42 |

