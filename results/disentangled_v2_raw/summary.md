# v2_final_winner_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 55.02 | 75.74 | 70.00 | 87.15 |
| crop | 70.40 | 87.38 | 91.00 | 95.25 |
| disease | 60.26 | 78.49 | 71.55 | 84.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 4.94 | 28.76 | 5.59 | 34.32 |
| crop | 66.25 | 76.88 | 68.78 | 79.97 |
| disease | 13.98 | 25.01 | 8.15 | 18.37 |
| composition_unseen_only | 80.49 | 96.87 | 77.42 | 94.48 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 9.07 | 41.69 |
| per_image | 10.36 | 49.25 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 23 | 71.88 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_hispa | paddy_blast | 11 | 91.67 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 10 | 25.64 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 7 | 17.95 |
| cassava_green_mottle | cassava_mosaic | 6 | 30.00 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| cassava_mosaic | cassava_green_mottle | 5 | 4.03 |
| corn_leaf_blight | corn_northern_leaf_blight | 5 | 31.25 |
| peanut_healthy | peanut_early_leaf_spot | 5 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | peach | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| paddy | palm | 3 | 1.90 |
| guava | cassava | 3 | 23.08 |
| apple | cassava | 3 | 1.26 |
| peach | raspberry | 2 | 5.00 |
| apple | rose | 2 | 0.84 |
| apple | cotton | 2 | 0.84 |
| apple | guava | 2 | 0.84 |
| tomato | plum | 2 | 0.69 |
| wheat | corn | 2 | 4.76 |
| lettuce | chilli | 2 | 66.67 |
| paddy | wheat | 2 | 1.27 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 22 | 6.23 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| hispa | blast | 11 | 91.67 |
| healthy | brown_streak_disease | 10 | 2.83 |
| septoria_leaf_spot | verticulium_wilt | 10 | 24.39 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| healthy | caterpillar | 7 | 1.98 |
| mosaic | healthy | 7 | 5.38 |
| green_mottle | mosaic | 6 | 30.00 |
| mosaic | green_mottle | 5 | 3.85 |
| leaf_blight | northern_leaf_blight | 5 | 12.82 |
| healthy | early_leaf_spot | 5 | 1.42 |
| green_mottle | brown_streak_disease | 5 | 25.00 |
| healthy | mosaic | 5 | 1.42 |

