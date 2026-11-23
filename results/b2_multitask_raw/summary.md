# b2_multitask_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 54.74 | 73.32 | 70.15 | 88.20 |
| crop | 67.63 | 81.49 | 90.75 | 95.85 |
| disease | 55.26 | 76.85 | 71.05 | 89.40 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 2.86 | 24.77 | 2.14 | 22.79 |
| crop | 61.19 | 77.71 | 65.68 | 81.56 |
| disease | 18.06 | 35.41 | 10.15 | 26.45 |
| composition_unseen_only | 79.28 | 97.09 | 76.59 | 95.44 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 5.43 | 37.03 |
| per_image | 4.15 | 36.22 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 19 | 59.38 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 15 | 65.22 |
| soybean_healthy | soybean_diabrotica_speciosa | 12 | 41.38 |
| paddy_hispa | paddy_blast | 10 | 83.33 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| coffee_rust | coffee_healthy | 8 | 40.00 |
| tomato_healthy | tomato_leaf_blight | 7 | 21.88 |
| cassava_green_mottle | cassava_mosaic | 6 | 30.00 |
| paddy_healthy | paddy_hispa | 6 | 18.75 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |
| corn_leaf_blight | corn_northern_leaf_blight | 5 | 31.25 |
| peanut_healthy | peanut_early_leaf_spot | 5 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| apple | peach | 5 | 2.10 |
| apple | guava | 5 | 2.10 |
| wheat | paddy | 4 | 9.52 |
| cherry | apple | 4 | 28.57 |
| apple | rose | 3 | 1.26 |
| raspberry | tomato | 3 | 27.27 |
| tomato | chilli | 3 | 1.03 |
| paddy | palm | 3 | 1.90 |
| apple | cassava | 3 | 1.26 |
| apricot | peach | 3 | 75.00 |
| guava | cassava | 3 | 23.08 |
| paddy | corn | 2 | 1.27 |
| tobacco | tomato | 2 | 50.00 |
| apple | cotton | 2 | 0.84 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 19 | 5.38 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| healthy | diabrotica_speciosa | 12 | 3.40 |
| hispa | blast | 10 | 83.33 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| rust | healthy | 10 | 11.90 |
| mosaic | healthy | 9 | 6.92 |
| healthy | leaf_blight | 8 | 2.27 |
| stripe_rust | rust | 7 | 70.00 |
| blast | brown_spot | 6 | 27.27 |
| green_mottle | mosaic | 6 | 30.00 |
| healthy | early_leaf_spot | 6 | 1.70 |
| healthy | brown_streak_disease | 5 | 1.42 |
| leaf_blight | northern_leaf_blight | 5 | 12.82 |

