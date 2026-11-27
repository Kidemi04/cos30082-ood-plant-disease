# wise_b2_tuned_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 63.44 | 86.83 | 76.05 | 94.10 |
| crop | 80.68 | 95.96 | 95.75 | 99.00 |
| disease | 63.65 | 88.17 | 77.10 | 94.45 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 8.71 | 50.38 | 5.73 | 46.20 |
| crop | 85.22 | 95.07 | 89.43 | 97.24 |
| disease | 18.18 | 42.79 | 8.56 | 34.67 |
| composition_unseen_only | 92.24 | 99.73 | 92.89 | 99.52 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 15.32 | 63.76 |
| per_image | 10.66 | 61.97 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 12 | 44.44 |
| paddy_healthy | paddy_leafblast | 9 | 28.12 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| apple_marssonina_blotch | apple_alternaria_leaf_spot | 6 | 42.86 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| tomato_late_blight | tomato_bacterial_leaf_spot | 5 | 17.24 |
| cassava_brown_streak_disease | cassava_healthy | 5 | 17.24 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |
| soybean_healthy | soybean_diabrotica_speciosa | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| apple | peach | 3 | 1.26 |
| cabbage | cauliflower | 3 | 75.00 |
| apple | tomato | 2 | 0.84 |
| basil | tomato | 2 | 18.18 |
| apple | rose | 2 | 0.84 |
| zucchini | cucumber | 2 | 50.00 |
| broccoli | cauliflower | 2 | 40.00 |
| apricot | peach | 2 | 50.00 |
| broccoli | cucumber | 1 | 20.00 |
| peach | raspberry | 1 | 2.50 |
| cucumber | zucchini | 1 | 8.33 |
| soybean | cotton | 1 | 1.39 |
| palm | paddy | 1 | 7.69 |
| corn | sugarcane | 1 | 0.49 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 16 | 4.53 |
| healthy | caterpillar | 12 | 3.40 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| bacterial_spot | septoria_leaf_spot | 12 | 23.08 |
| healthy | leafblast | 9 | 2.55 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| mosaic | healthy | 9 | 6.92 |
| hispa | blast | 8 | 66.67 |
| stripe_rust | rust | 6 | 60.00 |
| marssonina_blotch | alternaria_leaf_spot | 5 | 35.71 |
| brown_streak_disease | healthy | 5 | 17.24 |
| blight_leaf | brown_spot | 5 | 71.43 |
| healthy | diabrotica_speciosa | 5 | 1.42 |
| green_mottle | brown_streak_disease | 4 | 20.00 |

