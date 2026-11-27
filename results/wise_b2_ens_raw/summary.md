# wise_b2_ens_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 63.08 | 88.23 | 76.00 | 94.45 |
| crop | 81.23 | 97.27 | 95.70 | 99.15 |
| disease | 64.49 | 89.83 | 77.05 | 94.90 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 10.20 | 56.24 | 6.84 | 50.69 |
| crop | 86.19 | 97.00 | 90.26 | 98.14 |
| disease | 19.40 | 45.57 | 8.63 | 36.74 |
| composition_unseen_only | 94.17 | 99.76 | 95.10 | 99.59 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 17.57 | 68.69 |
| per_image | 12.55 | 65.97 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 15 | 55.56 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| soybean_healthy | soybean_caterpillar | 13 | 44.83 |
| paddy_healthy | paddy_blast | 12 | 37.50 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 7 | 17.95 |
| paddy_hispa | paddy_blast | 7 | 58.33 |
| tomato_late_blight | tomato_bacterial_leaf_spot | 5 | 17.24 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 5 | 12.82 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |
| cassava_brown_streak_disease | cassava_healthy | 5 | 17.24 |
| cassava_mosaic | cassava_healthy | 5 | 4.03 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |
| soybean_healthy | soybean_diabrotica_speciosa | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 4 | 7.55 |
| apple | peach | 3 | 1.26 |
| cabbage | cauliflower | 3 | 75.00 |
| cotton | squash | 2 | 10.53 |
| apple | tomato | 2 | 0.84 |
| cauliflower | tomato | 2 | 33.33 |
| paddy | palm | 2 | 1.27 |
| cherry | apple | 2 | 14.29 |
| apple | rose | 2 | 0.84 |
| broccoli | cabbage | 2 | 40.00 |
| zucchini | cucumber | 2 | 50.00 |
| broccoli | cauliflower | 2 | 40.00 |
| apple | cotton | 2 | 0.84 |
| apricot | peach | 2 | 50.00 |
| broccoli | cucumber | 1 | 20.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| bacterial_spot | septoria_leaf_spot | 15 | 28.85 |
| healthy | leafblast | 14 | 3.97 |
| healthy | caterpillar | 13 | 3.68 |
| healthy | blast | 12 | 3.40 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| septoria_leaf_spot | leaf_blight | 7 | 17.07 |
| hispa | blast | 7 | 58.33 |
| marssonina_blotch | alternaria_leaf_spot | 5 | 35.71 |
| septoria_leaf_spot | verticulium_wilt | 5 | 12.20 |
| healthy | brown_streak_disease | 5 | 1.42 |
| brown_streak_disease | healthy | 5 | 17.24 |
| mosaic | healthy | 5 | 3.85 |
| blight_leaf | brown_spot | 5 | 71.43 |
| healthy | diabrotica_speciosa | 5 | 1.42 |

