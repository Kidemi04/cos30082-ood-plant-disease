# wise_b2_tuned_s2_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 62.88 | 87.46 | 76.00 | 94.30 |
| crop | 80.38 | 96.23 | 95.80 | 98.75 |
| disease | 64.11 | 88.55 | 77.30 | 94.65 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 6.96 | 52.32 | 4.70 | 45.65 |
| crop | 85.16 | 95.97 | 89.57 | 97.58 |
| disease | 16.55 | 40.84 | 6.98 | 28.80 |
| composition_unseen_only | 93.38 | 99.69 | 93.85 | 99.45 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.53 | 65.47 |
| per_image | 8.85 | 61.52 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| paddy_healthy | paddy_leafblast | 12 | 37.50 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 12 | 44.44 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 7 | 17.95 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |
| tomato_septoria_leaf_spot | tomato_curl | 5 | 12.82 |
| coffee_rust | coffee_healthy | 5 | 25.00 |
| tomato_late_blight | tomato_bacterial_leaf_spot | 5 | 17.24 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 4 | 7.55 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| cabbage | cauliflower | 3 | 75.00 |
| cherry | apple | 3 | 21.43 |
| cotton | squash | 2 | 10.53 |
| cauliflower | tomato | 2 | 33.33 |
| basil | tomato | 2 | 18.18 |
| paddy | palm | 2 | 1.27 |
| apple | rose | 2 | 0.84 |
| zucchini | cucumber | 2 | 50.00 |
| broccoli | cauliflower | 2 | 40.00 |
| apple | cotton | 2 | 0.84 |
| apricot | peach | 2 | 50.00 |
| apple | guava | 2 | 0.84 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 13 | 3.68 |
| healthy | caterpillar | 12 | 3.40 |
| healthy | leafblast | 12 | 3.40 |
| bacterial_spot | septoria_leaf_spot | 12 | 23.08 |
| hispa | blast | 9 | 75.00 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| septoria_leaf_spot | leaf_blight | 7 | 17.07 |
| stripe_rust | rust | 6 | 60.00 |
| healthy | diabrotica_speciosa | 6 | 1.70 |
| septoria_leaf_spot | curl | 5 | 12.20 |
| rust | healthy | 5 | 5.95 |
| healthy | brown_streak_disease | 5 | 1.42 |
| brown_streak_disease | healthy | 5 | 17.24 |

