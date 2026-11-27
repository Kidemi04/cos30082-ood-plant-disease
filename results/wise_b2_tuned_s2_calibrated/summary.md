# wise_b2_tuned_s2_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 56.55 | 83.48 | 72.45 | 92.55 |
| crop | 73.56 | 91.50 | 93.75 | 98.25 |
| disease | 57.82 | 87.97 | 75.25 | 94.45 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 71.44 | 99.01 | 60.50 | 99.10 |
| crop | 94.92 | 99.57 | 94.96 | 99.31 |
| disease | 70.97 | 99.00 | 58.84 | 99.03 |
| composition_unseen_only | 93.36 | 99.69 | 93.78 | 99.45 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 63.13 | 90.59 |
| per_image | 65.94 | 95.71 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| paddy_healthy | paddy_leafblast | 12 | 37.50 |
| tomato_septoria_leaf_spot | tomato_early_blight | 11 | 28.21 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| tomato_bacterial_spot | tomato_early_blight | 9 | 33.33 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| potato_early_blight | tomato_early_blight | 7 | 29.17 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 6 | 15.38 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |
| coffee_rust | coffee_healthy | 5 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 7 | 13.21 |
| cotton | cucumber | 4 | 21.05 |
| paddy | corn | 3 | 1.90 |
| apple | tomato | 3 | 1.26 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| wheat | sugarcane | 3 | 7.14 |
| cherry | apple | 3 | 21.43 |
| apple | cucumber | 3 | 1.26 |
| zucchini | cucumber | 3 | 75.00 |
| corn | sugarcane | 2 | 0.97 |
| blueberry | sugarcane | 2 | 11.76 |
| cauliflower | tomato | 2 | 33.33 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 13 | 3.68 |
| healthy | caterpillar | 12 | 3.40 |
| healthy | leafblast | 12 | 3.40 |
| septoria_leaf_spot | early_blight | 10 | 24.39 |
| hispa | blast | 9 | 75.00 |
| bacterial_spot | early_blight | 9 | 17.31 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| septoria_leaf_spot | leaf_blight | 7 | 17.07 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| bacterial_spot | septoria_leaf_spot | 6 | 11.54 |
| healthy | brown_spot | 6 | 1.70 |
| stripe_rust | rust | 6 | 60.00 |
| healthy | diabrotica_speciosa | 6 | 1.70 |
| rust | healthy | 5 | 5.95 |

