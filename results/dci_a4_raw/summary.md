# dci_a4_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 57.28 | 82.20 | 71.40 | 91.30 |
| crop | 72.55 | 90.15 | 93.10 | 97.70 |
| disease | 55.50 | 82.67 | 71.45 | 90.70 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 8.78 | 26.32 | 7.73 | 23.96 |
| crop | 68.92 | 85.69 | 76.93 | 90.33 |
| disease | 17.68 | 31.48 | 8.36 | 22.51 |
| composition_unseen_only | 82.11 | 98.69 | 81.15 | 98.27 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 15.22 | 39.87 |
| per_image | 13.96 | 37.96 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 26 | 81.25 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 13 | 33.33 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 11 | 40.74 |
| cassava_brown_spot | cassava_bacterial_blight | 10 | 52.63 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| paddy_hispa | paddy_blast | 7 | 58.33 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| paddy_hispa | paddy_leafblast | 5 | 41.67 |
| cassava_green_mottle | cassava_brown_streak_disease | 5 | 25.00 |
| coffee_rust | coffee_healthy | 5 | 25.00 |
| paddy_brown_spot | paddy_blast | 5 | 31.25 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| cherry | apple | 5 | 35.71 |
| apple | rose | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| apple | potato | 3 | 1.26 |
| soybean | bean | 2 | 2.78 |
| cabbage | cauliflower | 2 | 50.00 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| banana | corn | 2 | 6.25 |
| sugarcane | paddy | 2 | 4.55 |
| bean | soybean | 2 | 11.76 |
| cucumber | zucchini | 1 | 8.33 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 26 | 7.37 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| septoria_leaf_spot | leaf_blight | 15 | 36.59 |
| bacterial_spot | septoria_leaf_spot | 11 | 21.15 |
| septoria_leaf_spot | verticulium_wilt | 10 | 24.39 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| healthy | caterpillar | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| healthy | brown_streak_disease | 6 | 1.70 |
| mosaic | healthy | 6 | 4.62 |
| brown_spot | blast | 5 | 6.94 |
| leaf_blight | northern_leaf_blight | 5 | 12.82 |
| green_mottle | brown_streak_disease | 5 | 25.00 |

