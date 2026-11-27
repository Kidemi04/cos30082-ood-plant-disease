# dci_a2_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 56.61 | 83.85 | 71.40 | 92.40 |
| crop | 73.85 | 91.52 | 93.30 | 98.00 |
| disease | 55.33 | 83.16 | 72.15 | 90.85 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 4.15 | 26.21 | 3.80 | 23.00 |
| crop | 68.55 | 85.96 | 76.59 | 90.40 |
| disease | 17.64 | 33.23 | 8.36 | 24.45 |
| composition_unseen_only | 80.75 | 99.23 | 81.70 | 98.90 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 7.74 | 39.94 |
| per_image | 7.21 | 36.83 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 21 | 65.62 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 14 | 35.90 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| paddy_hispa | paddy_leafblast | 6 | 50.00 |
| paddy_hispa | paddy_blast | 6 | 50.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |
| cassava_brown_spot | cassava_bacterial_blight | 5 | 26.32 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |
| apple_alternaria_leaf_spot | apple_marssonina_blotch | 4 | 50.00 |
| cassava_healthy | cassava_brown_streak_disease | 4 | 14.29 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| cherry | apple | 6 | 42.86 |
| apple | rose | 4 | 1.68 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| soybean | bean | 2 | 2.78 |
| cabbage | cauliflower | 2 | 50.00 |
| potato | apple | 2 | 3.77 |
| paddy | wheat | 2 | 1.27 |
| banana | corn | 2 | 6.25 |
| sugarcane | paddy | 2 | 4.55 |
| bean | soybean | 2 | 11.76 |
| peach | raspberry | 1 | 2.50 |
| cucumber | zucchini | 1 | 8.33 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 21 | 5.95 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| septoria_leaf_spot | leaf_blight | 15 | 36.59 |
| healthy | caterpillar | 11 | 3.12 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| healthy | leafblast | 7 | 1.98 |
| marssonina_blotch | downy_mildew | 5 | 35.71 |
| green_mottle | brown_streak_disease | 5 | 25.00 |
| blight_leaf | brown_spot | 5 | 71.43 |
| healthy | diabrotica_speciosa | 5 | 1.42 |
| hispa | leafblast | 4 | 33.33 |
| healthy | brown_streak_disease | 4 | 1.13 |

