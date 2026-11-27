# wise_f6_tuned_s1_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 60.00 | 83.11 | 74.05 | 92.35 |
| crop | 74.44 | 92.21 | 94.10 | 98.15 |
| disease | 62.83 | 85.02 | 75.60 | 93.00 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 5.04 | 40.98 | 4.63 | 40.47 |
| crop | 68.60 | 87.27 | 76.31 | 91.92 |
| disease | 19.55 | 43.58 | 9.46 | 35.84 |
| composition_unseen_only | 88.76 | 99.79 | 88.74 | 99.65 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 9.30 | 54.89 |
| per_image | 8.71 | 56.28 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_blast | 18 | 56.25 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 11 | 40.74 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| cassava_brown_spot | cassava_bacterial_blight | 8 | 42.11 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| paddy_healthy | paddy_leafblast | 6 | 18.75 |
| apple_marssonina_blotch | apple_alternaria_leaf_spot | 6 | 42.86 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |
| paddy_healthy | paddy_brown_spot | 5 | 15.62 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | peach | 4 | 1.68 |
| apple | tomato | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| tomato | pepper_bell | 3 | 1.03 |
| paddy | wheat | 3 | 1.90 |
| cabbage | brassica | 2 | 50.00 |
| tobacco | tomato | 2 | 50.00 |
| cauliflower | cotton | 2 | 33.33 |
| wheat | corn | 2 | 4.76 |
| guava | cassava | 2 | 15.38 |
| cherry | peach | 2 | 14.29 |
| plum | peach | 2 | 20.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 18 | 5.10 |
| bacterial_spot | septoria_leaf_spot | 11 | 21.15 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| hispa | blast | 9 | 75.00 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| stripe_rust | rust | 9 | 90.00 |
| mosaic | healthy | 8 | 6.15 |
| healthy | caterpillar | 7 | 1.98 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| healthy | leafblast | 6 | 1.70 |
| healthy | brown_streak_disease | 5 | 1.42 |
| marssonina_blotch | alternaria_leaf_spot | 5 | 35.71 |
| late_blight | blossom_end_rot | 5 | 10.00 |

