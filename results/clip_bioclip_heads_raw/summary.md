# clip_bioclip_heads_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 58.63 | 81.19 | 72.85 | 91.75 |
| crop | 74.12 | 91.02 | 93.95 | 97.90 |
| disease | 60.11 | 83.49 | 73.65 | 92.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 2.85 | 33.03 | 3.87 | 34.94 |
| crop | 68.49 | 88.30 | 75.97 | 91.71 |
| disease | 17.81 | 36.59 | 9.25 | 32.53 |
| composition_unseen_only | 84.65 | 99.38 | 85.43 | 99.10 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 5.43 | 46.95 |
| per_image | 7.34 | 50.61 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| cassava_green_mottle | cassava_brown_streak_disease | 5 | 25.00 |
| apple_healthy | apple_marssonina_blotch | 5 | 10.00 |
| cassava_healthy | cassava_bacterial_blight | 5 | 17.86 |
| cassava_brown_streak_disease | cassava_healthy | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| cherry | apple | 6 | 42.86 |
| apple | peach | 4 | 1.68 |
| potato | tomato | 4 | 7.55 |
| eggplant | tomato | 3 | 60.00 |
| plum | apple | 3 | 30.00 |
| bean | soybean | 3 | 17.65 |
| soybean | blackgram | 2 | 2.78 |
| plum | pear | 2 | 20.00 |
| cabbage | cauliflower | 2 | 50.00 |
| basil | tomato | 2 | 18.18 |
| raspberry | tomato | 2 | 18.18 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| paddy | wheat | 2 | 1.27 |
| banana | corn | 2 | 6.25 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | leafblast | 11 | 3.12 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| mosaic | healthy | 9 | 6.92 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| healthy | caterpillar | 8 | 2.27 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| green_mottle | brown_streak_disease | 5 | 25.00 |
| healthy | marssonina_blotch | 5 | 1.42 |
| healthy | bacterial_blight | 5 | 1.42 |
| brown_streak_disease | healthy | 5 | 17.24 |

