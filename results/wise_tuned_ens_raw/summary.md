# wise_tuned_ens_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 61.23 | 83.79 | 75.15 | 92.85 |
| crop | 76.52 | 93.27 | 94.40 | 98.35 |
| disease | 61.86 | 85.29 | 76.35 | 93.25 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 5.39 | 47.05 | 4.21 | 45.99 |
| crop | 70.21 | 89.42 | 78.04 | 93.30 |
| disease | 19.71 | 47.53 | 8.84 | 41.16 |
| composition_unseen_only | 91.14 | 99.86 | 90.68 | 99.72 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 9.91 | 60.26 |
| per_image | 7.98 | 61.52 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 18 | 56.25 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| paddy_healthy | paddy_leafblast | 6 | 18.75 |
| apple_healthy | apple_marssonina_blotch | 6 | 12.00 |
| apple_marssonina_blotch | apple_alternaria_leaf_spot | 6 | 42.86 |
| cassava_brown_spot | cassava_bacterial_blight | 6 | 31.58 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |
| cassava_green_mottle | cassava_brown_streak_disease | 5 | 25.00 |
| tomato_late_blight | pepper_bell_blossom_end_rot | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 5 | 2.10 |
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| tomato | pepper_bell | 4 | 1.37 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| paddy | wheat | 3 | 1.90 |
| soybean | blackgram | 2 | 2.78 |
| apple | tomato | 2 | 0.84 |
| cabbage | brassica | 2 | 50.00 |
| apple | cotton | 2 | 0.84 |
| cauliflower | cotton | 2 | 33.33 |
| cherry | peach | 2 | 14.29 |
| brassica | cabbage | 2 | 100.00 |
| eggplant | tomato | 2 | 40.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 18 | 5.10 |
| caterpillar | diabrotica_speciosa | 12 | 52.17 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| healthy | caterpillar | 10 | 2.83 |
| hispa | blast | 9 | 75.00 |
| stripe_rust | rust | 9 | 90.00 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| mosaic | healthy | 8 | 6.15 |
| healthy | leafblast | 6 | 1.70 |
| healthy | marssonina_blotch | 6 | 1.70 |
| brown_spot | bacterial_blight | 6 | 8.33 |
| healthy | diabrotica_speciosa | 6 | 1.70 |
| green_mottle | brown_streak_disease | 5 | 25.00 |
| brown_spot | blast | 5 | 6.94 |
| healthy | brown_streak_disease | 5 | 1.42 |

