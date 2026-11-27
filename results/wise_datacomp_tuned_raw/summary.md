# wise_datacomp_tuned_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 59.01 | 83.36 | 73.35 | 93.75 |
| crop | 75.11 | 92.63 | 94.05 | 98.40 |
| disease | 60.67 | 86.11 | 74.80 | 93.90 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 9.48 | 47.02 | 8.29 | 41.09 |
| crop | 72.78 | 87.95 | 79.42 | 92.20 |
| disease | 20.84 | 42.49 | 10.57 | 32.87 |
| composition_unseen_only | 92.37 | 99.64 | 91.23 | 99.45 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 16.34 | 60.12 |
| per_image | 14.89 | 57.14 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 13 | 56.52 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| paddy_healthy | paddy_blast | 10 | 31.25 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_hispa | paddy_leafblast | 8 | 66.67 |
| apple_healthy | apple_marssonina_blotch | 8 | 16.00 |
| cassava_brown_spot | cassava_bacterial_blight | 8 | 42.11 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| paddy_blast | paddy_brown_spot | 6 | 27.27 |
| coffee_rust | coffee_healthy | 6 | 30.00 |
| cassava_healthy | cassava_bacterial_blight | 6 | 21.43 |
| apple_alternaria_leaf_spot | apple_marssonina_blotch | 5 | 62.50 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| cherry | apple | 5 | 35.71 |
| apple | peach | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| plum | apple | 3 | 30.00 |
| soybean | tomato | 2 | 2.78 |
| cabbage | brassica | 2 | 50.00 |
| tobacco | tomato | 2 | 50.00 |
| blackgram | rose | 2 | 13.33 |
| potato | apple | 2 | 3.77 |
| apple | pear | 2 | 0.84 |
| plum | peach | 2 | 20.00 |
| apricot | peach | 2 | 50.00 |
| cucumber | tomato | 2 | 16.67 |
| grape | cotton | 2 | 4.88 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 13 | 56.52 |
| healthy | caterpillar | 12 | 3.40 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| healthy | blast | 10 | 2.83 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| healthy | marssonina_blotch | 8 | 2.27 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| mosaic | healthy | 7 | 5.38 |
| blast | brown_spot | 6 | 27.27 |
| rust | healthy | 6 | 7.14 |
| healthy | bacterial_blight | 6 | 1.70 |
| mosaic | green_mottle | 5 | 3.85 |

