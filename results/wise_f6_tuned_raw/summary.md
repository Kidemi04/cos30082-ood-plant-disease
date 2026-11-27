# wise_f6_tuned_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 59.66 | 81.57 | 74.30 | 92.10 |
| crop | 76.52 | 93.34 | 93.75 | 98.20 |
| disease | 59.93 | 84.57 | 75.60 | 92.30 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 5.13 | 49.44 | 4.49 | 50.69 |
| crop | 68.76 | 89.27 | 77.35 | 92.89 |
| disease | 19.30 | 48.90 | 8.56 | 43.58 |
| composition_unseen_only | 91.16 | 99.86 | 90.81 | 99.72 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 9.45 | 61.56 |
| per_image | 8.47 | 65.39 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 15 | 46.88 |
| soybean_healthy | soybean_caterpillar | 14 | 48.28 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 11 | 47.83 |
| paddy_healthy | paddy_leafblast | 9 | 28.12 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_hispa | paddy_leafblast | 8 | 66.67 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| apple_healthy | apple_marssonina_blotch | 6 | 12.00 |
| cassava_brown_spot | cassava_bacterial_blight | 6 | 31.58 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| tomato_late_blight | pepper_bell_blossom_end_rot | 5 | 17.24 |
| coffee_rust | coffee_spider_mites | 5 | 25.00 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 5 | 2.10 |
| potato | tomato | 5 | 9.43 |
| tomato | pepper_bell | 4 | 1.37 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| cherry | peach | 3 | 21.43 |
| cherry | apple | 3 | 21.43 |
| paddy | wheat | 3 | 1.90 |
| soybean | blackgram | 2 | 2.78 |
| cotton | squash | 2 | 10.53 |
| apple | tomato | 2 | 0.84 |
| grape | strawberry | 2 | 4.88 |
| cabbage | brassica | 2 | 50.00 |
| apple | cotton | 2 | 0.84 |
| potato | apple | 2 | 3.77 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 15 | 4.25 |
| healthy | caterpillar | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| healthy | leafblast | 9 | 2.55 |
| mosaic | healthy | 9 | 6.92 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| healthy | marssonina_blotch | 6 | 1.70 |
| brown_spot | bacterial_blight | 6 | 8.33 |
| healthy | brown_streak_disease | 6 | 1.70 |
| rust | spider_mites | 5 | 5.95 |
| blight_leaf | brown_spot | 5 | 71.43 |
| green_mottle | green_mite | 4 | 20.00 |

