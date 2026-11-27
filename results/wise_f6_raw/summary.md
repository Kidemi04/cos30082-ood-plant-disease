# wise_f6_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 59.34 | 80.46 | 72.55 | 91.45 |
| crop | 72.98 | 90.66 | 93.10 | 97.55 |
| disease | 58.19 | 81.17 | 73.30 | 91.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 6.77 | 40.42 | 6.77 | 38.81 |
| crop | 66.35 | 86.80 | 74.59 | 91.37 |
| disease | 18.86 | 38.87 | 10.01 | 30.39 |
| composition_unseen_only | 87.55 | 99.55 | 85.36 | 99.31 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.15 | 53.81 |
| per_image | 12.38 | 54.50 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| soybean_healthy | soybean_caterpillar | 11 | 37.93 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| wheat_stripe_rust | wheat_rust | 10 | 100.00 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| apple_healthy | apple_marssonina_blotch | 8 | 16.00 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| paddy_healthy | paddy_brown_spot | 6 | 18.75 |
| tomato_late_blight | pepper_bell_blossom_end_rot | 5 | 17.24 |
| coffee_rust | coffee_spider_mites | 5 | 25.00 |
| cassava_brown_spot | cassava_bacterial_blight | 5 | 26.32 |
| cassava_mosaic | cassava_brown_leaf_spot | 5 | 4.03 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato | pepper_bell | 5 | 1.72 |
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | peach | 4 | 1.68 |
| paddy | wheat | 4 | 2.53 |
| soybean | blackgram | 3 | 4.17 |
| apple | broccoli | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| apple | cotton | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| basil | mango | 2 | 18.18 |
| cabbage | brassica | 2 | 50.00 |
| carrot | tomato | 2 | 66.67 |
| cauliflower | cotton | 2 | 33.33 |
| guava | cassava | 2 | 15.38 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | blast | 16 | 4.53 |
| healthy | caterpillar | 11 | 3.12 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| stripe_rust | rust | 10 | 100.00 |
| healthy | leafblast | 8 | 2.27 |
| hispa | blast | 8 | 66.67 |
| healthy | marssonina_blotch | 8 | 2.27 |
| healthy | brown_streak_disease | 6 | 1.70 |
| healthy | brown_spot | 6 | 1.70 |
| rust | healthy | 5 | 5.95 |
| rust | spider_mites | 5 | 5.95 |
| brown_spot | bacterial_blight | 5 | 6.94 |
| mosaic | brown_leaf_spot | 5 | 3.85 |

