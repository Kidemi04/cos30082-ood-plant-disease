# wise_f6_tuned_s2_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 58.28 | 82.30 | 74.10 | 91.80 |
| crop | 74.23 | 93.21 | 93.90 | 98.05 |
| disease | 59.80 | 84.35 | 75.35 | 92.90 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 4.98 | 39.53 | 3.73 | 40.26 |
| crop | 68.69 | 89.85 | 76.38 | 92.96 |
| disease | 19.44 | 44.56 | 9.12 | 37.85 |
| composition_unseen_only | 89.06 | 99.90 | 89.23 | 99.79 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 9.18 | 53.41 |
| per_image | 7.10 | 55.97 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 23 | 71.88 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 13 | 56.52 |
| paddy_hispa | paddy_blast | 11 | 91.67 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| apple_healthy | apple_marssonina_blotch | 5 | 10.00 |
| cassava_green_mottle | cassava_brown_streak_disease | 5 | 25.00 |
| paddy_brown_spot | paddy_blast | 5 | 31.25 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| basil | tomato | 5 | 45.45 |
| potato | tomato | 5 | 9.43 |
| apple | peach | 4 | 1.68 |
| tomato | pepper_bell | 4 | 1.37 |
| cherry | apple | 4 | 28.57 |
| apple | tomato | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| cherry | peach | 3 | 21.43 |
| paddy | corn | 2 | 1.27 |
| cabbage | brassica | 2 | 50.00 |
| lettuce | tomato | 2 | 66.67 |
| brassica | cabbage | 2 | 100.00 |
| cucumber | tomato | 2 | 16.67 |
| eggplant | tomato | 2 | 40.00 |
| paddy | palm | 2 | 1.27 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 23 | 6.52 |
| caterpillar | diabrotica_speciosa | 13 | 56.52 |
| hispa | blast | 11 | 91.67 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| healthy | caterpillar | 9 | 2.55 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| stripe_rust | rust | 8 | 80.00 |
| brown_spot | bacterial_blight | 7 | 9.72 |
| mosaic | healthy | 7 | 5.38 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| healthy | brown_streak_disease | 6 | 1.70 |
| healthy | marssonina_blotch | 5 | 1.42 |
| green_mottle | brown_streak_disease | 5 | 25.00 |
| brown_spot | blast | 5 | 6.94 |
| blight_leaf | brown_spot | 5 | 71.43 |

