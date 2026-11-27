# ccd_cpn_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 58.52 | 81.13 | 71.05 | 90.05 |
| crop | 74.04 | 92.55 | 93.85 | 98.20 |
| disease | 57.39 | 82.14 | 71.15 | 89.25 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 5.70 | 35.73 | 6.22 | 30.87 |
| crop | 70.76 | 87.00 | 78.31 | 90.68 |
| disease | 15.75 | 32.81 | 7.39 | 21.55 |
| composition_unseen_only | 83.75 | 99.33 | 84.94 | 98.83 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 10.39 | 49.61 |
| per_image | 11.43 | 45.98 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 17 | 53.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| cassava_mosaic | cassava_green_mottle | 10 | 8.06 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| paddy_healthy | paddy_blast | 9 | 28.12 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |
| cassava_brown_spot | cassava_bacterial_blight | 8 | 42.11 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| soybean_healthy | soybean_caterpillar | 6 | 20.69 |
| tomato_septoria_leaf_spot | tomato_curl | 6 | 15.38 |
| peanut_healthy | peanut_early_leaf_spot | 5 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | guava | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| cotton | squash | 2 | 10.53 |
| cabbage | cauliflower | 2 | 50.00 |
| apple | peach | 2 | 0.84 |
| tomato | pepper_bell | 2 | 0.69 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| paddy | wheat | 2 | 1.27 |
| banana | corn | 2 | 6.25 |
| bean | soybean | 2 | 11.76 |
| sugarcane | paddy | 2 | 4.55 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| mosaic | green_mottle | 10 | 7.69 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| hispa | leafblast | 9 | 75.00 |
| healthy | blast | 9 | 2.55 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| healthy | brown_streak_disease | 7 | 1.98 |
| healthy | caterpillar | 6 | 1.70 |
| septoria_leaf_spot | curl | 6 | 14.63 |
| healthy | bacterial_blight | 6 | 1.70 |
| stripe_rust | rust | 6 | 60.00 |

