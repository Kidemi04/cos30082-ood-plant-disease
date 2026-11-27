# cpn_p3_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 54.54 | 76.71 | 71.10 | 87.80 |
| crop | 70.73 | 86.64 | 92.55 | 96.90 |
| disease | 55.41 | 76.77 | 71.90 | 87.50 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 6.39 | 36.70 | 5.46 | 29.35 |
| crop | 68.71 | 82.51 | 75.14 | 85.64 |
| disease | 19.87 | 37.52 | 10.70 | 26.93 |
| composition_unseen_only | 83.77 | 97.21 | 81.08 | 96.20 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 11.44 | 49.65 |
| per_image | 10.13 | 43.99 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 14 | 35.90 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 11 | 47.83 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| paddy_healthy | paddy_leafblast | 7 | 21.88 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| cassava_mosaic | cassava_green_mottle | 6 | 4.84 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 6 | 15.38 |
| paddy_healthy | paddy_brown_spot | 6 | 18.75 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 6 | 2.52 |
| cherry | apple | 6 | 42.86 |
| potato | tomato | 4 | 7.55 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| wheat | corn | 3 | 7.14 |
| apple | guava | 3 | 1.26 |
| apple | rose | 2 | 0.84 |
| basil | mango | 2 | 18.18 |
| cabbage | brassica | 2 | 50.00 |
| tobacco | tomato | 2 | 50.00 |
| pepper_bell | cassava | 2 | 20.00 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| potato | cotton | 2 | 3.77 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 17 | 4.82 |
| septoria_leaf_spot | leaf_blight | 14 | 34.15 |
| mosaic | healthy | 12 | 9.23 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| hispa | blast | 8 | 66.67 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| stripe_rust | rust | 8 | 80.00 |
| healthy | caterpillar | 7 | 1.98 |
| healthy | leafblast | 7 | 1.98 |
| healthy | brown_streak_disease | 7 | 1.98 |
| brown_spot | bacterial_blight | 7 | 9.72 |
| mosaic | green_mottle | 6 | 4.62 |
| septoria_leaf_spot | verticulium_wilt | 6 | 14.63 |
| healthy | brown_spot | 6 | 1.70 |

