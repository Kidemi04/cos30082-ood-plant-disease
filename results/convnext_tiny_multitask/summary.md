# convnext_tiny_multitask

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 52.71 | 74.35 | 68.90 | 88.75 |
| crop | 66.42 | 84.28 | 90.75 | 96.10 |
| disease | 54.34 | 76.56 | 70.45 | 89.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 3.40 | 27.32 | 2.28 | 27.21 |
| crop | 62.53 | 77.40 | 66.16 | 81.42 |
| disease | 18.25 | 33.07 | 10.08 | 25.76 |
| composition_unseen_only | 80.44 | 97.34 | 76.04 | 95.99 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 6.40 | 39.96 |
| per_image | 4.41 | 41.65 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 15 | 46.88 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 10 | 43.48 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| cassava_mosaic | cassava_green_mottle | 8 | 6.45 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| cassava_healthy | cassava_bacterial_blight | 6 | 21.43 |
| paddy_healthy | paddy_brown_spot | 6 | 18.75 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| paddy_blast | paddy_brown_spot | 6 | 27.27 |
| paddy_healthy | paddy_leafblast | 5 | 15.62 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 8 | 3.36 |
| cherry | apple | 6 | 42.86 |
| potato | tomato | 5 | 9.43 |
| apple | guava | 4 | 1.68 |
| raspberry | tomato | 3 | 27.27 |
| basil | tomato | 3 | 27.27 |
| paddy | palm | 3 | 1.90 |
| tomato | apple | 3 | 1.03 |
| apple | cassava | 3 | 1.26 |
| guava | cassava | 3 | 23.08 |
| paddy | corn | 2 | 1.27 |
| apple | rose | 2 | 0.84 |
| tobacco | tomato | 2 | 50.00 |
| apple | cotton | 2 | 0.84 |
| wheat | paddy | 2 | 4.76 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 15 | 4.25 |
| healthy | caterpillar | 10 | 2.83 |
| caterpillar | diabrotica_speciosa | 10 | 43.48 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| mosaic | healthy | 9 | 6.92 |
| hispa | blast | 8 | 66.67 |
| mosaic | green_mottle | 8 | 6.15 |
| stripe_rust | rust | 8 | 80.00 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| blast | brown_spot | 7 | 31.82 |
| healthy | leaf_blight | 6 | 1.70 |
| rust | healthy | 6 | 7.14 |
| healthy | bacterial_blight | 6 | 1.70 |
| healthy | brown_spot | 6 | 1.70 |
