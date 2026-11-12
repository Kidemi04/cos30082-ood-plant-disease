# b1_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 51.27 | 74.82 | 68.30 | 89.20 |
| crop | 67.06 | 87.82 | 90.80 | 96.85 |
| disease | 50.35 | 75.91 | 68.90 | 87.85 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 12.65 | 38.97 | 10.29 | 39.30 |
| crop | 67.00 | 80.65 | 72.38 | 84.39 |
| disease | 19.00 | 36.10 | 11.05 | 28.25 |
| composition_unseen_only | 79.11 | 97.16 | 78.31 | 95.10 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 20.29 | 51.25 |
| per_image | 17.89 | 54.56 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_blast | 17 | 53.12 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| soybean_healthy | soybean_diabrotica_speciosa | 11 | 37.93 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| tomato_healthy | tomato_septoria_leaf_spot | 7 | 21.88 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| paddy_blast | paddy_tungro | 6 | 27.27 |
| cassava_green_mottle | cassava_mosaic | 5 | 25.00 |
| corn_leaf_blight | corn_northern_leaf_blight | 5 | 31.25 |
| palm_healthy | palm_white_scale | 5 | 71.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 8 | 3.36 |
| potato | tomato | 7 | 13.21 |
| cherry | apple | 6 | 42.86 |
| paddy | palm | 5 | 3.16 |
| basil | tomato | 4 | 36.36 |
| apple | tomato | 4 | 1.68 |
| paddy | wheat | 4 | 2.53 |
| apple | guava | 3 | 1.26 |
| tomato | chilli | 3 | 1.03 |
| tomato | peach | 3 | 1.03 |
| peach | raspberry | 2 | 5.00 |
| paddy | corn | 2 | 1.27 |
| carrot | celery | 2 | 66.67 |
| apple | cotton | 2 | 0.84 |
| wheat | paddy | 2 | 4.76 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 17 | 4.82 |
| healthy | diabrotica_speciosa | 11 | 3.12 |
| mosaic | healthy | 11 | 8.46 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| hispa | blast | 9 | 75.00 |
| healthy | leafblast | 8 | 2.27 |
| bacterial_spot | septoria_leaf_spot | 7 | 13.46 |
| healthy | septoria_leaf_spot | 7 | 1.98 |
| stripe_rust | rust | 7 | 70.00 |
| marssonina_blotch | downy_mildew | 6 | 42.86 |
| leaf_blight | northern_leaf_blight | 6 | 15.38 |
| healthy | early_leaf_spot | 6 | 1.70 |
| blast | tungro | 6 | 27.27 |
| rust | healthy | 6 | 7.14 |

