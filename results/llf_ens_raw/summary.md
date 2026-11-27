# llf_ens_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 58.85 | 83.49 | 73.85 | 92.55 |
| crop | 73.91 | 92.54 | 93.90 | 98.20 |
| disease | 58.87 | 86.09 | 74.85 | 92.60 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 8.55 | 42.76 | 8.22 | 40.68 |
| crop | 72.28 | 90.44 | 78.94 | 92.89 |
| disease | 21.30 | 41.71 | 11.26 | 32.60 |
| composition_unseen_only | 88.05 | 99.62 | 88.05 | 99.31 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 14.93 | 56.56 |
| per_image | 14.79 | 56.51 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 15 | 46.88 |
| paddy_healthy | paddy_blast | 11 | 34.38 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 10 | 43.48 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |
| paddy_hispa | paddy_blast | 6 | 50.00 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |
| paddy_hispa | paddy_leafblast | 5 | 41.67 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |
| soybean_healthy | soybean_diabrotica_speciosa | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| cherry | apple | 4 | 28.57 |
| apple | guava | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| cabbage | brassica | 2 | 50.00 |
| basil | tomato | 2 | 18.18 |
| cherry | peach | 2 | 14.29 |
| potato | apple | 2 | 3.77 |
| apple | tomato | 2 | 0.84 |
| cassava | corn | 2 | 0.74 |
| lettuce | tomato | 2 | 66.67 |
| banana | corn | 2 | 6.25 |
| broccoli | tomato | 2 | 40.00 |
| bean | soybean | 2 | 11.76 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 15 | 4.25 |
| healthy | blast | 11 | 3.12 |
| healthy | caterpillar | 10 | 2.83 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| caterpillar | diabrotica_speciosa | 10 | 43.48 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| mosaic | green_mottle | 7 | 5.38 |
| mosaic | healthy | 7 | 5.38 |
| hispa | blast | 6 | 50.00 |
| healthy | brown_streak_disease | 5 | 1.42 |
| marssonina_blotch | downy_mildew | 5 | 35.71 |
| brown_spot | bacterial_blight | 5 | 6.94 |
| hispa | leafblast | 5 | 41.67 |

