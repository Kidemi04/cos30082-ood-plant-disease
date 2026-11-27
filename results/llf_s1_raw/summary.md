# llf_s1_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 57.70 | 81.62 | 72.90 | 92.00 |
| crop | 74.51 | 92.10 | 93.80 | 98.05 |
| disease | 56.28 | 83.78 | 73.30 | 91.60 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.98 | 39.85 | 8.01 | 43.02 |
| crop | 71.62 | 88.95 | 78.18 | 91.64 |
| disease | 17.83 | 38.89 | 9.46 | 32.67 |
| composition_unseen_only | 82.50 | 99.37 | 84.81 | 98.90 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 14.02 | 53.55 |
| per_image | 14.44 | 58.63 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 14 | 43.75 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 9 | 39.13 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| cassava_mosaic | cassava_green_mottle | 8 | 6.45 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| tomato_target_spot | tomato_healthy | 6 | 40.00 |
| paddy_blast | paddy_brown_spot | 5 | 22.73 |
| tomato_leaf_blight | tomato_verticulium_wilt | 5 | 50.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 5 | 2.10 |
| cherry | apple | 5 | 35.71 |
| potato | tomato | 4 | 7.55 |
| plum | apple | 3 | 30.00 |
| apple | cotton | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| banana | corn | 3 | 9.38 |
| plum | pear | 2 | 20.00 |
| cabbage | brassica | 2 | 50.00 |
| wheat | corn | 2 | 4.76 |
| cherry | peach | 2 | 14.29 |
| cassava | corn | 2 | 0.74 |
| broccoli | tomato | 2 | 40.00 |
| bean | soybean | 2 | 11.76 |
| cotton | cassava | 2 | 10.53 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 14 | 3.97 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| mosaic | healthy | 9 | 6.92 |
| caterpillar | diabrotica_speciosa | 9 | 39.13 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| healthy | caterpillar | 8 | 2.27 |
| healthy | leafblast | 8 | 2.27 |
| mosaic | green_mottle | 8 | 6.15 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| marssonina_blotch | downy_mildew | 6 | 42.86 |
| target_spot | healthy | 6 | 28.57 |
| blast | brown_spot | 5 | 22.73 |

