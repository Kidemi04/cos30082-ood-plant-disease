# clip_bioclip_full_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 54.91 | 79.40 | 70.10 | 90.60 |
| crop | 70.58 | 85.54 | 93.10 | 97.05 |
| disease | 54.32 | 78.31 | 71.15 | 92.00 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 9.09 | 59.10 | 6.84 | 53.38 |
| crop | 68.16 | 89.22 | 74.65 | 91.30 |
| disease | 23.42 | 68.62 | 13.47 | 60.98 |
| composition_unseen_only | 74.64 | 99.03 | 76.73 | 97.93 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 15.60 | 67.76 |
| per_image | 12.46 | 67.18 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 24 | 75.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| cassava_mosaic | cassava_healthy | 12 | 9.68 |
| paddy_hispa | paddy_blast | 11 | 91.67 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |
| coffee_rust | coffee_healthy | 7 | 35.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| paddy_healthy | paddy_leafblast | 6 | 18.75 |
| paddy_brown_spot | paddy_blast | 6 | 37.50 |
| cassava_brown_streak_disease | cassava_healthy | 5 | 17.24 |
| apple_healthy | apple_marssonina_blotch | 5 | 10.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| apple | rose | 4 | 1.68 |
| plum | apple | 4 | 40.00 |
| apple | cotton | 4 | 1.68 |
| cherry | apple | 4 | 28.57 |
| apple | mango | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| cherry | peach | 3 | 21.43 |
| paddy | wheat | 3 | 1.90 |
| cabbage | brassica | 2 | 50.00 |
| apple | peach | 2 | 0.84 |
| cotton | cassava | 2 | 10.53 |
| tomato | pepper_bell | 2 | 0.69 |
| potato | apple | 2 | 3.77 |
| lettuce | tomato | 2 | 66.67 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 24 | 6.80 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| mosaic | healthy | 14 | 10.77 |
| healthy | caterpillar | 12 | 3.40 |
| hispa | blast | 11 | 91.67 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| rust | healthy | 9 | 10.71 |
| stripe_rust | rust | 8 | 80.00 |
| mosaic | green_mottle | 7 | 5.38 |
| brown_streak_disease | healthy | 6 | 20.69 |
| marssonina_blotch | downy_mildew | 6 | 42.86 |
| bacterial_blight | healthy | 6 | 19.35 |
| brown_spot | blast | 6 | 8.33 |
| septoria_leaf_spot | verticulium_wilt | 6 | 14.63 |

