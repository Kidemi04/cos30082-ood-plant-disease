# cpn_p0_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 57.62 | 83.28 | 72.55 | 91.90 |
| crop | 74.04 | 92.55 | 93.85 | 98.20 |
| disease | 59.25 | 85.19 | 73.75 | 91.45 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.93 | 31.34 | 6.98 | 29.49 |
| crop | 70.76 | 87.00 | 78.31 | 90.68 |
| disease | 19.09 | 35.41 | 9.46 | 25.69 |
| composition_unseen_only | 84.25 | 99.20 | 85.43 | 98.83 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 13.95 | 45.54 |
| per_image | 12.73 | 44.65 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 16 | 50.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 12 | 30.77 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| paddy_hispa | paddy_blast | 6 | 50.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |
| paddy_blast | paddy_brown_spot | 5 | 22.73 |
| apple_healthy | apple_marssonina_blotch | 5 | 10.00 |
| paddy_hispa | paddy_leafblast | 5 | 41.67 |

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
| healthy | blast | 16 | 4.53 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| septoria_leaf_spot | verticulium_wilt | 12 | 29.27 |
| healthy | caterpillar | 9 | 2.55 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | leafblast | 8 | 2.27 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| mosaic | green_mottle | 7 | 5.38 |
| mosaic | healthy | 7 | 5.38 |
| hispa | blast | 6 | 50.00 |
| healthy | diabrotica_speciosa | 6 | 1.70 |
| blast | brown_spot | 5 | 22.73 |
| marssonina_blotch | downy_mildew | 5 | 35.71 |
| healthy | marssonina_blotch | 5 | 1.42 |

