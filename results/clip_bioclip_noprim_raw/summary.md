# clip_bioclip_noprim_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 55.79 | 78.76 | 71.25 | 89.60 |
| crop | 70.58 | 83.19 | 92.55 | 96.60 |
| disease | 54.89 | 76.45 | 71.95 | 91.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 6.95 | 49.38 | 6.49 | 45.72 |
| crop | 69.91 | 88.70 | 76.52 | 92.27 |
| disease | 19.93 | 62.46 | 12.36 | 55.25 |
| composition_unseen_only | 71.07 | 98.04 | 69.13 | 96.48 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.36 | 60.70 |
| per_image | 11.90 | 60.54 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 25 | 78.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| soybean_healthy | soybean_caterpillar | 13 | 44.83 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| cassava_mosaic | cassava_healthy | 10 | 8.06 |
| cassava_mosaic | cassava_green_mottle | 9 | 7.26 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| coffee_rust | coffee_healthy | 6 | 30.00 |
| cassava_brown_spot | cassava_bacterial_blight | 6 | 31.58 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 6 | 15.38 |
| paddy_healthy | paddy_leafblast | 5 | 15.62 |
| cassava_brown_streak_disease | cassava_healthy | 5 | 17.24 |
| paddy_brown_spot | paddy_blast | 5 | 31.25 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cotton | 6 | 2.52 |
| cherry | apple | 5 | 35.71 |
| potato | tomato | 4 | 7.55 |
| apple | tomato | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| eggplant | tomato | 3 | 60.00 |
| sunflower | tomato | 2 | 20.00 |
| grape | strawberry | 2 | 4.88 |
| palm | mango | 2 | 15.38 |
| cabbage | brassica | 2 | 50.00 |
| carrot | mango | 2 | 66.67 |
| apple | mango | 2 | 0.84 |
| blueberry | tomato | 2 | 11.76 |
| tomato | pepper_bell | 2 | 0.69 |
| guava | mango | 2 | 15.38 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 24 | 6.80 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| mosaic | healthy | 12 | 9.23 |
| healthy | caterpillar | 11 | 3.12 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| mosaic | green_mottle | 9 | 6.92 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| rust | healthy | 8 | 9.52 |
| stripe_rust | rust | 8 | 80.00 |
| hispa | blast | 7 | 58.33 |
| brown_streak_disease | healthy | 6 | 20.69 |
| brown_spot | bacterial_blight | 6 | 8.33 |
| septoria_leaf_spot | verticulium_wilt | 6 | 14.63 |
| healthy | leafblast | 5 | 1.42 |
| bacterial_blight | healthy | 5 | 16.13 |

