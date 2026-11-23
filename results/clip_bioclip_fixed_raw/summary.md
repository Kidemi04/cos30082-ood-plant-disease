# clip_bioclip_fixed_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 52.41 | 76.21 | 69.65 | 90.05 |
| crop | 67.38 | 81.00 | 92.10 | 96.30 |
| disease | 50.72 | 79.17 | 70.70 | 92.40 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 3.64 | 39.70 | 3.52 | 38.81 |
| crop | 59.56 | 81.40 | 67.89 | 85.84 |
| disease | 19.05 | 59.20 | 10.29 | 53.80 |
| composition_unseen_only | 64.04 | 93.34 | 68.30 | 95.23 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 6.80 | 52.20 |
| per_image | 6.71 | 54.24 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 20 | 62.50 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| soybean_healthy | soybean_caterpillar | 14 | 48.28 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| paddy_healthy | paddy_leafblast | 7 | 21.88 |
| apple_healthy | apple_marssonina_blotch | 7 | 14.00 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| coffee_rust | coffee_healthy | 6 | 30.00 |
| coffee_rust | coffee_spider_mites | 6 | 30.00 |
| tomato_blight_leaf | tomato_brown_spot | 6 | 85.71 |
| cassava_green_mottle | cassava_mosaic | 5 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| plum | apple | 5 | 50.00 |
| basil | tomato | 5 | 45.45 |
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | rose | 3 | 1.26 |
| tomato | pepper_bell | 3 | 1.03 |
| guava | mango | 3 | 23.08 |
| lettuce | tomato | 3 | 100.00 |
| apple | cotton | 3 | 1.26 |
| cotton | squash | 2 | 10.53 |
| palm | paddy | 2 | 15.38 |
| palm | mango | 2 | 15.38 |
| cabbage | tomato | 2 | 50.00 |
| carrot | tomato | 2 | 66.67 |
| apple | mango | 2 | 0.84 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 18 | 5.10 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | caterpillar | 12 | 3.40 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| mosaic | healthy | 11 | 8.46 |
| stripe_rust | rust | 10 | 100.00 |
| hispa | blast | 8 | 66.67 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| rust | healthy | 8 | 9.52 |
| healthy | leafblast | 7 | 1.98 |
| brown_spot | bacterial_blight | 7 | 9.72 |
| rust | spider_mites | 6 | 7.14 |
| green_mottle | mosaic | 5 | 25.00 |
| bacterial_blight | healthy | 5 | 16.13 |
| marssonina_blotch | downy_mildew | 5 | 35.71 |

