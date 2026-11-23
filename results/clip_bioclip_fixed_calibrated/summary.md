# clip_bioclip_fixed_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 43.88 | 71.73 | 64.60 | 86.95 |
| crop | 61.21 | 79.23 | 89.00 | 95.60 |
| disease | 44.64 | 72.31 | 68.20 | 90.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 47.39 | 84.65 | 46.27 | 87.64 |
| crop | 67.72 | 88.79 | 74.38 | 93.23 |
| disease | 49.20 | 87.94 | 46.13 | 89.16 |
| composition_unseen_only | 64.04 | 93.34 | 68.30 | 95.23 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 45.56 | 77.66 |
| per_image | 53.92 | 87.29 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 18 | 56.25 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 15 | 65.22 |
| soybean_healthy | soybean_caterpillar | 13 | 44.83 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| paddy_healthy | paddy_leafblast | 7 | 21.88 |
| apple_healthy | apple_marssonina_blotch | 7 | 14.00 |
| cassava_brown_spot | cassava_bacterial_blight | 7 | 36.84 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| coffee_rust | coffee_healthy | 6 | 30.00 |
| coffee_rust | coffee_spider_mites | 6 | 30.00 |
| cassava_green_mottle | cassava_mosaic | 5 | 25.00 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | mango | 10 | 4.20 |
| potato | tomato | 7 | 13.21 |
| paddy | corn | 6 | 3.80 |
| guava | mango | 6 | 46.15 |
| basil | tomato | 5 | 45.45 |
| tomato | cucumber | 5 | 1.72 |
| apple | cucumber | 4 | 1.68 |
| plum | apple | 4 | 40.00 |
| tomato | pepper_bell | 4 | 1.37 |
| soybean | mango | 3 | 4.17 |
| cotton | cucumber | 3 | 15.79 |
| palm | corn | 3 | 23.08 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |
| apple | sugarcane | 3 | 1.26 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 16 | 4.53 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| healthy | caterpillar | 12 | 3.40 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| mosaic | healthy | 10 | 7.69 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| healthy | leafblast | 7 | 1.98 |
| rust | healthy | 7 | 8.33 |
| healthy | brown_spot | 6 | 1.70 |
| black_rot | anthracnose | 6 | 37.50 |
| healthy | anthracnose | 6 | 1.70 |
| blight_leaf | brown_spot | 6 | 85.71 |

