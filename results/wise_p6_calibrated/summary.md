# wise_p6_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 41.91 | 68.73 | 61.80 | 85.00 |
| crop | 66.45 | 82.91 | 89.05 | 96.20 |
| disease | 40.46 | 68.92 | 65.10 | 88.75 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 54.64 | 88.44 | 60.70 | 90.47 |
| crop | 68.25 | 92.96 | 75.69 | 94.89 |
| disease | 56.02 | 92.00 | 60.77 | 92.68 |
| composition_unseen_only | 68.73 | 96.47 | 73.34 | 96.89 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 47.43 | 77.35 |
| per_image | 61.25 | 87.65 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| cassava_mosaic | cassava_healthy | 11 | 8.87 |
| tomato_target_spot | tomato_early_blight | 10 | 66.67 |
| tomato_septoria_leaf_spot | tomato_early_blight | 9 | 23.08 |
| apple_healthy | apple_marssonina_blotch | 9 | 18.00 |
| tomato_bacterial_spot | tomato_early_blight | 9 | 33.33 |
| coffee_rust | coffee_healthy | 9 | 45.00 |
| grape_esca | grape_black_rot | 9 | 56.25 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 7 | 17.95 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato | cucumber | 8 | 2.75 |
| paddy | corn | 7 | 4.43 |
| apple | mango | 7 | 2.94 |
| potato | tomato | 7 | 13.21 |
| apple | cucumber | 6 | 2.52 |
| guava | mango | 6 | 46.15 |
| cotton | cucumber | 4 | 21.05 |
| banana | mango | 4 | 12.50 |
| cassava | cotton | 4 | 1.49 |
| eggplant | tomato | 4 | 80.00 |
| squash | cucumber | 3 | 100.00 |
| plum | apple | 3 | 30.00 |
| wheat | corn | 3 | 7.14 |
| potato | apple | 3 | 5.66 |
| apple | cotton | 3 | 1.26 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| mosaic | healthy | 12 | 9.23 |
| healthy | leafblast | 11 | 3.12 |
| healthy | blast | 11 | 3.12 |
| rust | healthy | 11 | 13.10 |
| target_spot | early_blight | 10 | 47.62 |
| healthy | marssonina_blotch | 9 | 2.55 |
| healthy | brown_spot | 9 | 2.55 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| esca | black_rot | 8 | 50.00 |
| septoria_leaf_spot | early_blight | 7 | 17.07 |
| healthy | caterpillar | 7 | 1.98 |
| late_blight | early_blight | 7 | 14.00 |
| bacterial_spot | early_blight | 7 | 13.46 |

