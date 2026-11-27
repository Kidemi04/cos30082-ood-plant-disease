# wise_f6_tuned_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 49.72 | 76.04 | 68.10 | 89.55 |
| crop | 67.86 | 87.49 | 91.30 | 97.15 |
| disease | 50.43 | 82.34 | 71.40 | 92.35 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 83.60 | 99.07 | 81.84 | 99.03 |
| crop | 87.23 | 99.71 | 89.02 | 99.52 |
| disease | 82.89 | 99.27 | 81.08 | 99.17 |
| composition_unseen_only | 91.16 | 99.86 | 90.81 | 99.72 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 62.36 | 86.04 |
| per_image | 74.34 | 94.05 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 15 | 46.88 |
| soybean_healthy | soybean_caterpillar | 14 | 48.28 |
| tomato_septoria_leaf_spot | tomato_early_blight | 11 | 28.21 |
| tomato_bacterial_spot | tomato_early_blight | 11 | 40.74 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 11 | 47.83 |
| paddy_healthy | paddy_leafblast | 9 | 28.12 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| grape_esca | grape_black_rot | 9 | 56.25 |
| paddy_hispa | paddy_leafblast | 8 | 66.67 |
| tomato_target_spot | tomato_early_blight | 7 | 46.67 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 7 | 15.91 |
| corn_leaf_blight | corn_brown_spot | 7 | 43.75 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 6 | 15.38 |
| tomato_healthy | tomato_early_blight | 6 | 18.75 |
| apple_healthy | apple_marssonina_blotch | 6 | 12.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 9 | 5.70 |
| apple | cucumber | 6 | 2.52 |
| apple | mango | 6 | 2.52 |
| potato | tomato | 6 | 11.32 |
| basil | tomato | 4 | 36.36 |
| broccoli | cucumber | 3 | 60.00 |
| cotton | cucumber | 3 | 15.79 |
| plum | cherry | 3 | 30.00 |
| palm | corn | 3 | 23.08 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |
| apple | sugarcane | 3 | 1.26 |
| tomato | pepper_bell | 3 | 1.03 |
| wheat | sugarcane | 3 | 7.14 |
| cherry | peach | 3 | 21.43 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | caterpillar | 14 | 3.97 |
| healthy | blast | 14 | 3.97 |
| septoria_leaf_spot | early_blight | 11 | 26.83 |
| bacterial_spot | early_blight | 11 | 21.15 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| healthy | leafblast | 9 | 2.55 |
| healthy | brown_spot | 9 | 2.55 |
| mosaic | healthy | 9 | 6.92 |
| esca | black_rot | 9 | 56.25 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| target_spot | early_blight | 7 | 33.33 |
| yellow_leaf_curl_virus | early_blight | 7 | 15.91 |
| leaf_blight | brown_spot | 7 | 17.95 |
| brown_spot | rust | 7 | 9.72 |

