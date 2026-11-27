# wise_datacomp_tuned_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 46.37 | 74.14 | 63.00 | 89.75 |
| crop | 63.04 | 83.06 | 88.75 | 96.60 |
| disease | 50.01 | 82.27 | 68.50 | 93.75 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 84.41 | 99.39 | 83.63 | 99.10 |
| crop | 91.36 | 99.37 | 90.61 | 99.17 |
| disease | 83.89 | 99.39 | 82.94 | 99.10 |
| composition_unseen_only | 92.37 | 99.64 | 91.23 | 99.45 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 59.86 | 84.93 |
| per_image | 71.86 | 94.19 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 23 | 85.19 |
| tomato_septoria_leaf_spot | tomato_early_blight | 20 | 51.28 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| tomato_late_blight | tomato_early_blight | 14 | 48.28 |
| potato_early_blight | tomato_early_blight | 13 | 54.17 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 13 | 56.52 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| grape_esca | grape_black_rot | 12 | 75.00 |
| tomato_healthy | tomato_early_blight | 10 | 31.25 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| corn_leaf_blight | corn_brown_spot | 9 | 56.25 |
| paddy_healthy | paddy_blast | 9 | 28.12 |
| paddy_hispa | paddy_leafblast | 8 | 66.67 |
| apple_healthy | apple_marssonina_blotch | 8 | 16.00 |
| pumpkin_powdery_mildew | cucumber_powdery_mildew | 8 | 50.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 14 | 26.42 |
| paddy | corn | 8 | 5.06 |
| pumpkin | cucumber | 8 | 50.00 |
| blackgram | cucumber | 7 | 46.67 |
| wheat | sugarcane | 7 | 16.67 |
| apple | sugarcane | 6 | 2.52 |
| mango | cucumber | 6 | 11.32 |
| cassava | cotton | 5 | 1.86 |
| apple | tomato | 5 | 2.10 |
| plum | cherry | 4 | 40.00 |
| apple | mango | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| guava | mango | 4 | 30.77 |
| cotton | cucumber | 3 | 15.79 |
| soybean | tomato | 3 | 4.17 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 22 | 42.31 |
| septoria_leaf_spot | early_blight | 20 | 48.78 |
| late_blight | early_blight | 17 | 34.00 |
| caterpillar | diabrotica_speciosa | 13 | 56.52 |
| healthy | leafblast | 12 | 3.40 |
| esca | black_rot | 12 | 75.00 |
| healthy | caterpillar | 11 | 3.12 |
| healthy | early_blight | 11 | 3.12 |
| leaf_mold | early_blight | 9 | 37.50 |
| leaf_blight | brown_spot | 9 | 23.08 |
| healthy | blast | 9 | 2.55 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| healthy | bacterial_blight | 8 | 2.27 |

