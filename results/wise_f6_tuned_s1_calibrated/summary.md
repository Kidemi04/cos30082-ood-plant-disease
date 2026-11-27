# wise_f6_tuned_s1_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 42.82 | 74.22 | 62.10 | 88.65 |
| crop | 60.83 | 81.50 | 87.60 | 96.05 |
| disease | 45.93 | 79.86 | 68.25 | 91.75 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 85.39 | 99.56 | 83.49 | 99.45 |
| crop | 88.19 | 99.25 | 88.74 | 99.38 |
| disease | 85.26 | 99.26 | 82.94 | 99.24 |
| composition_unseen_only | 88.76 | 99.79 | 88.74 | 99.65 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 57.04 | 85.04 |
| per_image | 71.23 | 93.74 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_septoria_leaf_spot | tomato_early_blight | 25 | 64.10 |
| tomato_healthy | tomato_early_blight | 19 | 59.38 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_blast | 18 | 56.25 |
| tomato_bacterial_spot | tomato_early_blight | 17 | 62.96 |
| corn_leaf_blight | corn_brown_spot | 11 | 68.75 |
| grape_esca | grape_black_rot | 10 | 62.50 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| tomato_target_spot | tomato_early_blight | 9 | 60.00 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| wheat_stripe_rust | sugarcane_rust | 8 | 80.00 |
| mango_powdery_mildew | cucumber_powdery_mildew | 7 | 100.00 |
| tomato_late_blight | tomato_early_blight | 7 | 24.14 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 15 | 9.49 |
| wheat | sugarcane | 15 | 35.71 |
| apple | sugarcane | 10 | 4.20 |
| apple | cucumber | 7 | 2.94 |
| potato | tomato | 7 | 13.21 |
| pumpkin | cucumber | 7 | 43.75 |
| mango | cucumber | 6 | 11.32 |
| cotton | cucumber | 5 | 26.32 |
| broccoli | cucumber | 4 | 80.00 |
| apple | mango | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| tomato | pepper_bell | 4 | 1.37 |
| apple | tomato | 4 | 1.68 |
| cassava | cotton | 4 | 1.49 |
| apple | corn | 4 | 1.68 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 25 | 60.98 |
| healthy | early_blight | 19 | 5.38 |
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 17 | 4.82 |
| bacterial_spot | early_blight | 15 | 28.85 |
| leaf_blight | brown_spot | 11 | 28.21 |
| healthy | brown_spot | 10 | 2.83 |
| esca | black_rot | 10 | 62.50 |
| hispa | blast | 9 | 75.00 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| target_spot | early_blight | 9 | 42.86 |
| stripe_rust | rust | 9 | 90.00 |
| leaf_mold | early_blight | 8 | 33.33 |
| late_blight | early_blight | 8 | 16.00 |
| mosaic | healthy | 8 | 6.15 |

