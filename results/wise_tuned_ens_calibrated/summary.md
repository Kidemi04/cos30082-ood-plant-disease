# wise_tuned_ens_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 44.73 | 72.73 | 63.85 | 88.70 |
| crop | 62.57 | 81.05 | 88.35 | 96.05 |
| disease | 47.59 | 79.08 | 69.90 | 91.90 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 88.89 | 99.68 | 86.95 | 99.45 |
| crop | 90.33 | 99.84 | 90.33 | 99.72 |
| disease | 88.29 | 99.77 | 86.26 | 99.59 |
| composition_unseen_only | 91.14 | 99.86 | 90.68 | 99.72 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 59.51 | 84.10 |
| per_image | 73.63 | 93.77 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 21 | 77.78 |
| tomato_septoria_leaf_spot | tomato_early_blight | 19 | 48.72 |
| paddy_healthy | paddy_blast | 18 | 56.25 |
| tomato_healthy | tomato_early_blight | 15 | 46.88 |
| corn_leaf_blight | corn_brown_spot | 13 | 81.25 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| tomato_leaf_mold | tomato_early_blight | 11 | 45.83 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| paddy_hispa | paddy_blast | 9 | 75.00 |
| tomato_target_spot | tomato_early_blight | 9 | 60.00 |
| tomato_late_blight | tomato_early_blight | 9 | 31.03 |
| grape_esca | grape_black_rot | 9 | 56.25 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| mango_powdery_mildew | cucumber_powdery_mildew | 7 | 100.00 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 7 | 15.91 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 15 | 9.49 |
| wheat | sugarcane | 12 | 28.57 |
| potato | tomato | 8 | 15.09 |
| pumpkin | cucumber | 7 | 43.75 |
| apple | cucumber | 6 | 2.52 |
| apple | sugarcane | 6 | 2.52 |
| apple | mango | 6 | 2.52 |
| cotton | cucumber | 5 | 26.32 |
| mango | cucumber | 5 | 9.43 |
| basil | tomato | 4 | 36.36 |
| tomato | pepper_bell | 4 | 1.37 |
| apple | tomato | 4 | 1.68 |
| broccoli | cucumber | 3 | 60.00 |
| plum | cherry | 3 | 30.00 |
| palm | corn | 3 | 23.08 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 19 | 46.34 |
| healthy | blast | 18 | 5.10 |
| bacterial_spot | early_blight | 16 | 30.77 |
| healthy | early_blight | 14 | 3.97 |
| leaf_blight | brown_spot | 13 | 33.33 |
| caterpillar | diabrotica_speciosa | 12 | 52.17 |
| healthy | caterpillar | 10 | 2.83 |
| leaf_mold | early_blight | 10 | 41.67 |
| hispa | blast | 9 | 75.00 |
| late_blight | early_blight | 9 | 18.00 |
| esca | black_rot | 9 | 56.25 |
| stripe_rust | rust | 9 | 90.00 |
| target_spot | early_blight | 8 | 38.10 |
| healthy | brown_spot | 8 | 2.27 |
| mosaic | healthy | 8 | 6.15 |

