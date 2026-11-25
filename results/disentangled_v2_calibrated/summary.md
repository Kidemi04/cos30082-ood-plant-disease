# v2_final_winner_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 44.68 | 68.68 | 59.15 | 83.20 |
| crop | 62.80 | 77.91 | 85.50 | 94.10 |
| disease | 51.38 | 76.82 | 66.60 | 87.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 61.38 | 94.65 | 58.70 | 91.71 |
| crop | 80.80 | 95.42 | 77.42 | 92.89 |
| disease | 60.85 | 95.16 | 57.80 | 92.20 |
| composition_unseen_only | 80.49 | 96.87 | 77.42 | 94.48 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 51.71 | 79.60 |
| per_image | 58.92 | 87.25 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple_rust | sugarcane_rust | 27 | 77.14 |
| tomato_septoria_leaf_spot | tomato_early_blight | 23 | 58.97 |
| paddy_healthy | paddy_blast | 23 | 71.88 |
| tomato_healthy | tomato_early_blight | 20 | 62.50 |
| cassava_bacterial_blight | cotton_bacterial_blight | 18 | 78.26 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| tomato_bacterial_spot | tomato_early_blight | 12 | 44.44 |
| paddy_hispa | paddy_blast | 11 | 91.67 |
| tomato_bacterial_spot | pepper_bell_bacterial_spot | 9 | 33.33 |
| corn_leaf_blight | corn_brown_spot | 8 | 50.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| tomato_target_spot | tomato_early_blight | 7 | 46.67 |
| tomato_leaf_mold | tomato_early_blight | 7 | 29.17 |
| mango_powdery_mildew | cucumber_powdery_mildew | 7 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | sugarcane | 25 | 10.50 |
| cassava | cotton | 20 | 7.43 |
| apple | cucumber | 9 | 3.78 |
| mango | cucumber | 8 | 15.09 |
| potato | tomato | 5 | 9.43 |
| apple | cherry | 5 | 2.10 |
| soybean | cucumber | 4 | 5.56 |
| apple | cotton | 4 | 1.68 |
| cherry | apple | 4 | 28.57 |
| basil | tomato | 4 | 36.36 |
| paddy | sugarcane | 4 | 2.53 |
| tomato | cucumber | 4 | 1.37 |
| cotton | cucumber | 3 | 15.79 |
| plum | cherry | 3 | 30.00 |
| squash | cucumber | 3 | 100.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 23 | 56.10 |
| healthy | blast | 23 | 6.52 |
| healthy | early_blight | 20 | 5.67 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| bacterial_spot | early_blight | 12 | 23.08 |
| hispa | blast | 11 | 91.67 |
| leaf_blight | brown_spot | 8 | 20.51 |
| late_blight | early_blight | 8 | 16.00 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| target_spot | early_blight | 7 | 33.33 |
| healthy | bacterial_blight | 7 | 1.98 |
| leaf_mold | early_blight | 7 | 29.17 |
| mosaic | healthy | 7 | 5.38 |
| mosaic_virus | early_blight | 7 | 12.96 |
| healthy | caterpillar | 6 | 1.70 |

