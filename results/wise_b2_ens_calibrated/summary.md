# wise_b2_ens_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 54.59 | 83.79 | 70.20 | 92.60 |
| crop | 75.80 | 92.86 | 93.30 | 98.30 |
| disease | 57.38 | 87.35 | 73.45 | 94.35 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 81.02 | 99.42 | 74.17 | 99.38 |
| crop | 95.80 | 99.61 | 95.72 | 99.38 |
| disease | 80.88 | 99.42 | 73.90 | 99.38 |
| composition_unseen_only | 94.17 | 99.76 | 95.10 | 99.59 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 65.23 | 90.94 |
| per_image | 72.13 | 95.87 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_septoria_leaf_spot | tomato_early_blight | 18 | 46.15 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| tomato_bacterial_spot | tomato_early_blight | 18 | 66.67 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| soybean_healthy | soybean_caterpillar | 13 | 44.83 |
| paddy_healthy | paddy_blast | 12 | 37.50 |
| potato_early_blight | tomato_early_blight | 10 | 41.67 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| tomato_late_blight | tomato_early_blight | 8 | 27.59 |
| tomato_leaf_mold | tomato_early_blight | 7 | 29.17 |
| paddy_hispa | paddy_blast | 7 | 58.33 |
| tomato_healthy | tomato_early_blight | 7 | 21.88 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 6 | 15.38 |
| cotton_powdery_mildew | cucumber_powdery_mildew | 5 | 100.00 |
| tomato_target_spot | tomato_early_blight | 5 | 33.33 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 10 | 18.87 |
| paddy | corn | 5 | 3.16 |
| cotton | cucumber | 5 | 26.32 |
| wheat | sugarcane | 5 | 11.90 |
| apple | tomato | 4 | 1.68 |
| plum | cherry | 4 | 40.00 |
| apple | cucumber | 4 | 1.68 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |
| apple | peach | 3 | 1.26 |
| apple | grape | 3 | 1.26 |
| apple | sugarcane | 3 | 1.26 |
| zucchini | cucumber | 3 | 75.00 |
| paddy | sugarcane | 3 | 1.90 |
| corn | sugarcane | 2 | 0.97 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 18 | 43.90 |
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| bacterial_spot | early_blight | 17 | 32.69 |
| healthy | leafblast | 14 | 3.97 |
| healthy | caterpillar | 12 | 3.40 |
| healthy | blast | 12 | 3.40 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| late_blight | early_blight | 8 | 16.00 |
| leaf_mold | early_blight | 7 | 29.17 |
| hispa | blast | 7 | 58.33 |
| healthy | early_blight | 7 | 1.98 |
| stripe_rust | rust | 7 | 70.00 |
| septoria_leaf_spot | leaf_blight | 6 | 14.63 |
| blast | brown_spot | 5 | 22.73 |
| healthy | brown_spot | 5 | 1.42 |

