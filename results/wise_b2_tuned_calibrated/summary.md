# wise_b2_tuned_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 52.66 | 81.48 | 69.50 | 91.70 |
| crop | 71.77 | 89.78 | 92.00 | 97.65 |
| disease | 55.35 | 85.47 | 73.65 | 94.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 78.80 | 99.24 | 70.99 | 99.10 |
| crop | 94.90 | 99.62 | 94.68 | 99.38 |
| disease | 78.84 | 99.31 | 69.54 | 99.17 |
| composition_unseen_only | 92.24 | 99.73 | 92.89 | 99.52 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 63.13 | 89.48 |
| per_image | 70.24 | 95.26 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| tomato_bacterial_spot | tomato_early_blight | 18 | 66.67 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| tomato_septoria_leaf_spot | tomato_early_blight | 12 | 30.77 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| paddy_healthy | paddy_leafblast | 9 | 28.12 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| potato_early_blight | tomato_early_blight | 9 | 37.50 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| wheat_rust | sugarcane_rust | 7 | 77.78 |
| tomato_leaf_mold | tomato_early_blight | 6 | 25.00 |
| tomato_late_blight | tomato_early_blight | 6 | 20.69 |
| cotton_powdery_mildew | cucumber_powdery_mildew | 5 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 9 | 16.98 |
| wheat | sugarcane | 7 | 16.67 |
| paddy | corn | 6 | 3.80 |
| cotton | cucumber | 5 | 26.32 |
| apple | sugarcane | 5 | 2.10 |
| apple | tomato | 4 | 1.68 |
| blueberry | sugarcane | 4 | 23.53 |
| apple | cucumber | 4 | 1.68 |
| broccoli | cucumber | 3 | 60.00 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |
| basil | tomato | 3 | 27.27 |
| lettuce | cucumber | 3 | 100.00 |
| plum | cherry | 3 | 30.00 |
| grape | cucumber | 3 | 7.32 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | blast | 16 | 4.53 |
| bacterial_spot | early_blight | 15 | 28.85 |
| septoria_leaf_spot | early_blight | 12 | 29.27 |
| healthy | caterpillar | 12 | 3.40 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| healthy | leafblast | 9 | 2.55 |
| mosaic | healthy | 9 | 6.92 |
| hispa | blast | 8 | 66.67 |
| stripe_rust | rust | 7 | 70.00 |
| leaf_mold | early_blight | 6 | 25.00 |
| late_blight | early_blight | 6 | 12.00 |
| healthy | bacterial_blight | 6 | 1.70 |
| blast | brown_spot | 5 | 22.73 |

