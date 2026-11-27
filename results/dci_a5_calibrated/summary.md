# dci_a5_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 44.05 | 66.64 | 65.40 | 83.10 |
| crop | 63.30 | 76.56 | 88.75 | 94.40 |
| disease | 48.09 | 71.32 | 68.85 | 86.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 61.91 | 95.69 | 58.49 | 94.96 |
| crop | 82.36 | 96.25 | 82.25 | 95.79 |
| disease | 63.78 | 95.94 | 58.49 | 95.30 |
| composition_unseen_only | 77.80 | 96.74 | 78.87 | 96.06 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 51.48 | 78.57 |
| per_image | 61.75 | 88.63 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| tomato_healthy | tomato_early_blight | 15 | 46.88 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| tomato_bacterial_spot | tomato_early_blight | 10 | 37.04 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| corn_leaf_blight | corn_brown_spot | 8 | 50.00 |
| tomato_late_blight | tomato_early_blight | 8 | 27.59 |
| paddy_hispa | paddy_blast | 7 | 58.33 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| tomato_septoria_leaf_spot | tomato_early_blight | 7 | 17.95 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 11 | 4.62 |
| potato | tomato | 8 | 15.09 |
| paddy | corn | 7 | 4.43 |
| basil | tomato | 6 | 54.55 |
| apple | mango | 5 | 2.10 |
| wheat | sugarcane | 5 | 11.90 |
| apple | tomato | 5 | 2.10 |
| apple | corn | 5 | 2.10 |
| soybean | tomato | 4 | 5.56 |
| cherry | apple | 4 | 28.57 |
| grape | cucumber | 4 | 9.76 |
| cabbage | grape | 3 | 75.00 |
| apple | sugarcane | 3 | 1.26 |
| paddy | sugarcane | 3 | 1.90 |
| cassava | corn | 3 | 1.12 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | early_blight | 18 | 5.10 |
| healthy | blast | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | leafblast | 11 | 3.12 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| healthy | caterpillar | 9 | 2.55 |
| bacterial_spot | early_blight | 9 | 17.31 |
| stripe_rust | rust | 9 | 90.00 |
| leaf_blight | brown_spot | 8 | 20.51 |
| late_blight | early_blight | 8 | 16.00 |
| hispa | blast | 7 | 58.33 |
| bacterial_spot | septoria_leaf_spot | 7 | 13.46 |
| mosaic | healthy | 7 | 5.38 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| septoria_leaf_spot | early_blight | 7 | 17.07 |

