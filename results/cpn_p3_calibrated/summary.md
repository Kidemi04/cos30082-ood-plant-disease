# cpn_p3_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 44.11 | 69.40 | 63.05 | 84.35 |
| crop | 61.58 | 80.53 | 88.75 | 95.15 |
| disease | 48.22 | 73.63 | 68.25 | 87.40 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 63.24 | 95.51 | 56.28 | 94.48 |
| crop | 83.17 | 96.68 | 82.87 | 95.58 |
| disease | 63.15 | 96.51 | 55.59 | 95.23 |
| composition_unseen_only | 83.77 | 97.21 | 81.08 | 96.20 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 51.97 | 80.39 |
| per_image | 59.48 | 89.13 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| tomato_bacterial_spot | tomato_early_blight | 14 | 51.85 |
| coffee_rust | sugarcane_rust | 13 | 65.00 |
| peach_bacterial_spot | pepper_bell_bacterial_spot | 12 | 63.16 |
| potato_early_blight | tomato_early_blight | 12 | 50.00 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| corn_leaf_blight | corn_brown_spot | 11 | 68.75 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 11 | 47.83 |
| tomato_septoria_leaf_spot | tomato_early_blight | 9 | 23.08 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| tomato_late_blight | tomato_early_blight | 9 | 31.03 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_healthy | tomato_early_blight | 8 | 25.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 12 | 7.59 |
| potato | tomato | 12 | 22.64 |
| peach | pepper_bell | 11 | 27.50 |
| apple | cucumber | 5 | 2.10 |
| palm | corn | 5 | 38.46 |
| coffee | sugarcane | 5 | 12.82 |
| basil | tomato | 4 | 36.36 |
| cherry | apple | 4 | 28.57 |
| banana | corn | 4 | 12.50 |
| wheat | corn | 4 | 9.52 |
| blackgram | cucumber | 4 | 26.67 |
| cotton | cucumber | 3 | 15.79 |
| mango | cotton | 3 | 5.66 |
| cabbage | grape | 3 | 75.00 |
| plum | apple | 3 | 30.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | brown_spot | 16 | 4.53 |
| healthy | blast | 15 | 4.25 |
| bacterial_spot | early_blight | 14 | 26.92 |
| mosaic | healthy | 13 | 10.00 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| leaf_blight | brown_spot | 11 | 28.21 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| septoria_leaf_spot | early_blight | 9 | 21.95 |
| late_blight | early_blight | 9 | 18.00 |
| hispa | blast | 8 | 66.67 |
| healthy | early_blight | 8 | 2.27 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| stripe_rust | rust | 8 | 80.00 |
| healthy | caterpillar | 7 | 1.98 |
| healthy | leafblast | 7 | 1.98 |

