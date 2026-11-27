# dci_a4_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 43.44 | 68.57 | 62.30 | 84.15 |
| crop | 61.92 | 77.62 | 86.30 | 94.95 |
| disease | 48.41 | 73.27 | 67.60 | 87.60 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 63.03 | 97.97 | 59.53 | 97.31 |
| crop | 84.10 | 98.46 | 83.36 | 98.07 |
| disease | 65.05 | 98.18 | 59.60 | 97.51 |
| composition_unseen_only | 82.11 | 98.69 | 81.15 | 98.27 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 51.44 | 80.67 |
| per_image | 60.88 | 90.25 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 26 | 81.25 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| tomato_bacterial_spot | tomato_early_blight | 10 | 37.04 |
| potato_early_blight | tomato_early_blight | 10 | 41.67 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| cassava_brown_spot | cassava_bacterial_blight | 8 | 42.11 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| paddy_hispa | paddy_blast | 7 | 58.33 |
| cassava_bacterial_blight | cotton_bacterial_blight | 7 | 30.43 |
| tomato_late_blight | tomato_early_blight | 7 | 24.14 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| tomato_target_spot | tomato_early_blight | 7 | 46.67 |
| tomato_septoria_leaf_spot | tomato_early_blight | 6 | 15.38 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 14 | 5.88 |
| potato | tomato | 12 | 22.64 |
| paddy | corn | 11 | 6.96 |
| cassava | cotton | 9 | 3.35 |
| apple | sugarcane | 7 | 2.94 |
| wheat | sugarcane | 7 | 16.67 |
| tomato | cucumber | 6 | 2.06 |
| peach | pepper_bell | 6 | 15.00 |
| apple | corn | 6 | 2.52 |
| tomato | corn | 5 | 1.72 |
| basil | tomato | 5 | 45.45 |
| blackgram | cucumber | 5 | 33.33 |
| grape | cucumber | 5 | 12.20 |
| cassava | corn | 5 | 1.86 |
| apple | mango | 4 | 1.68 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 25 | 7.08 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| bacterial_spot | early_blight | 9 | 17.31 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | caterpillar | 8 | 2.27 |
| healthy | early_blight | 8 | 2.27 |
| late_blight | early_blight | 8 | 16.00 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| hispa | blast | 7 | 58.33 |
| bacterial_spot | septoria_leaf_spot | 7 | 13.46 |
| target_spot | early_blight | 7 | 33.33 |
| healthy | bacterial_blight | 7 | 1.98 |

