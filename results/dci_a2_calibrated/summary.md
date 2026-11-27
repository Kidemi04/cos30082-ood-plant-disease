# dci_a2_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 41.81 | 68.52 | 62.65 | 84.35 |
| crop | 62.84 | 80.79 | 87.55 | 95.20 |
| disease | 45.46 | 74.61 | 67.25 | 88.35 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 67.19 | 98.62 | 64.36 | 98.07 |
| crop | 85.58 | 98.92 | 84.32 | 98.55 |
| disease | 68.13 | 98.70 | 64.16 | 98.20 |
| composition_unseen_only | 80.75 | 99.23 | 81.70 | 98.90 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 51.55 | 80.86 |
| per_image | 63.50 | 90.69 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 21 | 65.62 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| tomato_bacterial_spot | tomato_early_blight | 13 | 48.15 |
| tomato_septoria_leaf_spot | tomato_early_blight | 11 | 28.21 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| tomato_healthy | tomato_early_blight | 8 | 25.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| tomato_leaf_mold | tomato_early_blight | 7 | 29.17 |
| potato_early_blight | tomato_early_blight | 7 | 29.17 |
| grape_esca | grape_black_rot | 7 | 43.75 |
| paddy_hispa | paddy_leafblast | 6 | 50.00 |
| cassava_bacterial_blight | cotton_bacterial_blight | 6 | 26.09 |
| paddy_hispa | paddy_blast | 6 | 50.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 14 | 5.88 |
| paddy | corn | 9 | 5.70 |
| potato | tomato | 8 | 15.09 |
| cassava | cotton | 7 | 2.60 |
| wheat | sugarcane | 7 | 16.67 |
| apple | mango | 6 | 2.52 |
| corn | sugarcane | 5 | 2.43 |
| cherry | apple | 5 | 35.71 |
| apple | tomato | 5 | 2.10 |
| basil | tomato | 4 | 36.36 |
| peach | pepper_bell | 4 | 10.00 |
| blackgram | cucumber | 4 | 26.67 |
| broccoli | cucumber | 3 | 60.00 |
| peach | cucumber | 3 | 7.50 |
| soybean | tomato | 3 | 4.17 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 21 | 5.95 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| bacterial_spot | early_blight | 12 | 23.08 |
| septoria_leaf_spot | early_blight | 11 | 26.83 |
| healthy | caterpillar | 10 | 2.83 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| healthy | early_blight | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | leafblast | 8 | 2.27 |
| healthy | bacterial_blight | 8 | 2.27 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| leaf_mold | early_blight | 7 | 29.17 |
| esca | black_rot | 7 | 43.75 |
| hispa | leafblast | 6 | 50.00 |
| leaf_blight | early_blight | 6 | 15.38 |

