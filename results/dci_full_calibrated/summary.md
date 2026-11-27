# dci_full_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 43.53 | 69.47 | 63.10 | 85.35 |
| crop | 63.48 | 79.16 | 87.10 | 95.00 |
| disease | 48.93 | 75.85 | 67.90 | 88.85 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 60.37 | 96.97 | 55.39 | 96.20 |
| crop | 81.24 | 97.90 | 80.46 | 97.38 |
| disease | 63.74 | 97.31 | 56.01 | 96.48 |
| composition_unseen_only | 80.05 | 98.37 | 77.42 | 97.79 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 50.59 | 80.95 |
| per_image | 58.99 | 90.45 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 16 | 50.00 |
| tomato_healthy | tomato_early_blight | 16 | 50.00 |
| paddy_healthy | paddy_blast | 14 | 43.75 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| cassava_mosaic | cassava_green_mottle | 11 | 8.87 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 10 | 25.64 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| tomato_late_blight | tomato_early_blight | 9 | 31.03 |
| tomato_bacterial_spot | tomato_early_blight | 7 | 25.93 |
| paddy_brown_spot | corn_brown_spot | 7 | 43.75 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| potato_early_blight | tomato_early_blight | 6 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 12 | 7.59 |
| apple | cucumber | 10 | 4.20 |
| potato | tomato | 7 | 13.21 |
| cassava | corn | 7 | 2.60 |
| apple | corn | 6 | 2.52 |
| apple | tomato | 6 | 2.52 |
| corn | sugarcane | 5 | 2.43 |
| blackgram | cucumber | 5 | 33.33 |
| wheat | sugarcane | 5 | 11.90 |
| peach | pepper_bell | 5 | 12.50 |
| peach | cucumber | 4 | 10.00 |
| apple | mango | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| grape | cucumber | 4 | 9.76 |
| soybean | cucumber | 3 | 4.17 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | early_blight | 18 | 5.10 |
| healthy | leafblast | 16 | 4.53 |
| healthy | blast | 14 | 3.97 |
| healthy | caterpillar | 12 | 3.40 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| caterpillar | diabrotica_speciosa | 12 | 52.17 |
| mosaic | green_mottle | 11 | 8.46 |
| septoria_leaf_spot | verticulium_wilt | 10 | 24.39 |
| late_blight | early_blight | 10 | 20.00 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| hispa | leafblast | 9 | 75.00 |
| stripe_rust | rust | 8 | 80.00 |
| bacterial_spot | early_blight | 6 | 11.54 |
| healthy | bacterial_blight | 6 | 1.70 |
| healthy | brown_streak_disease | 5 | 1.42 |

