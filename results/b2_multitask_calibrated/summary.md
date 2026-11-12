# b2_multitask_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 41.78 | 65.36 | 61.65 | 84.85 |
| crop | 58.09 | 72.36 | 85.50 | 93.40 |
| disease | 43.17 | 69.95 | 67.25 | 88.35 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 66.44 | 95.18 | 58.01 | 93.65 |
| crop | 80.05 | 95.74 | 77.56 | 94.27 |
| disease | 65.70 | 95.80 | 56.70 | 94.20 |
| composition_unseen_only | 79.28 | 97.09 | 76.59 | 95.44 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 51.30 | 77.50 |
| per_image | 59.78 | 89.03 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 19 | 59.38 |
| tomato_septoria_leaf_spot | tomato_early_blight | 18 | 46.15 |
| tomato_healthy | tomato_early_blight | 15 | 46.88 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 15 | 65.22 |
| soybean_healthy | soybean_diabrotica_speciosa | 11 | 37.93 |
| paddy_hispa | paddy_blast | 10 | 83.33 |
| tomato_bacterial_spot | tomato_early_blight | 10 | 37.04 |
| cassava_bacterial_blight | cotton_bacterial_blight | 8 | 34.78 |
| coffee_rust | coffee_healthy | 8 | 40.00 |
| apple_rust | sugarcane_rust | 7 | 20.00 |
| cassava_green_mottle | cassava_mosaic | 6 | 30.00 |
| potato_early_blight | tomato_early_blight | 6 | 25.00 |
| paddy_healthy | paddy_hispa | 6 | 18.75 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | sugarcane | 11 | 4.62 |
| cassava | cotton | 10 | 3.72 |
| wheat | sugarcane | 9 | 21.43 |
| paddy | corn | 7 | 4.43 |
| potato | tomato | 7 | 13.21 |
| apple | cucumber | 6 | 2.52 |
| apple | mango | 6 | 2.52 |
| cotton | cucumber | 5 | 26.32 |
| paddy | sugarcane | 5 | 3.16 |
| pumpkin | cucumber | 5 | 31.25 |
| palm | corn | 4 | 30.77 |
| corn | sugarcane | 4 | 1.94 |
| tomato | pepper_bell | 4 | 1.37 |
| basil | tomato | 4 | 36.36 |
| apple | corn | 4 | 1.68 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 19 | 5.38 |
| septoria_leaf_spot | early_blight | 17 | 41.46 |
| healthy | early_blight | 15 | 4.25 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| hispa | blast | 10 | 83.33 |
| healthy | diabrotica_speciosa | 10 | 2.83 |
| healthy | bacterial_blight | 8 | 2.27 |
| bacterial_spot | early_blight | 8 | 15.38 |
| rust | healthy | 8 | 9.52 |
| stripe_rust | rust | 8 | 80.00 |
| mosaic | healthy | 7 | 5.38 |
| blast | brown_spot | 6 | 27.27 |
| green_mottle | mosaic | 6 | 30.00 |
| healthy | brown_spot | 6 | 1.70 |
| healthy | hispa | 6 | 1.70 |

