# clip_bioclip_frozen_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 40.86 | 67.50 | 56.05 | 82.25 |
| crop | 65.53 | 86.74 | 87.80 | 96.10 |
| disease | 41.80 | 72.88 | 58.35 | 86.00 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 16.33 | 55.95 | 12.98 | 52.97 |
| crop | 63.20 | 90.25 | 69.20 | 91.23 |
| disease | 27.72 | 65.49 | 18.85 | 59.60 |
| composition_unseen_only | 78.43 | 98.42 | 77.49 | 97.65 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 23.33 | 61.18 |
| per_image | 21.08 | 64.44 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_healthy | soybean_caterpillar | 15 | 51.72 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 15 | 65.22 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| cassava_green_mottle | cassava_brown_streak_disease | 8 | 40.00 |
| cassava_mosaic | cassava_green_mottle | 8 | 6.45 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |
| paddy_healthy | paddy_hispa | 7 | 21.88 |
| coffee_rust | coffee_healthy | 7 | 35.00 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| paddy_healthy | paddy_blast | 6 | 18.75 |
| paddy_hispa | paddy_leafblast | 6 | 50.00 |
| tomato_septoria_leaf_spot | tomato_brown_spot | 6 | 15.38 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| plum | apple | 5 | 50.00 |
| paddy | wheat | 5 | 3.16 |
| apple | peach | 4 | 1.68 |
| potato | tomato | 4 | 7.55 |
| apple | broccoli | 4 | 1.68 |
| bean | soybean | 4 | 23.53 |
| paddy | corn | 3 | 1.90 |
| apple | plum | 3 | 1.26 |
| sugarcane | paddy | 3 | 6.82 |
| potato | apple | 3 | 5.66 |
| guava | cassava | 3 | 23.08 |
| cherry | apple | 3 | 21.43 |
| corn | mango | 3 | 1.46 |
| tomato | grape | 3 | 1.03 |
| soybean | cassava | 3 | 4.17 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | caterpillar | 15 | 4.25 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| healthy | leafblast | 14 | 3.97 |
| mosaic | healthy | 13 | 10.00 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| rust | healthy | 9 | 10.71 |
| green_mottle | brown_streak_disease | 8 | 40.00 |
| healthy | brown_streak_disease | 7 | 1.98 |
| healthy | hispa | 7 | 1.98 |
| septoria_leaf_spot | brown_spot | 7 | 17.07 |
| stripe_rust | rust | 7 | 70.00 |
| healthy | blast | 6 | 1.70 |
| healthy | rust | 6 | 1.70 |
| hispa | leafblast | 6 | 50.00 |
| bacterial_spot | septoria_leaf_spot | 6 | 11.54 |

