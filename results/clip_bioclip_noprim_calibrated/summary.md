# clip_bioclip_noprim_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 50.57 | 71.51 | 68.55 | 85.70 |
| crop | 67.12 | 81.42 | 90.80 | 96.05 |
| disease | 49.69 | 70.63 | 69.95 | 89.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 48.97 | 91.51 | 42.68 | 87.71 |
| crop | 79.54 | 95.69 | 83.49 | 97.31 |
| disease | 48.95 | 89.42 | 41.23 | 85.08 |
| composition_unseen_only | 71.07 | 98.04 | 69.13 | 96.48 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 49.76 | 80.28 |
| per_image | 52.61 | 86.69 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 24 | 75.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| soybean_healthy | soybean_caterpillar | 13 | 44.83 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| cassava_mosaic | cassava_healthy | 10 | 8.06 |
| cassava_mosaic | cassava_green_mottle | 9 | 7.26 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| coffee_rust | coffee_healthy | 6 | 30.00 |
| cassava_brown_spot | cassava_bacterial_blight | 6 | 31.58 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 6 | 15.38 |
| paddy_healthy | paddy_leafblast | 5 | 15.62 |
| cassava_brown_streak_disease | cassava_healthy | 5 | 17.24 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cotton | 8 | 3.36 |
| apple | mango | 5 | 2.10 |
| paddy | corn | 4 | 2.53 |
| basil | tomato | 4 | 36.36 |
| potato | tomato | 4 | 7.55 |
| cotton | cucumber | 3 | 15.79 |
| apple | corn | 3 | 1.26 |
| squash | cucumber | 3 | 100.00 |
| guava | mango | 3 | 23.08 |
| cherry | apple | 3 | 21.43 |
| peach | mango | 3 | 7.50 |
| banana | corn | 3 | 9.38 |
| broccoli | cucumber | 2 | 40.00 |
| soybean | cotton | 2 | 2.78 |
| grape | strawberry | 2 | 4.88 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 24 | 6.80 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| mosaic | healthy | 12 | 9.23 |
| healthy | caterpillar | 11 | 3.12 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| stripe_rust | rust | 9 | 90.00 |
| mosaic | green_mottle | 8 | 6.15 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| rust | healthy | 8 | 9.52 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| hispa | blast | 7 | 58.33 |
| brown_streak_disease | healthy | 6 | 20.69 |
| brown_spot | rust | 6 | 8.33 |
| septoria_leaf_spot | verticulium_wilt | 6 | 14.63 |
| healthy | bacterial_blight | 6 | 1.70 |

