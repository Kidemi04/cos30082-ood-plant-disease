# wise_f6_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 45.47 | 76.22 | 65.05 | 89.25 |
| crop | 63.45 | 84.08 | 89.00 | 96.55 |
| disease | 46.48 | 81.61 | 69.15 | 92.15 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 73.00 | 98.74 | 66.37 | 98.41 |
| crop | 83.43 | 99.14 | 84.74 | 98.90 |
| disease | 71.43 | 98.95 | 64.30 | 98.69 |
| composition_unseen_only | 87.44 | 99.55 | 85.29 | 99.31 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 56.04 | 86.03 |
| per_image | 65.70 | 93.61 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| soybean_healthy | soybean_caterpillar | 11 | 37.93 |
| tomato_target_spot | tomato_early_blight | 9 | 60.00 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| apple_healthy | apple_marssonina_blotch | 8 | 16.00 |
| tomato_late_blight | tomato_early_blight | 8 | 27.59 |
| tomato_septoria_leaf_spot | tomato_early_blight | 7 | 17.95 |
| apple_marssonina_blotch | cucumber_downy_mildew | 7 | 50.00 |
| tomato_leaf_mold | tomato_early_blight | 6 | 25.00 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 6 | 22.22 |
| paddy_healthy | paddy_brown_spot | 6 | 18.75 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 9 | 3.78 |
| potato | tomato | 9 | 16.98 |
| wheat | sugarcane | 8 | 19.05 |
| paddy | corn | 7 | 4.43 |
| apple | sugarcane | 5 | 2.10 |
| apple | pepper_bell | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| tomato | pepper_bell | 4 | 1.37 |
| tomato | cucumber | 4 | 1.37 |
| broccoli | cucumber | 3 | 60.00 |
| cotton | cucumber | 3 | 15.79 |
| palm | corn | 3 | 23.08 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |
| plum | apple | 3 | 30.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | blast | 16 | 4.53 |
| healthy | caterpillar | 11 | 3.12 |
| late_blight | early_blight | 10 | 20.00 |
| stripe_rust | rust | 10 | 100.00 |
| target_spot | early_blight | 9 | 42.86 |
| healthy | brown_spot | 9 | 2.55 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| hispa | blast | 8 | 66.67 |
| healthy | marssonina_blotch | 8 | 2.27 |
| septoria_leaf_spot | early_blight | 7 | 17.07 |
| healthy | leafblast | 7 | 1.98 |
| marssonina_blotch | downy_mildew | 7 | 50.00 |
| leaf_mold | early_blight | 6 | 25.00 |
| bacterial_spot | septoria_leaf_spot | 6 | 11.54 |

