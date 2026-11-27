# llf_s2_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 46.17 | 75.19 | 65.85 | 88.95 |
| crop | 62.04 | 82.70 | 89.45 | 96.50 |
| disease | 47.27 | 81.58 | 69.85 | 92.00 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 66.14 | 97.08 | 66.16 | 97.31 |
| crop | 86.38 | 98.45 | 86.19 | 98.00 |
| disease | 65.65 | 97.34 | 65.12 | 97.65 |
| composition_unseen_only | 83.73 | 98.91 | 84.19 | 98.55 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 54.38 | 84.74 |
| per_image | 66.00 | 92.94 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 19 | 59.38 |
| tomato_bacterial_spot | tomato_early_blight | 14 | 51.85 |
| tomato_late_blight | tomato_early_blight | 12 | 41.38 |
| soybean_healthy | soybean_caterpillar | 11 | 37.93 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 11 | 47.83 |
| tomato_target_spot | tomato_early_blight | 9 | 60.00 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| paddy_hispa | paddy_leafblast | 8 | 66.67 |
| tomato_septoria_leaf_spot | tomato_early_blight | 8 | 20.51 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| apple_marssonina_blotch | cucumber_downy_mildew | 7 | 50.00 |
| paddy_healthy | paddy_blast | 7 | 21.88 |
| potato_early_blight | tomato_early_blight | 7 | 29.17 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 11 | 20.75 |
| paddy | corn | 8 | 5.06 |
| apple | cucumber | 8 | 3.36 |
| blackgram | cucumber | 5 | 33.33 |
| wheat | sugarcane | 5 | 11.90 |
| apple | tomato | 5 | 2.10 |
| banana | corn | 5 | 15.62 |
| apple | sugarcane | 4 | 1.68 |
| grape | cucumber | 4 | 9.76 |
| broccoli | cucumber | 3 | 60.00 |
| cotton | cucumber | 3 | 15.79 |
| plum | cherry | 3 | 30.00 |
| plum | apple | 3 | 30.00 |
| apple | mango | 3 | 1.26 |
| wheat | corn | 3 | 7.14 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 19 | 5.38 |
| bacterial_spot | early_blight | 14 | 26.92 |
| late_blight | early_blight | 12 | 24.00 |
| healthy | caterpillar | 11 | 3.12 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| target_spot | early_blight | 9 | 42.86 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| septoria_leaf_spot | early_blight | 8 | 19.51 |
| leaf_mold | early_blight | 8 | 33.33 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| mosaic | healthy | 8 | 6.15 |
| marssonina_blotch | downy_mildew | 7 | 50.00 |
| healthy | blast | 7 | 1.98 |

