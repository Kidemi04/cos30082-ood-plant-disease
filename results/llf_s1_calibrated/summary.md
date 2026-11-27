# llf_s1_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 45.17 | 74.13 | 63.75 | 89.15 |
| crop | 62.84 | 82.56 | 89.00 | 96.25 |
| disease | 48.29 | 78.07 | 68.40 | 91.75 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 68.85 | 98.01 | 74.38 | 97.51 |
| crop | 85.10 | 98.87 | 85.57 | 98.20 |
| disease | 67.99 | 98.07 | 73.27 | 97.79 |
| composition_unseen_only | 82.50 | 99.37 | 84.81 | 98.90 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 54.55 | 84.41 |
| per_image | 68.66 | 93.14 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 16 | 59.26 |
| tomato_septoria_leaf_spot | tomato_early_blight | 14 | 35.90 |
| paddy_healthy | paddy_blast | 14 | 43.75 |
| corn_streak_virus | corn_brown_spot | 11 | 20.00 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| tomato_late_blight | tomato_early_blight | 9 | 31.03 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 9 | 39.13 |
| tomato_healthy | tomato_early_blight | 9 | 28.12 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| apple_marssonina_blotch | cucumber_downy_mildew | 8 | 57.14 |
| cassava_mosaic | cassava_green_mottle | 8 | 6.45 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 8 | 18.18 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 14 | 5.88 |
| wheat | sugarcane | 10 | 23.81 |
| paddy | corn | 9 | 5.70 |
| potato | tomato | 7 | 13.21 |
| apple | mango | 6 | 2.52 |
| blackgram | cucumber | 5 | 33.33 |
| basil | tomato | 4 | 36.36 |
| tomato | cucumber | 4 | 1.37 |
| coffee | sugarcane | 4 | 10.26 |
| peach | pepper_bell | 4 | 10.00 |
| broccoli | cucumber | 3 | 60.00 |
| cotton | cucumber | 3 | 15.79 |
| soybean | sugarcane | 3 | 4.17 |
| palm | corn | 3 | 23.08 |
| squash | cucumber | 3 | 100.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 14 | 3.97 |
| bacterial_spot | early_blight | 14 | 26.92 |
| septoria_leaf_spot | early_blight | 13 | 31.71 |
| streak_virus | brown_spot | 11 | 20.00 |
| leaf_mold | early_blight | 9 | 37.50 |
| late_blight | early_blight | 9 | 18.00 |
| mosaic_virus | early_blight | 9 | 16.67 |
| mosaic | healthy | 9 | 6.92 |
| caterpillar | diabrotica_speciosa | 9 | 39.13 |
| healthy | early_blight | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| healthy | caterpillar | 8 | 2.27 |
| healthy | leafblast | 8 | 2.27 |
| marssonina_blotch | downy_mildew | 8 | 57.14 |

