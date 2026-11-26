# clip_bioclip_heads_strong_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 45.41 | 76.13 | 62.20 | 88.15 |
| crop | 64.52 | 84.50 | 87.85 | 96.75 |
| disease | 51.43 | 83.46 | 68.25 | 91.85 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 67.75 | 97.04 | 70.58 | 97.24 |
| crop | 86.75 | 99.04 | 84.12 | 98.55 |
| disease | 68.15 | 97.39 | 69.61 | 97.65 |
| composition_unseen_only | 82.55 | 99.53 | 81.77 | 99.17 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 54.38 | 85.33 |
| per_image | 66.13 | 92.47 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 20 | 74.07 |
| apple_rust | sugarcane_rust | 17 | 48.57 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| paddy_healthy | paddy_blast | 14 | 43.75 |
| potato_early_blight | tomato_early_blight | 11 | 45.83 |
| tomato_late_blight | tomato_early_blight | 10 | 34.48 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| tomato_septoria_leaf_spot | tomato_early_blight | 8 | 20.51 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| tomato_healthy | tomato_early_blight | 7 | 21.88 |
| grape_esca | grape_black_rot | 7 | 43.75 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | sugarcane | 16 | 6.72 |
| potato | tomato | 12 | 22.64 |
| apple | cucumber | 10 | 4.20 |
| wheat | sugarcane | 9 | 21.43 |
| cassava | cotton | 8 | 2.97 |
| paddy | corn | 8 | 5.06 |
| apple | mango | 6 | 2.52 |
| tomato | corn | 5 | 1.72 |
| tomato | pepper_bell | 5 | 1.72 |
| banana | corn | 5 | 15.62 |
| cherry | apple | 4 | 28.57 |
| peach | pepper_bell | 4 | 10.00 |
| cotton | cucumber | 3 | 15.79 |
| squash | cucumber | 3 | 100.00 |
| cabbage | grape | 3 | 75.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 19 | 36.54 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| healthy | leafblast | 14 | 3.97 |
| healthy | blast | 13 | 3.68 |
| late_blight | early_blight | 11 | 22.00 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| leaf_mold | early_blight | 9 | 37.50 |
| healthy | bacterial_blight | 9 | 2.55 |
| septoria_leaf_spot | early_blight | 8 | 19.51 |
| healthy | caterpillar | 8 | 2.27 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| stripe_rust | rust | 8 | 80.00 |
| marssonina_blotch | downy_mildew | 7 | 50.00 |
| esca | black_rot | 7 | 43.75 |

