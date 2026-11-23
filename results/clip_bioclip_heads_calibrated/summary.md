# clip_bioclip_heads_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 47.40 | 75.44 | 66.25 | 89.50 |
| crop | 66.18 | 84.88 | 90.10 | 96.90 |
| disease | 50.73 | 82.81 | 70.60 | 92.60 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 67.23 | 97.74 | 66.85 | 97.51 |
| crop | 88.02 | 99.01 | 87.50 | 98.62 |
| disease | 65.43 | 98.04 | 65.12 | 97.79 |
| composition_unseen_only | 84.65 | 99.38 | 85.43 | 99.10 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 55.60 | 85.15 |
| per_image | 66.55 | 93.34 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 20 | 74.07 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| potato_early_blight | tomato_early_blight | 9 | 37.50 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| tomato_leaf_mold | tomato_early_blight | 8 | 33.33 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| tomato_septoria_leaf_spot | tomato_early_blight | 7 | 17.95 |
| grape_esca | grape_black_rot | 7 | 43.75 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 11 | 20.75 |
| apple | cucumber | 8 | 3.36 |
| apple | mango | 6 | 2.52 |
| paddy | corn | 6 | 3.80 |
| apple | tomato | 4 | 1.68 |
| cherry | apple | 4 | 28.57 |
| cotton | cucumber | 3 | 15.79 |
| plum | cherry | 3 | 30.00 |
| squash | cucumber | 3 | 100.00 |
| basil | cucumber | 3 | 27.27 |
| apple | sugarcane | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| tomato | pepper_bell | 3 | 1.03 |
| wheat | corn | 3 | 7.14 |
| peach | pepper_bell | 3 | 7.50 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 18 | 34.62 |
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | blast | 15 | 4.25 |
| healthy | leafblast | 10 | 2.83 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| mosaic | healthy | 9 | 6.92 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| healthy | caterpillar | 8 | 2.27 |
| leaf_mold | early_blight | 8 | 33.33 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| septoria_leaf_spot | early_blight | 7 | 17.07 |
| esca | black_rot | 7 | 43.75 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| bacterial_blight | brown_spot | 6 | 19.35 |

