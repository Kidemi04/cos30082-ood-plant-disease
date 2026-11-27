# llf_ens_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 47.39 | 76.01 | 66.00 | 89.85 |
| crop | 64.15 | 82.45 | 89.95 | 96.45 |
| disease | 50.31 | 82.33 | 70.65 | 92.75 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 72.37 | 99.02 | 73.62 | 98.69 |
| crop | 90.28 | 99.35 | 88.95 | 98.96 |
| disease | 71.37 | 99.06 | 71.96 | 98.83 |
| composition_unseen_only | 88.05 | 99.62 | 88.05 | 99.31 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 57.28 | 86.01 |
| per_image | 69.60 | 94.06 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 16 | 59.26 |
| paddy_healthy | paddy_leafblast | 15 | 46.88 |
| tomato_septoria_leaf_spot | tomato_early_blight | 13 | 33.33 |
| tomato_late_blight | tomato_early_blight | 13 | 44.83 |
| paddy_healthy | paddy_blast | 11 | 34.38 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| potato_early_blight | tomato_early_blight | 10 | 41.67 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 10 | 43.48 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| tomato_healthy | tomato_early_blight | 8 | 25.00 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |
| tomato_target_spot | tomato_early_blight | 7 | 46.67 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| paddy_hispa | paddy_blast | 6 | 50.00 |
| apple_marssonina_blotch | cucumber_downy_mildew | 6 | 42.86 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 12 | 22.64 |
| paddy | corn | 8 | 5.06 |
| apple | cucumber | 8 | 3.36 |
| wheat | sugarcane | 7 | 16.67 |
| apple | tomato | 6 | 2.52 |
| apple | mango | 5 | 2.10 |
| peach | pepper_bell | 5 | 12.50 |
| apple | sugarcane | 4 | 1.68 |
| basil | tomato | 4 | 36.36 |
| blackgram | cucumber | 4 | 26.67 |
| grape | cucumber | 4 | 9.76 |
| banana | corn | 4 | 12.50 |
| broccoli | cucumber | 3 | 60.00 |
| cotton | cucumber | 3 | 15.79 |
| plum | cherry | 3 | 30.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 15 | 4.25 |
| septoria_leaf_spot | early_blight | 13 | 31.71 |
| late_blight | early_blight | 13 | 26.00 |
| bacterial_spot | early_blight | 11 | 21.15 |
| healthy | blast | 11 | 3.12 |
| healthy | caterpillar | 10 | 2.83 |
| caterpillar | diabrotica_speciosa | 10 | 43.48 |
| leaf_mold | early_blight | 9 | 37.50 |
| stripe_rust | rust | 9 | 90.00 |
| mosaic | green_mottle | 7 | 5.38 |
| target_spot | early_blight | 7 | 33.33 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| mosaic | healthy | 7 | 5.38 |
| hispa | blast | 6 | 50.00 |
| marssonina_blotch | downy_mildew | 6 | 42.86 |

