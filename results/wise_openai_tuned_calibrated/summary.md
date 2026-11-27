# wise_openai_tuned_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 48.92 | 72.54 | 67.10 | 89.15 |
| crop | 64.77 | 80.37 | 89.85 | 96.00 |
| disease | 49.88 | 79.70 | 71.70 | 92.65 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 74.16 | 99.28 | 69.13 | 98.90 |
| crop | 87.13 | 99.60 | 88.95 | 99.38 |
| disease | 74.23 | 99.28 | 68.44 | 98.90 |
| composition_unseen_only | 89.21 | 99.67 | 90.06 | 99.38 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 58.95 | 83.83 |
| per_image | 68.10 | 93.77 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_septoria_leaf_spot | tomato_early_blight | 22 | 56.41 |
| tomato_bacterial_spot | tomato_early_blight | 19 | 70.37 |
| paddy_healthy | paddy_blast | 17 | 53.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_late_blight | tomato_early_blight | 9 | 31.03 |
| cassava_healthy | cassava_bacterial_blight | 8 | 28.57 |
| tomato_leaf_mold | tomato_early_blight | 8 | 33.33 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 7 | 15.91 |
| tomato_leaf_blight | tomato_early_blight | 6 | 60.00 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 5 | 12.82 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy | corn | 7 | 4.43 |
| potato | tomato | 7 | 13.21 |
| apple | cucumber | 6 | 2.52 |
| basil | tomato | 6 | 54.55 |
| cotton | cucumber | 5 | 26.32 |
| pumpkin | cucumber | 5 | 31.25 |
| apple | tomato | 5 | 2.10 |
| wheat | sugarcane | 5 | 11.90 |
| apple | corn | 5 | 2.10 |
| soybean | cotton | 4 | 5.56 |
| apple | sugarcane | 4 | 1.68 |
| cherry | apple | 4 | 28.57 |
| banana | corn | 4 | 12.50 |
| plum | cherry | 3 | 30.00 |
| squash | cucumber | 3 | 100.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 22 | 53.66 |
| healthy | blast | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| bacterial_spot | early_blight | 14 | 26.92 |
| healthy | leafblast | 11 | 3.12 |
| healthy | bacterial_blight | 10 | 2.83 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| healthy | caterpillar | 9 | 2.55 |
| late_blight | early_blight | 9 | 18.00 |
| stripe_rust | rust | 9 | 90.00 |
| leaf_mold | early_blight | 8 | 33.33 |
| hispa | blast | 8 | 66.67 |
| mosaic | healthy | 8 | 6.15 |
| yellow_leaf_curl_virus | early_blight | 7 | 15.91 |
| brown_spot | bacterial_blight | 7 | 9.72 |

