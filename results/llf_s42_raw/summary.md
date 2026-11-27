# llf_s42_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 57.65 | 81.52 | 72.80 | 92.45 |
| crop | 73.77 | 90.28 | 93.70 | 97.85 |
| disease | 58.90 | 82.83 | 73.75 | 92.05 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 12.51 | 44.96 | 10.98 | 39.02 |
| crop | 73.54 | 90.33 | 78.59 | 92.33 |
| disease | 23.43 | 39.42 | 13.47 | 30.80 |
| composition_unseen_only | 87.85 | 99.28 | 86.19 | 98.69 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 20.56 | 57.95 |
| per_image | 19.08 | 54.88 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 13 | 40.62 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 10 | 43.48 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |
| tomato_leaf_blight | tomato_verticulium_wilt | 5 | 50.00 |
| tomato_yellow_leaf_curl_virus | tomato_mosaic_virus | 5 | 11.36 |
| cassava_brown_streak_disease | cassava_green_mottle | 5 | 17.24 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | guava | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| cabbage | brassica | 2 | 50.00 |
| apple | bean | 2 | 0.84 |
| basil | tomato | 2 | 18.18 |
| potato | apple | 2 | 3.77 |
| lettuce | tomato | 2 | 66.67 |
| cabbage | cauliflower | 2 | 50.00 |
| grape | mango | 2 | 4.88 |
| bean | soybean | 2 | 11.76 |
| basil | corn | 2 | 18.18 |
| cucumber | zucchini | 1 | 8.33 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 13 | 3.68 |
| healthy | blast | 13 | 3.68 |
| caterpillar | diabrotica_speciosa | 10 | 43.48 |
| healthy | caterpillar | 9 | 2.55 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| hispa | blast | 8 | 66.67 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| mosaic | healthy | 7 | 5.38 |
| stripe_rust | rust | 7 | 70.00 |
| healthy | diabrotica_speciosa | 6 | 1.70 |
| healthy | brown_streak_disease | 5 | 1.42 |
| leaf_blight | verticulium_wilt | 5 | 12.82 |
| yellow_leaf_curl_virus | mosaic_virus | 5 | 11.36 |
| bacterial_blight | brown_spot | 5 | 16.13 |

