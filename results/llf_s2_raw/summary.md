# llf_s2_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 58.21 | 82.44 | 72.90 | 91.70 |
| crop | 72.17 | 91.42 | 93.10 | 98.00 |
| disease | 57.71 | 84.17 | 73.80 | 91.65 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.57 | 35.66 | 6.56 | 33.91 |
| crop | 70.86 | 88.13 | 77.14 | 90.88 |
| disease | 21.05 | 39.23 | 10.29 | 29.90 |
| composition_unseen_only | 83.73 | 98.91 | 84.19 | 98.55 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 13.39 | 49.78 |
| per_image | 12.04 | 49.51 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 19 | 59.38 |
| soybean_healthy | soybean_caterpillar | 11 | 37.93 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 11 | 47.83 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 10 | 37.04 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| paddy_hispa | paddy_leafblast | 8 | 66.67 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| paddy_healthy | paddy_blast | 7 | 21.88 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| tomato_yellow_leaf_curl_virus | tomato_mosaic_virus | 5 | 11.36 |
| tomato_septoria_leaf_spot | tomato_curl | 5 | 12.82 |
| cassava_green_mottle | cassava_green_mite | 4 | 20.00 |
| cassava_green_mottle | cassava_brown_streak_disease | 4 | 20.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| apple | peach | 4 | 1.68 |
| cherry | apple | 4 | 28.57 |
| apple | guava | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| paddy | wheat | 3 | 1.90 |
| wheat | corn | 3 | 7.14 |
| sugarcane | paddy | 3 | 6.82 |
| cabbage | brassica | 2 | 50.00 |
| carrot | celery | 2 | 66.67 |
| raspberry | tomato | 2 | 18.18 |
| potato | apple | 2 | 3.77 |
| apple | tomato | 2 | 0.84 |
| lettuce | tomato | 2 | 66.67 |
| plum | peach | 2 | 20.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 19 | 5.38 |
| healthy | caterpillar | 11 | 3.12 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |
| healthy | blast | 7 | 1.98 |
| mosaic | healthy | 7 | 5.38 |
| mosaic | green_mottle | 7 | 5.38 |
| yellow_leaf_curl_virus | mosaic_virus | 5 | 11.36 |
| marssonina_blotch | alternaria_leaf_spot | 5 | 35.71 |
| septoria_leaf_spot | curl | 5 | 12.20 |
| green_mottle | green_mite | 4 | 20.00 |

