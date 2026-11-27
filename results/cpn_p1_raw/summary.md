# cpn_p1_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 55.85 | 78.90 | 72.10 | 87.90 |
| crop | 71.40 | 90.93 | 93.40 | 97.55 |
| disease | 56.02 | 79.26 | 73.55 | 87.15 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.25 | 45.38 | 5.87 | 50.14 |
| crop | 69.48 | 86.67 | 77.62 | 90.61 |
| disease | 19.54 | 39.49 | 9.53 | 33.08 |
| composition_unseen_only | 81.71 | 98.29 | 85.29 | 97.58 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.84 | 57.62 |
| per_image | 10.86 | 63.85 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 14 | 43.75 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| cassava_mosaic | cassava_green_mottle | 6 | 4.84 |
| apple_marssonina_blotch | apple_alternaria_leaf_spot | 5 | 35.71 |
| cassava_brown_spot | cassava_bacterial_blight | 5 | 26.32 |
| corn_leaf_blight | corn_northern_leaf_blight | 4 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| cherry | apple | 6 | 42.86 |
| potato | tomato | 5 | 9.43 |
| basil | tomato | 4 | 36.36 |
| paddy | corn | 3 | 1.90 |
| apple | guava | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| guava | pear | 3 | 23.08 |
| apple | rose | 2 | 0.84 |
| cabbage | cauliflower | 2 | 50.00 |
| apple | peach | 2 | 0.84 |
| wheat | paddy | 2 | 4.76 |
| tomato | pepper_bell | 2 | 0.69 |
| potato | apple | 2 | 3.77 |
| plum | peach | 2 | 20.00 |
| celery | zucchini | 2 | 100.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| healthy | leafblast | 13 | 3.68 |
| healthy | blast | 13 | 3.68 |
| healthy | caterpillar | 9 | 2.55 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 8 | 66.67 |
| bacterial_spot | septoria_leaf_spot | 7 | 13.46 |
| mosaic | healthy | 7 | 5.38 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| marssonina_blotch | alternaria_leaf_spot | 6 | 42.86 |
| brown_spot | bacterial_blight | 6 | 8.33 |
| mosaic | green_mottle | 6 | 4.62 |
| blast | brown_spot | 5 | 22.73 |

