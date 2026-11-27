# wise_openai_tuned_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 59.68 | 82.91 | 74.50 | 92.70 |
| crop | 76.45 | 93.07 | 93.95 | 98.15 |
| disease | 59.32 | 85.16 | 75.70 | 93.65 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.01 | 35.45 | 6.42 | 31.01 |
| crop | 70.84 | 86.61 | 77.97 | 91.99 |
| disease | 20.95 | 41.77 | 11.95 | 30.73 |
| composition_unseen_only | 89.21 | 99.67 | 90.06 | 99.38 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.54 | 49.67 |
| per_image | 11.83 | 46.47 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 11 | 40.74 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| cassava_healthy | cassava_bacterial_blight | 8 | 28.57 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| apple_marssonina_blotch | apple_alternaria_leaf_spot | 6 | 42.86 |
| paddy_blast | paddy_brown_spot | 5 | 22.73 |
| cassava_green_mottle | cassava_mosaic | 5 | 25.00 |
| cassava_mosaic | cassava_green_mottle | 5 | 4.03 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 6 | 2.52 |
| potato | tomato | 6 | 11.32 |
| cherry | apple | 6 | 42.86 |
| basil | tomato | 5 | 45.45 |
| plum | apple | 3 | 30.00 |
| bean | soybean | 3 | 17.65 |
| cabbage | brassica | 2 | 50.00 |
| raspberry | tomato | 2 | 18.18 |
| guava | cassava | 2 | 15.38 |
| brassica | cabbage | 2 | 100.00 |
| cucumber | tomato | 2 | 16.67 |
| eggplant | tomato | 2 | 40.00 |
| paddy | wheat | 2 | 1.27 |
| apple | guava | 2 | 0.84 |
| sugarcane | wheat | 2 | 4.55 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| healthy | leafblast | 11 | 3.12 |
| bacterial_spot | septoria_leaf_spot | 11 | 21.15 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| healthy | caterpillar | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | bacterial_blight | 8 | 2.27 |
| hispa | blast | 8 | 66.67 |
| mosaic | healthy | 8 | 6.15 |
| marssonina_blotch | alternaria_leaf_spot | 6 | 42.86 |
| brown_spot | bacterial_blight | 6 | 8.33 |
| blast | brown_spot | 5 | 22.73 |
| green_mottle | mosaic | 5 | 25.00 |

