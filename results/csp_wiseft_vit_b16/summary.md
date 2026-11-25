# csp_wiseft_vit_b16

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 52.81 | 76.30 | 69.50 | 90.30 |
| crop | 71.07 | 91.17 | 91.75 | 97.85 |
| disease | 56.14 | 80.16 | 71.40 | 92.65 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 19.51 | 64.90 | 18.92 | 63.95 |
| crop | 70.18 | 96.10 | 75.90 | 96.34 |
| disease | 31.02 | 70.91 | 23.34 | 67.68 |
| composition_unseen_only | 85.11 | 99.94 | 83.77 | 99.79 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 28.50 | 70.14 |
| per_image | 29.75 | 74.87 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| paddy_healthy | paddy_leafblast | 13 | 40.62 |
| soybean_healthy | soybean_diabrotica_speciosa | 11 | 37.93 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| paddy_healthy | paddy_brown_spot | 9 | 28.12 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| cassava_mosaic | cassava_healthy | 9 | 7.26 |
| cassava_healthy | cassava_bacterial_blight | 8 | 28.57 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 7 | 17.95 |
| paddy_hispa | paddy_leafblast | 5 | 41.67 |
| paddy_blast | paddy_brown_spot | 5 | 22.73 |
| paddy_hispa | paddy_blast | 4 | 33.33 |
| tomato_bacterial_spot | tomato_late_blight | 4 | 14.81 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| cherry | apple | 5 | 35.71 |
| apple | guava | 4 | 1.68 |
| guava | cassava | 4 | 30.77 |
| apple | peach | 4 | 1.68 |
| paddy | corn | 3 | 1.90 |
| mango | cashew | 3 | 5.66 |
| paddy | wheat | 3 | 1.90 |
| potato | tomato | 3 | 5.66 |
| paddy | sugarcane | 3 | 1.90 |
| soybean | cotton | 2 | 2.78 |
| eggplant | cotton | 2 | 40.00 |
| cabbage | brassica | 2 | 50.00 |
| tobacco | tomato | 2 | 50.00 |
| apple | cotton | 2 | 0.84 |
| raspberry | tomato | 2 | 18.18 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 17 | 73.91 |
| healthy | leafblast | 13 | 3.68 |
| healthy | diabrotica_speciosa | 12 | 3.40 |
| brown_spot | bacterial_blight | 10 | 13.89 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| healthy | brown_spot | 9 | 2.55 |
| mosaic | healthy | 9 | 6.92 |
| healthy | bacterial_blight | 8 | 2.27 |
| stripe_rust | rust | 8 | 80.00 |
| healthy | caterpillar | 7 | 1.98 |
| septoria_leaf_spot | leaf_blight | 7 | 17.07 |
| blast | brown_spot | 6 | 27.27 |
| hispa | leafblast | 5 | 41.67 |
| spot | leaf_blight | 5 | 23.81 |
| bacterial_spot | late_blight | 5 | 9.62 |

