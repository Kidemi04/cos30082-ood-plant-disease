# dci_a5_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 56.17 | 79.99 | 72.50 | 91.10 |
| crop | 71.42 | 91.00 | 93.35 | 98.00 |
| disease | 54.08 | 81.49 | 73.25 | 90.80 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.65 | 31.60 | 7.25 | 30.52 |
| crop | 67.83 | 84.53 | 75.97 | 89.57 |
| disease | 20.70 | 33.40 | 10.64 | 24.03 |
| composition_unseen_only | 77.80 | 96.74 | 78.87 | 96.06 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 13.47 | 45.30 |
| per_image | 13.18 | 45.73 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 17 | 73.91 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| paddy_healthy | paddy_leafblast | 11 | 34.38 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| paddy_hispa | paddy_blast | 7 | 58.33 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| cassava_mosaic | cassava_green_mottle | 6 | 4.84 |
| paddy_hispa | paddy_leafblast | 5 | 41.67 |
| cassava_healthy | cassava_brown_streak_disease | 5 | 17.86 |
| peanut_healthy | peanut_early_leaf_spot | 5 | 100.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 7 | 13.21 |
| apple | peach | 4 | 1.68 |
| cherry | apple | 4 | 28.57 |
| apple | rose | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| apple | tomato | 3 | 1.26 |
| banana | corn | 3 | 9.38 |
| cabbage | cauliflower | 2 | 50.00 |
| wheat | corn | 2 | 4.76 |
| basil | tomato | 2 | 18.18 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| eggplant | tomato | 2 | 40.00 |
| bean | soybean | 2 | 11.76 |
| sugarcane | paddy | 2 | 4.55 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 17 | 4.82 |
| caterpillar | diabrotica_speciosa | 15 | 65.22 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| healthy | leafblast | 11 | 3.12 |
| bacterial_spot | septoria_leaf_spot | 10 | 19.23 |
| healthy | caterpillar | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | blast | 7 | 58.33 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| mosaic | healthy | 7 | 5.38 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| mosaic | green_mottle | 6 | 4.62 |
| hispa | leafblast | 5 | 41.67 |
| marssonina_blotch | downy_mildew | 5 | 35.71 |
| healthy | early_leaf_spot | 5 | 1.42 |

