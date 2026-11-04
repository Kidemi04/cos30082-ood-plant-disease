# convnext_tiny_independent

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 50.18 | 73.28 | 67.25 | 89.05 |
| crop | 66.83 | 83.94 | 90.55 | 96.15 |
| disease | 52.91 | 75.89 | 68.00 | 88.40 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 9.36 | 36.87 | 5.73 | 31.08 |
| crop | 67.38 | 81.27 | 71.06 | 84.81 |
| disease | 14.64 | 30.31 | 6.28 | 20.58 |
| composition_unseen_only | 76.26 | 97.54 | 74.93 | 95.58 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 15.78 | 49.06 |
| per_image | 10.56 | 46.08 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 17 | 53.12 |
| cassava_mosaic | cassava_green_mottle | 16 | 12.90 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| paddy_healthy | paddy_leafblast | 10 | 31.25 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| cassava_healthy | cassava_green_mottle | 9 | 32.14 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| cassava_brown_spot | cassava_bacterial_blight | 8 | 42.11 |
| cassava_mosaic | cassava_healthy | 8 | 6.45 |
| tomato_bacterial_spot | tomato_leaf_mold | 7 | 25.93 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| soybean_healthy | soybean_diabrotica_speciosa | 7 | 24.14 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| cherry | apple | 6 | 42.86 |
| apple | peach | 5 | 2.10 |
| corn | sugarcane | 5 | 2.43 |
| paddy | palm | 4 | 2.53 |
| paddy | sugarcane | 4 | 2.53 |
| apple | pear | 4 | 1.68 |
| apple | rose | 3 | 1.26 |
| paddy | wheat | 3 | 1.90 |
| apricot | peach | 3 | 75.00 |
| peach | raspberry | 2 | 5.00 |
| paddy | corn | 2 | 1.27 |
| basil | tomato | 2 | 18.18 |
| potato | apple | 2 | 3.77 |
| apple | cassava | 2 | 0.84 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| mosaic | green_mottle | 16 | 12.31 |
| healthy | blast | 15 | 4.25 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| caterpillar | diabrotica_speciosa | 12 | 52.17 |
| mosaic | healthy | 11 | 8.46 |
| healthy | leafblast | 10 | 2.83 |
| healthy | green_mottle | 9 | 2.55 |
| healthy | caterpillar | 8 | 2.27 |
| hispa | blast | 8 | 66.67 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| brown_spot | bacterial_blight | 8 | 11.11 |
| bacterial_spot | leaf_mold | 7 | 13.46 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| healthy | diabrotica_speciosa | 7 | 1.98 |
| stripe_rust | rust | 7 | 70.00 |

