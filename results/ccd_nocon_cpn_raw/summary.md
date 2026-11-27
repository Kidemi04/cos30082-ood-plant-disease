# ccd_nocon_cpn_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 57.29 | 79.92 | 70.20 | 90.60 |
| crop | 74.04 | 92.55 | 93.85 | 98.20 |
| disease | 56.06 | 82.36 | 70.50 | 89.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 6.80 | 31.30 | 6.70 | 25.90 |
| crop | 70.76 | 87.00 | 78.31 | 90.68 |
| disease | 15.31 | 30.89 | 6.42 | 19.68 |
| composition_unseen_only | 82.69 | 98.62 | 80.87 | 97.44 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.15 | 44.98 |
| per_image | 12.23 | 40.28 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| paddy_healthy | paddy_blast | 13 | 40.62 |
| paddy_healthy | paddy_leafblast | 12 | 37.50 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 11 | 28.21 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| cassava_healthy | cassava_brown_streak_disease | 7 | 25.00 |
| cassava_mosaic | cassava_green_mottle | 6 | 4.84 |
| tomato_bacterial_spot | tomato_late_blight | 6 | 22.22 |
| paddy_blast | paddy_brown_spot | 5 | 22.73 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 5 | 9.43 |
| cherry | apple | 5 | 35.71 |
| apple | guava | 3 | 1.26 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| cotton | squash | 2 | 10.53 |
| cabbage | cauliflower | 2 | 50.00 |
| apple | peach | 2 | 0.84 |
| tomato | pepper_bell | 2 | 0.69 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| paddy | wheat | 2 | 1.27 |
| banana | corn | 2 | 6.25 |
| bean | soybean | 2 | 11.76 |
| sugarcane | paddy | 2 | 4.55 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| healthy | leafblast | 12 | 3.40 |
| healthy | blast | 12 | 3.40 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| septoria_leaf_spot | verticulium_wilt | 11 | 26.83 |
| hispa | leafblast | 9 | 75.00 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| stripe_rust | rust | 9 | 90.00 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| healthy | caterpillar | 7 | 1.98 |
| healthy | brown_streak_disease | 7 | 1.98 |
| mosaic | healthy | 7 | 5.38 |
| mosaic | green_mottle | 6 | 4.62 |
| bacterial_spot | late_blight | 6 | 11.54 |

