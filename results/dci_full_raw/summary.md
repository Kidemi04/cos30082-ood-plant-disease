# dci_full_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 54.93 | 81.94 | 71.10 | 91.15 |
| crop | 71.93 | 91.73 | 93.00 | 98.00 |
| disease | 55.10 | 83.34 | 71.70 | 90.35 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 8.47 | 26.54 | 6.77 | 23.20 |
| crop | 67.63 | 83.96 | 75.97 | 89.78 |
| disease | 18.53 | 30.86 | 8.22 | 19.20 |
| composition_unseen_only | 80.05 | 98.37 | 77.42 | 97.79 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 14.67 | 40.09 |
| per_image | 12.36 | 36.99 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 16 | 50.00 |
| paddy_healthy | paddy_blast | 14 | 43.75 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 13 | 33.33 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 13 | 48.15 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| cassava_mosaic | cassava_green_mottle | 11 | 8.87 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 10 | 25.64 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |
| cassava_brown_spot | cassava_bacterial_blight | 5 | 26.32 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |
| tomato_late_blight | pepper_bell_blossom_end_rot | 4 | 13.79 |
| paddy_brown_spot | paddy_blast | 4 | 25.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| cherry | apple | 5 | 35.71 |
| apple | guava | 4 | 1.68 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| banana | corn | 3 | 9.38 |
| apple | rose | 2 | 0.84 |
| cabbage | cauliflower | 2 | 50.00 |
| apple | peach | 2 | 0.84 |
| maple | cotton | 2 | 100.00 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| apple | potato | 2 | 0.84 |
| sugarcane | paddy | 2 | 4.55 |
| bean | soybean | 2 | 11.76 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 14 | 3.97 |
| healthy | blast | 14 | 3.97 |
| septoria_leaf_spot | leaf_blight | 13 | 31.71 |
| healthy | caterpillar | 12 | 3.40 |
| bacterial_spot | septoria_leaf_spot | 12 | 23.08 |
| caterpillar | diabrotica_speciosa | 11 | 47.83 |
| septoria_leaf_spot | verticulium_wilt | 10 | 24.39 |
| mosaic | green_mottle | 9 | 6.92 |
| stripe_rust | rust | 9 | 90.00 |
| hispa | leafblast | 8 | 66.67 |
| healthy | brown_streak_disease | 6 | 1.70 |
| marssonina_blotch | alternaria_leaf_spot | 5 | 35.71 |
| brown_spot | bacterial_blight | 5 | 6.94 |
| blight_leaf | brown_spot | 5 | 71.43 |
| hispa | blast | 4 | 33.33 |

