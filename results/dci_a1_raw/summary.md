# dci_a1_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 56.52 | 81.59 | 71.35 | 91.50 |
| crop | 73.69 | 91.65 | 93.60 | 98.20 |
| disease | 58.01 | 81.70 | 72.60 | 91.30 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.18 | 33.69 | 7.11 | 28.45 |
| crop | 69.30 | 87.93 | 77.21 | 91.85 |
| disease | 20.40 | 37.52 | 10.84 | 26.45 |
| composition_unseen_only | 84.87 | 99.06 | 84.88 | 98.62 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.75 | 47.69 |
| per_image | 12.94 | 43.41 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 17 | 53.12 |
| paddy_healthy | paddy_blast | 12 | 37.50 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 12 | 52.17 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| soybean_healthy | soybean_caterpillar | 10 | 34.48 |
| cassava_healthy | cassava_brown_streak_disease | 10 | 35.71 |
| paddy_hispa | paddy_leafblast | 9 | 75.00 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 9 | 33.33 |
| cassava_brown_spot | cassava_bacterial_blight | 9 | 47.37 |
| cassava_mosaic | cassava_green_mottle | 8 | 6.45 |
| wheat_stripe_rust | wheat_rust | 8 | 80.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| paddy_blast | paddy_brown_spot | 6 | 27.27 |
| tomato_bacterial_spot | tomato_powdery_mildew | 6 | 22.22 |
| soybean_healthy | soybean_diabrotica_speciosa | 6 | 20.69 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| cherry | apple | 7 | 50.00 |
| apple | peach | 5 | 2.10 |
| potato | tomato | 5 | 9.43 |
| plum | apple | 3 | 30.00 |
| apple | guava | 2 | 0.84 |
| cabbage | brassica | 2 | 50.00 |
| basil | tomato | 2 | 18.18 |
| guava | pear | 2 | 15.38 |
| potato | apple | 2 | 3.77 |
| paddy | corn | 2 | 1.27 |
| cabbage | cauliflower | 2 | 50.00 |
| banana | corn | 2 | 6.25 |
| bean | soybean | 2 | 11.76 |
| sugarcane | paddy | 2 | 4.55 |
| cucumber | zucchini | 1 | 8.33 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 16 | 4.53 |
| healthy | blast | 12 | 3.40 |
| caterpillar | diabrotica_speciosa | 12 | 52.17 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| healthy | caterpillar | 10 | 2.83 |
| healthy | brown_streak_disease | 10 | 2.83 |
| hispa | leafblast | 9 | 75.00 |
| bacterial_spot | septoria_leaf_spot | 9 | 17.31 |
| brown_spot | bacterial_blight | 9 | 12.50 |
| mosaic | green_mottle | 8 | 6.15 |
| stripe_rust | rust | 8 | 80.00 |
| septoria_leaf_spot | verticulium_wilt | 7 | 17.07 |
| blast | brown_spot | 6 | 27.27 |
| marssonina_blotch | alternaria_leaf_spot | 6 | 42.86 |
| bacterial_spot | powdery_mildew | 6 | 11.54 |

