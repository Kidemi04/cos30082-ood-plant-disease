# wise_p6_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 49.22 | 73.19 | 67.40 | 87.40 |
| crop | 71.84 | 84.12 | 91.45 | 96.65 |
| disease | 46.42 | 72.19 | 68.55 | 89.95 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.91 | 47.32 | 7.11 | 50.62 |
| crop | 58.47 | 83.44 | 67.20 | 87.09 |
| disease | 22.00 | 64.72 | 13.88 | 63.26 |
| composition_unseen_only | 68.73 | 96.47 | 73.34 | 96.89 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 13.62 | 57.48 |
| per_image | 12.87 | 64.11 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 19 | 82.61 |
| paddy_healthy | paddy_blast | 15 | 46.88 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 12 | 30.77 |
| paddy_healthy | paddy_leafblast | 12 | 37.50 |
| cassava_mosaic | cassava_healthy | 11 | 8.87 |
| apple_healthy | apple_marssonina_blotch | 9 | 18.00 |
| coffee_rust | coffee_healthy | 9 | 45.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| paddy_hispa | paddy_leafblast | 7 | 58.33 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| cassava_brown_spot | cassava_bacterial_blight | 5 | 26.32 |
| tomato_blight_leaf | tomato_brown_spot | 5 | 71.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| apple | cotton | 4 | 1.68 |
| paddy | corn | 3 | 1.90 |
| plum | apple | 3 | 30.00 |
| basil | tomato | 3 | 27.27 |
| cherry | peach | 3 | 21.43 |
| potato | apple | 3 | 5.66 |
| paddy | wheat | 3 | 1.90 |
| cherry | apple | 3 | 21.43 |
| eggplant | tomato | 3 | 60.00 |
| cotton | squash | 2 | 10.53 |
| basil | mango | 2 | 18.18 |
| palm | mango | 2 | 15.38 |
| apple | peach | 2 | 0.84 |
| apple | guava | 2 | 0.84 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 19 | 82.61 |
| healthy | blast | 15 | 4.25 |
| septoria_leaf_spot | leaf_blight | 12 | 29.27 |
| mosaic | healthy | 12 | 9.23 |
| healthy | leafblast | 11 | 3.12 |
| rust | healthy | 11 | 13.10 |
| healthy | marssonina_blotch | 9 | 2.55 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| hispa | leafblast | 7 | 58.33 |
| healthy | caterpillar | 7 | 1.98 |
| downy_mildew | powdery_mildew | 7 | 31.82 |
| stripe_rust | rust | 7 | 70.00 |
| brown_spot | bacterial_blight | 6 | 8.33 |

