# cpn_p2_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 55.76 | 80.62 | 71.85 | 88.85 |
| crop | 72.44 | 87.85 | 93.60 | 97.70 |
| disease | 57.57 | 79.49 | 73.15 | 87.55 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 7.18 | 38.24 | 6.91 | 43.51 |
| crop | 69.30 | 87.24 | 76.86 | 90.61 |
| disease | 19.05 | 36.35 | 10.22 | 25.76 |
| composition_unseen_only | 80.93 | 97.94 | 83.49 | 96.82 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 12.72 | 51.87 |
| per_image | 12.60 | 58.41 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 14 | 43.75 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| paddy_healthy | paddy_leafblast | 13 | 40.62 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| wheat_stripe_rust | wheat_rust | 9 | 90.00 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 8 | 29.63 |
| soybean_healthy | soybean_caterpillar | 7 | 24.14 |
| cassava_mosaic | cassava_healthy | 7 | 5.65 |
| tomato_blight_leaf | tomato_brown_spot | 6 | 85.71 |
| paddy_blast | paddy_brown_spot | 5 | 22.73 |
| apple_marssonina_blotch | apple_alternaria_leaf_spot | 5 | 35.71 |
| apple_healthy | apple_marssonina_blotch | 5 | 10.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| potato | tomato | 6 | 11.32 |
| cherry | apple | 6 | 42.86 |
| cabbage | cauliflower | 3 | 75.00 |
| plum | apple | 3 | 30.00 |
| apple | peach | 3 | 1.26 |
| paddy | wheat | 3 | 1.90 |
| cotton | squash | 2 | 10.53 |
| apple | guava | 2 | 0.84 |
| eggplant | tomato | 2 | 40.00 |
| basil | tomato | 2 | 18.18 |
| guava | pear | 2 | 15.38 |
| tomato | pepper_bell | 2 | 0.69 |
| potato | apple | 2 | 3.77 |
| celery | zucchini | 2 | 100.00 |
| banana | corn | 2 | 6.25 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| healthy | leafblast | 13 | 3.68 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| hispa | blast | 8 | 66.67 |
| bacterial_spot | septoria_leaf_spot | 8 | 15.38 |
| healthy | caterpillar | 7 | 1.98 |
| mosaic | healthy | 7 | 5.38 |
| marssonina_blotch | alternaria_leaf_spot | 6 | 42.86 |
| blight_leaf | brown_spot | 6 | 85.71 |
| blast | brown_spot | 5 | 22.73 |
| leaf_blight | northern_leaf_blight | 5 | 12.82 |

