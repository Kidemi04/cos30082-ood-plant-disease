# clip_bioclip_heads_strong_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 57.74 | 83.60 | 71.05 | 91.50 |
| crop | 75.65 | 90.97 | 93.60 | 97.90 |
| disease | 61.07 | 88.30 | 72.40 | 92.15 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 3.45 | 26.87 | 3.66 | 26.52 |
| crop | 67.74 | 87.78 | 74.31 | 90.68 |
| disease | 17.68 | 34.55 | 8.70 | 26.86 |
| composition_unseen_only | 82.55 | 99.53 | 81.77 | 99.17 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 6.51 | 40.66 |
| per_image | 6.96 | 41.12 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_leafblast | 16 | 50.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 16 | 69.57 |
| paddy_healthy | paddy_blast | 15 | 46.88 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 11 | 28.21 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 11 | 40.74 |
| soybean_healthy | soybean_caterpillar | 8 | 27.59 |
| soybean_healthy | soybean_diabrotica_speciosa | 8 | 27.59 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 8 | 20.51 |
| wheat_stripe_rust | wheat_rust | 7 | 70.00 |
| paddy_hispa | paddy_blast | 6 | 50.00 |
| cassava_mosaic | cassava_green_mottle | 6 | 4.84 |
| paddy_hispa | paddy_leafblast | 6 | 50.00 |
| cassava_green_mottle | cassava_brown_streak_disease | 6 | 30.00 |
| cassava_brown_spot | cassava_bacterial_blight | 6 | 31.58 |
| cassava_healthy | cassava_brown_streak_disease | 6 | 21.43 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| cherry | apple | 5 | 35.71 |
| apple | peach | 4 | 1.68 |
| potato | tomato | 4 | 7.55 |
| paddy | corn | 3 | 1.90 |
| plum | apple | 3 | 30.00 |
| bean | soybean | 3 | 17.65 |
| soybean | blackgram | 2 | 2.78 |
| apple | rose | 2 | 0.84 |
| squash | zucchini | 2 | 66.67 |
| grape | strawberry | 2 | 4.88 |
| cabbage | brassica | 2 | 50.00 |
| carrot | celery | 2 | 66.67 |
| tobacco | tomato | 2 | 50.00 |
| apple | guava | 2 | 0.84 |
| potato | apple | 2 | 3.77 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | leafblast | 16 | 4.53 |
| caterpillar | diabrotica_speciosa | 16 | 69.57 |
| healthy | blast | 15 | 4.25 |
| septoria_leaf_spot | leaf_blight | 11 | 26.83 |
| bacterial_spot | septoria_leaf_spot | 11 | 21.15 |
| marssonina_blotch | downy_mildew | 9 | 64.29 |
| healthy | caterpillar | 8 | 2.27 |
| healthy | diabrotica_speciosa | 8 | 2.27 |
| septoria_leaf_spot | verticulium_wilt | 8 | 19.51 |
| bacterial_blight | brown_spot | 7 | 22.58 |
| stripe_rust | rust | 7 | 70.00 |
| hispa | blast | 6 | 50.00 |
| mosaic | green_mottle | 6 | 4.62 |
| hispa | leafblast | 6 | 50.00 |
| green_mottle | brown_streak_disease | 6 | 30.00 |

