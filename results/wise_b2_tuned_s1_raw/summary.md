# wise_b2_tuned_s1_raw

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 59.50 | 88.25 | 72.95 | 93.60 |
| crop | 80.77 | 96.93 | 95.25 | 99.10 |
| disease | 62.11 | 88.08 | 74.10 | 92.85 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 12.08 | 56.15 | 8.63 | 50.90 |
| crop | 84.48 | 96.77 | 88.05 | 97.51 |
| disease | 20.09 | 41.80 | 8.56 | 31.28 |
| composition_unseen_only | 91.46 | 99.73 | 92.82 | 99.59 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 20.08 | 68.63 |
| per_image | 15.44 | 65.94 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| soybean_caterpillar | soybean_diabrotica_speciosa | 18 | 78.26 |
| paddy_healthy | paddy_leafblast | 17 | 53.12 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 16 | 59.26 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| cassava_brown_spot | cassava_bacterial_blight | 12 | 63.16 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 9 | 23.08 |
| paddy_healthy | paddy_blast | 8 | 25.00 |
| paddy_hispa | paddy_leafblast | 7 | 58.33 |
| cassava_green_mottle | cassava_brown_streak_disease | 6 | 30.00 |
| cassava_mosaic | cassava_healthy | 6 | 4.84 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| tomato_leaf_mold | tomato_late_blight | 5 | 20.83 |
| tomato_healthy | tomato_leaf_blight | 5 | 15.62 |
| cassava_healthy | cassava_bacterial_blight | 5 | 17.86 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 5 | 12.82 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | peach | 4 | 1.68 |
| paddy | wheat | 3 | 1.90 |
| cotton | squash | 2 | 10.53 |
| apple | broccoli | 2 | 0.84 |
| cabbage | brassica | 2 | 50.00 |
| potato | tomato | 2 | 3.77 |
| pepper_bell | chilli | 2 | 20.00 |
| paddy | palm | 2 | 1.27 |
| plum | peach | 2 | 20.00 |
| apple | rose | 2 | 0.84 |
| eggplant | tomato | 2 | 40.00 |
| broccoli | cabbage | 2 | 40.00 |
| zucchini | cucumber | 2 | 50.00 |
| wheat | paddy | 2 | 4.76 |
| broccoli | cauliflower | 2 | 40.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| caterpillar | diabrotica_speciosa | 18 | 78.26 |
| healthy | leafblast | 17 | 4.82 |
| bacterial_spot | septoria_leaf_spot | 16 | 30.77 |
| healthy | caterpillar | 12 | 3.40 |
| brown_spot | bacterial_blight | 12 | 16.67 |
| septoria_leaf_spot | leaf_blight | 9 | 21.95 |
| hispa | leafblast | 7 | 58.33 |
| mosaic | healthy | 7 | 5.38 |
| green_mottle | brown_streak_disease | 6 | 30.00 |
| stripe_rust | rust | 6 | 60.00 |
| leaf_mold | late_blight | 5 | 20.83 |
| marssonina_blotch | alternaria_leaf_spot | 5 | 35.71 |
| healthy | blast | 5 | 1.42 |
| healthy | leaf_blight | 5 | 1.42 |
| healthy | bacterial_blight | 5 | 1.42 |

