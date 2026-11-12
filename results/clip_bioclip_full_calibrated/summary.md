# clip_bioclip_full_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 41.18 | 69.02 | 62.50 | 84.45 |
| crop | 61.54 | 79.77 | 87.85 | 95.90 |
| disease | 44.32 | 70.92 | 67.15 | 89.10 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 63.10 | 97.03 | 60.64 | 95.23 |
| crop | 84.62 | 98.54 | 85.29 | 98.41 |
| disease | 61.60 | 95.99 | 58.29 | 94.34 |
| composition_unseen_only | 74.64 | 99.03 | 76.73 | 97.93 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 49.84 | 80.66 |
| per_image | 61.55 | 89.52 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| paddy_healthy | paddy_blast | 24 | 75.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| soybean_healthy | soybean_caterpillar | 12 | 41.38 |
| cassava_mosaic | cassava_healthy | 12 | 9.68 |
| paddy_hispa | paddy_blast | 11 | 91.67 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 10 | 25.64 |
| cassava_bacterial_blight | cotton_bacterial_blight | 10 | 43.48 |
| grape_leaf_blight | grape_black_rot | 9 | 100.00 |
| apple_marssonina_blotch | cucumber_downy_mildew | 7 | 50.00 |
| cassava_mosaic | cassava_green_mottle | 7 | 5.65 |
| tomato_bacterial_spot | tomato_septoria_leaf_spot | 7 | 25.93 |
| coffee_rust | coffee_healthy | 7 | 35.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 7 | 17.95 |
| wheat_stripe_rust | wheat_rust | 6 | 60.00 |
| paddy_healthy | paddy_leafblast | 5 | 15.62 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | cucumber | 13 | 5.46 |
| cassava | cotton | 9 | 3.35 |
| apple | mango | 8 | 3.36 |
| tomato | cucumber | 7 | 2.41 |
| paddy | corn | 6 | 3.80 |
| guava | mango | 6 | 46.15 |
| potato | tomato | 5 | 9.43 |
| palm | mango | 4 | 30.77 |
| banana | corn | 4 | 12.50 |
| tomato | pepper_bell | 4 | 1.37 |
| tomato | corn | 4 | 1.37 |
| banana | mango | 4 | 12.50 |
| cotton | cucumber | 3 | 15.79 |
| apple | corn | 3 | 1.26 |
| squash | cucumber | 3 | 100.00 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| healthy | blast | 22 | 6.23 |
| mosaic | healthy | 14 | 10.77 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| healthy | caterpillar | 12 | 3.40 |
| hispa | blast | 11 | 91.67 |
| septoria_leaf_spot | leaf_blight | 10 | 24.39 |
| healthy | bacterial_blight | 10 | 2.83 |
| leaf_blight | black_rot | 9 | 23.08 |
| rust | healthy | 8 | 9.52 |
| stripe_rust | rust | 8 | 80.00 |
| marssonina_blotch | downy_mildew | 7 | 50.00 |
| mosaic | green_mottle | 7 | 5.38 |
| bacterial_spot | septoria_leaf_spot | 7 | 13.46 |
| brown_spot | bacterial_blight | 7 | 9.72 |
| brown_streak_disease | healthy | 6 | 20.69 |

