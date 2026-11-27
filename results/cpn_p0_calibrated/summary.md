# cpn_p0_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 41.74 | 73.95 | 59.35 | 87.60 |
| crop | 60.75 | 80.20 | 85.65 | 95.95 |
| disease | 48.77 | 81.17 | 68.20 | 91.20 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 77.94 | 98.60 | 78.11 | 98.00 |
| crop | 87.39 | 98.77 | 86.74 | 98.34 |
| disease | 77.51 | 98.86 | 76.93 | 98.34 |
| composition_unseen_only | 84.25 | 99.20 | 85.43 | 98.83 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 54.37 | 84.51 |
| per_image | 67.45 | 92.51 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 20 | 74.07 |
| tomato_septoria_leaf_spot | tomato_early_blight | 18 | 46.15 |
| apple_rust | sugarcane_rust | 16 | 45.71 |
| paddy_healthy | paddy_blast | 16 | 50.00 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| peach_bacterial_spot | pepper_bell_bacterial_spot | 12 | 63.16 |
| tomato_healthy | tomato_early_blight | 11 | 34.38 |
| tomato_late_blight | tomato_early_blight | 11 | 37.93 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 10 | 25.64 |
| soybean_healthy | soybean_caterpillar | 9 | 31.03 |
| corn_northern_leaf_blight | corn_brown_spot | 9 | 50.00 |
| tomato_leaf_mold | tomato_early_blight | 9 | 37.50 |
| paddy_healthy | paddy_leafblast | 8 | 25.00 |
| coffee_rust | sugarcane_rust | 7 | 35.00 |
| apple_marssonina_blotch | cucumber_downy_mildew | 7 | 50.00 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| apple | sugarcane | 15 | 6.30 |
| apple | cucumber | 14 | 5.88 |
| peach | pepper_bell | 12 | 30.00 |
| paddy | corn | 10 | 6.33 |
| wheat | sugarcane | 10 | 23.81 |
| potato | tomato | 9 | 16.98 |
| pumpkin | cucumber | 7 | 43.75 |
| tomato | corn | 7 | 2.41 |
| apple | corn | 7 | 2.94 |
| coffee | sugarcane | 6 | 15.38 |
| cassava | cotton | 6 | 2.23 |
| blackgram | cucumber | 5 | 33.33 |
| banana | corn | 5 | 15.62 |
| cotton | cucumber | 4 | 21.05 |
| apple | grape | 4 | 1.68 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| septoria_leaf_spot | early_blight | 18 | 43.90 |
| bacterial_spot | early_blight | 18 | 34.62 |
| healthy | blast | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| healthy | brown_spot | 10 | 2.83 |
| healthy | early_blight | 10 | 2.83 |
| septoria_leaf_spot | verticulium_wilt | 10 | 24.39 |
| late_blight | early_blight | 10 | 20.00 |
| healthy | caterpillar | 9 | 2.55 |
| northern_leaf_blight | brown_spot | 9 | 50.00 |
| leaf_mold | early_blight | 9 | 37.50 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | leafblast | 8 | 2.27 |
| marssonina_blotch | downy_mildew | 7 | 50.00 |
| mosaic | green_mottle | 7 | 5.38 |

