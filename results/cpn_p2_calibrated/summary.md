# cpn_p2_calibrated

Candidates: all 333 compositions. Primary average: per-class mean accuracy (%).

## test_seen (2000 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 44.22 | 70.93 | 62.60 | 85.15 |
| crop | 63.60 | 81.47 | 88.80 | 95.90 |
| disease | 48.81 | 77.11 | 67.75 | 88.25 |

## test_unseen (1448 images)

| Level | Top-1 per-class | Top-5 per-class | Top-1 per-image | Top-5 per-image |
|---|---|---|---|---|
| composition | 71.02 | 97.42 | 70.99 | 96.27 |
| crop | 83.79 | 97.51 | 83.84 | 96.41 |
| disease | 69.64 | 97.40 | 69.48 | 96.20 |
| composition_unseen_only | 80.93 | 97.94 | 83.49 | 96.82 |

## Harmonic mean of seen and unseen composition accuracy

| | Top-1 | Top-5 |
|---|---|---|
| per_class | 54.51 | 82.09 |
| per_image | 66.53 | 90.37 |

## Most confused seen-test composition pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| tomato_bacterial_spot | tomato_early_blight | 21 | 77.78 |
| paddy_healthy | paddy_blast | 14 | 43.75 |
| soybean_caterpillar | soybean_diabrotica_speciosa | 14 | 60.87 |
| paddy_healthy | paddy_leafblast | 13 | 40.62 |
| tomato_late_blight | tomato_early_blight | 13 | 44.83 |
| tomato_septoria_leaf_spot | tomato_early_blight | 11 | 28.21 |
| corn_leaf_blight | corn_brown_spot | 11 | 68.75 |
| tomato_target_spot | tomato_early_blight | 10 | 66.67 |
| tomato_leaf_mold | tomato_early_blight | 10 | 41.67 |
| corn_rust | sugarcane_rust | 10 | 100.00 |
| tomato_septoria_leaf_spot | tomato_verticulium_wilt | 9 | 23.08 |
| tomato_yellow_leaf_curl_virus | tomato_early_blight | 9 | 20.45 |
| soybean_healthy | soybean_diabrotica_speciosa | 9 | 31.03 |
| paddy_hispa | paddy_blast | 8 | 66.67 |
| tomato_septoria_leaf_spot | tomato_leaf_blight | 8 | 20.51 |

## Most confused seen-test crop pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| wheat | sugarcane | 12 | 28.57 |
| apple | cucumber | 11 | 4.62 |
| potato | tomato | 8 | 15.09 |
| corn | sugarcane | 7 | 3.40 |
| paddy | corn | 6 | 3.80 |
| cotton | cucumber | 5 | 26.32 |
| cherry | apple | 5 | 35.71 |
| peach | pepper_bell | 5 | 12.50 |
| apple | sugarcane | 4 | 1.68 |
| apple | mango | 4 | 1.68 |
| apple | tomato | 4 | 1.68 |
| plum | apple | 3 | 30.00 |
| apple | grape | 3 | 1.26 |
| basil | tomato | 3 | 27.27 |
| soybean | sugarcane | 3 | 4.17 |

## Most confused seen-test disease pairs

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| bacterial_spot | early_blight | 20 | 38.46 |
| late_blight | early_blight | 14 | 28.00 |
| healthy | blast | 14 | 3.97 |
| caterpillar | diabrotica_speciosa | 14 | 60.87 |
| healthy | leafblast | 12 | 3.40 |
| septoria_leaf_spot | early_blight | 11 | 26.83 |
| leaf_blight | brown_spot | 11 | 28.21 |
| target_spot | early_blight | 10 | 47.62 |
| septoria_leaf_spot | verticulium_wilt | 9 | 21.95 |
| leaf_mold | early_blight | 9 | 37.50 |
| stripe_rust | rust | 9 | 90.00 |
| healthy | diabrotica_speciosa | 9 | 2.55 |
| hispa | blast | 8 | 66.67 |
| yellow_leaf_curl_virus | early_blight | 8 | 18.18 |
| septoria_leaf_spot | leaf_blight | 8 | 19.51 |

