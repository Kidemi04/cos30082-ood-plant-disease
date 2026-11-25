# Baseline 2: shared ConvNeXt-Tiny

One ImageNet-pretrained ConvNeXt-Tiny backbone feeds a 55-class crop head and a 175-class disease head. The loss is the sum of cross-entropy losses with label smoothing 0.1. Input transforms, the fixed validation split, composition-balanced square-root sampling, and evaluation use the shared project modules.

The validation run trains the heads for 3 epochs, then fine-tunes the whole network with early stopping. It selects a checkpoint by validation composition per-class Top-1 using the shared evaluator. This validation set contains seen compositions, so it is not a direct measure of unseen-composition generalisation. The independent baseline uses crop- or disease-specific sampling and selects each model separately; note these differences when comparing.

From the repository root, after setting up the dataset and the PyTorch environment:

```text
python baseline2.py train --output outputs/baseline2_validation_seed42
```

`config.json`, `history.csv`, and `best.pth` record the settings, timings, and selected stage/epoch. The full-model learning-rate schedule covers 30 epochs, even if early stopping occurs. Model weights and outputs stay out of Git.

For final retraining, use the selected full-model epoch from `best.pth`, or the selected head epoch with `--final-full-epochs 0` if the best checkpoint is from the head stage. Keep the original `--head-epochs` and `--epochs` schedule lengths. Example when full epoch 5 is selected:

```text
python baseline2.py train --output outputs/baseline2_final_seed42 --final-full-epochs 5
python baseline2.py predict --checkpoint outputs/baseline2_final_seed42/final.pth --split test_seen --output predictions/baseline2_test_seen.npz
python baseline2.py predict --checkpoint outputs/baseline2_final_seed42/final.pth --split test_unseen --output predictions/baseline2_test_unseen.npz
python evaluate.py run --name baseline2 --seen predictions/baseline2_test_seen.npz --unseen predictions/baseline2_test_unseen.npz
```

If the best checkpoint is head epoch 2, use `--final-head-epochs 2 --final-full-epochs 0` instead. The final run starts from pretrained weights and uses all 8,000 official training images. Do not select settings based on test results.
