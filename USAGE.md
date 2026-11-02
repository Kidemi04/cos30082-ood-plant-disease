# How to use this repository

A step-by-step guide for training a model on DPD-Small and getting results for the report. The [README](README.md) has the full reference.

## What is here

| File | Use it for |
|---|---|
| `setup_dataset.bat`, `scripts/bootstrap_dataset.py` | One-time dataset setup |
| `dataset_access.py` | Loading the official splits and the shared validation split |
| `preprocessing.py` | Image loading, augmentation and PyTorch datasets |
| `evaluate.py` | Saving predictions and computing every number for the report |
| `docs/data_exploration/` | Dataset figures and findings for the report |

## 1. Set up once

**Dataset.** Follow [First setup on Windows](README.md#first-setup-on-windows) in the README, or on macOS/Linux:

```text
python3 scripts/bootstrap_dataset.py --archive path/to/DPD_Small.7z
```

Wait for `Dataset setup succeeded`. Your fingerprint must match the one in the README.

**Training environment.** Create a separate environment for training and install PyTorch from https://pytorch.org/get-started/locally/. Pick the CUDA build if you have an NVIDIA GPU. Then:

```text
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (macOS/Linux)
(the torch + torchvision install command from pytorch.org)
pip install numpy matplotlib pillow
```

Check it:

```text
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Load the data

```python
from preprocessing import build_datasets

datasets = build_datasets()   # train 6,779 | val 1,221 | test_seen 2,000 | test_unseen 1,448
item = datasets['train'][0]
item['image']        # 3 x 224 x 224 tensor, augmented and normalised
item['crop']         # 0-54
item['disease']      # 0-174
item['composition']  # 0-332
item['filename']
```

`train` is augmented. `val`, `test_seen` and `test_unseen` use fixed preprocessing. Do not write your own transforms; everyone uses the same ones so the comparison is fair.

## 3. Train

Train on `datasets['train']` and choose epochs, learning rate and so on with `datasets['val']`. Never look at test results while tuning.

A minimal multi-task baseline:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from preprocessing import build_datasets


class MultiTaskNet(nn.Module):
    """One backbone, two heads: crop (55 classes) and disease (175 classes)."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.crop = nn.Linear(features, 55)
        self.disease = nn.Linear(features, 175)

    def forward(self, images):
        features = self.backbone(images)
        return self.crop(features), self.disease(features)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = build_datasets()
    loader = DataLoader(datasets['train'], batch_size=32, shuffle=True, num_workers=4)
    model = MultiTaskNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for batch in loader:
            crop_logits, disease_logits = model(batch['image'].to(device))
            loss = loss_fn(crop_logits, batch['crop'].to(device)) + loss_fn(disease_logits, batch['disease'].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        torch.save(model.state_dict(), 'multitask.pt')


if __name__ == '__main__':   # required on Windows when num_workers > 0
    main()
```

For the independent baseline, train two separate networks, one predicting `batch['crop']` and one predicting `batch['disease']`.

Class imbalance is severe (1 to 544 images per composition). To draw rare compositions more often, use the shared sampler instead of `shuffle=True`:

```python
from preprocessing import balanced_sampler
loader = DataLoader(datasets['train'], batch_size=32, sampler=balanced_sampler(datasets['train'].records),
                    num_workers=4)
```

## 4. Save predictions

Run the model over a whole split and save the raw scores. Logits are fine; no softmax needed.

```python
import torch
from torch.utils.data import DataLoader

from evaluate import save_predictions


@torch.no_grad()
def predict(model, dataset, path, device):
    model.eval()
    names, crop_scores, disease_scores = [], [], []
    for batch in DataLoader(dataset, batch_size=64, num_workers=4):   # no shuffling
        crop_logits, disease_logits = model(batch['image'].to(device))
        names += batch['filename']
        crop_scores.append(crop_logits.cpu())
        disease_scores.append(disease_logits.cpu())
    save_predictions(path, names, crop=torch.cat(crop_scores).numpy(), disease=torch.cat(disease_scores).numpy())


predict(model, datasets['val'], 'predictions/multitask_val.npz', device)
```

Which scores to save depends on the model:

| Model | Save |
|---|---|
| Multi-task baseline | `crop=` and `disease=` from its two heads |
| Independent baseline | `crop=` from the crop network, `disease=` from the disease network, in the same file. Predict both on the same dataset without shuffling so the rows line up. |
| Model that scores the 333 compositions directly (for example vision-language) | `composition=` |

Columns must follow the official IDs: 55 crops, 175 diseases, 333 compositions. The `predictions/` folder is ignored by Git.

## 5. Evaluate

**While developing,** use the validation split:

```text
python evaluate.py run --name multitask_dev --val predictions/multitask_val.npz
```

**Once the model is final,** predict both test splits once and evaluate:

```python
predict(model, datasets['test_seen'], 'predictions/multitask_test_seen.npz', device)
predict(model, datasets['test_unseen'], 'predictions/multitask_test_unseen.npz', device)
```

```text
python evaluate.py run --name resnet50_multitask --seen predictions/multitask_test_seen.npz --unseen predictions/multitask_test_unseen.npz
```

This prints a summary and writes `results/resnet50_multitask/`:

| File | Contents |
|---|---|
| `summary.md` | Top-1 and Top-5 for compositions, crops and diseases; seen, unseen and harmonic mean; most confused pairs |
| `metrics.json` | The same numbers, for `compare` |
| `confusion_test_seen_*.png`, `.csv` | Seen-test confusion matrices at crop, disease and composition level |

The headline numbers are per-class mean accuracy, ranked against all 333 compositions. The README's [Evaluation](README.md#evaluation) section explains why.

## 6. Compare models

When every model has its test results:

```text
python evaluate.py compare results/*/metrics.json
```

This writes `results/comparison.md`, the results table for the report. Commit your `results/<name>/` folder so the team can build it.

## Rules

- Use the official splits and IDs. Do not renumber classes or create your own validation split.
- Tune only on `val`. Run test once, on the final model.
- Use `preprocessing.py`. If you change the input size or add augmentation, say so in the report.
- Report the backbone, pretrained weights, loss, optimiser, learning rate, batch size, epochs and image size.

## Common problems

| Problem | Fix |
|---|---|
| `Run setup_dataset.bat before loading the dataset` | Do step 1. |
| `Predictions do not match the split: N missing ...` | Predict the whole split, not a subset, and save every filename once. |
| `crop scores need 55 columns` | Your output layer has the wrong number of classes. |
| DataLoader hangs or crashes on Windows | Put your code inside `if __name__ == '__main__':`, or use `num_workers=0`. |
| CUDA out of memory | Lower `batch_size`. |
| `ModuleNotFoundError: torch` | Activate your training environment. |
