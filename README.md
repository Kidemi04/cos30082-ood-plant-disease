# Shared DPD-Small dataset setup

All teammates use the same original archive, official splits, and label IDs. Setup verifies the release and prepares portable CSV files. It does not train a model or change the images.

## Get this repository

Download and extract the repository ZIP from GitHub, or run:

```text
git clone https://github.com/Kidemi04/cos30082-ood-plant-disease.git
cd cos30082-ood-plant-disease
```

This is a private team repository. Use a GitHub account that already has access. The dataset archive is downloaded separately from the lecturer; it is not stored on GitHub.

## First setup on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/ if it is not already installed. On Windows, enable the Python launcher or add Python to PATH.
2. Download the complete `DPD_Small.7z` supplied by the lecturer. Place it at `dataset/raw/DPD_Small.7z` inside this project. Do not rename a partial browser download.
3. Double-click `setup_dataset.bat`. The first run needs Internet access to install the pinned Python dependencies in `.venv-dataset`. It does not install them into your global Python environment.
4. Wait for `DATASET READY`, followed by `Dataset setup succeeded`. Compare the dataset fingerprint with the team. The same release and script version must produce the same fingerprint.

The archive is 3,399,045,528 bytes. Extraction writes about 3.46 GB in addition to the archive. Allow at least 8 GB free before downloading and extracting. The original archive remains in place. Setup uses an installed 7-Zip if available, otherwise the Python extraction library installed by the launcher.

Verified dataset fingerprint for this release:

```text
2a687d87a57f88f3815506299c131f2ec687bcb206278688165342d37cbc6a1c
```

Windows verification on 2026-11-01 decoded all 11,448 images, verified all 11,451 extracted file checksums, found no byte-identical image groups, and preserved the official counts. The launcher and repeated setup passed. Native 7-Zip was used for the real archive; Python fallback extraction was tested on a small archive, not on the complete 3.2 GB release.

```text
ML-Project/
  setup_dataset.bat
  dataset/
    raw/DPD_Small.7z             <- place your downloaded archive here
    extracted/                  <- original images and three original CSV files
      DPD_Small_dataset/
    metadata/dataset_reference.json  <- shared archive hash and file checksums
    prepared/                   <- generated automatically; do not edit
      train.csv
      test_seen.csv
      test_unseen.csv
      label_map.json
      dataset_info.json
      validation_report.json
```

No images are copied into separate split folders. The prepared CSV files point to the original extracted images. All generated paths are relative; teammates do not need the same Windows username or drive letter.

## Repeat runs and errors

Run the same launcher again. Unchanged files reuse the previous successful full verification. Setup still checks the file list, sizes, modification times and metadata; it does not re-extract a completed dataset. This quick check does not reread every image byte. For a fresh archive checksum and full image check, run:

```bat
setup_dataset.bat --verify
```

An interrupted extraction stays in `dataset/.extracting` and is checked on the next run. A complete staging folder is reused; an incomplete one is re-extracted. The prepared dataset is only marked ready after validation succeeds. Extraction recovery may decompress the solid archive again; it does not promise compressed-byte resume.

If a completed dataset has missing or changed files, setup reports the problem. It does not silently delete files or choose a different release. Preserve any work you added there before repairing the extracted copy. Generated output metadata is regenerated on each successful run. Never use a nonzero exit result as success.

For an existing archive elsewhere, no copy is required:

```bat
setup_dataset.bat --archive "D:\Downloads\DPD_Small.7z"
```

Keep that archive available when rerunning setup, or put it in `dataset/raw` and use the default command. The Windows launcher pauses so double-click users can read errors. On macOS/Linux, use Python 3.10+ with `python3 scripts/bootstrap_dataset.py` instead; these platforms have not been tested on a real DPD-Small extraction here.

## Use in model code

From the project directory:

```python
from PIL import Image
from dataset_access import load_split, load_labels

train = load_split('train')
labels = load_labels()
sample = train[0]
image = Image.open(sample['image_path']).convert('RGB')
crop_target = sample['crop_id']
disease_target = sample['disease_id']
composition_target = sample['composition_id']
```

Each prepared CSV has a header: `image_path,filename,crop_id,disease_id,composition_id,crop_name,disease_name,composition_name`. `image_path` is relative to `dataset/extracted`. The original lecturer CSV files have **no header**. Do not accidentally discard the first sample using a default CSV reader header.

## Shared experimental rules

- Preserve the official splits: train 8,000; test seen 2,000; test unseen 1,448.
- Preserve crop IDs 0-54, disease IDs 0-174, and composition IDs 0-332. There are 323 seen compositions and 10 unseen compositions. Training composition IDs are not contiguous: do not independently renumber them or use the maximum train ID to infer the number of seen classes.
- The unseen compositions use crop and disease components that already occur in training. `label_map.json` records their component pairs and the seen/unseen composition IDs. Label-map JSON keys are strings; CSV IDs and loader targets are integers.
- Only training images and labels are training examples. Test image labels are for final evaluation. Do not tune thresholds, preprocessing, model selection or prompts using test results.
- No validation split has been invented. Agree on one fixed split from the official train data before model selection, then share its manifest in Git. Do not create a different random validation split per teammate.
- Keep original images. Resize, augmentation and normalization belong in each model's input pipeline and must be documented for fair comparisons.
- Byte-identical images, if found, are listed in `validation_report.json`. They are reported without silently changing the lecturer's splits. Different encodings or near-duplicate images are not detected by this check.

## Validation split

The shared validation split is `dataset/metadata/validation_split.csv`: 1,221 training images, 15% of every composition with at least 3 training images (seed 30082). The 73 compositions with fewer than 3 images stay entirely in training. Do not regenerate it; everyone uses this file.

```python
from dataset_access import load_validation_split
train, val = load_validation_split()   # 6,779 + 1,221 records
```

Use `val` for model selection and tuning. The test splits are for final results only.

## Data exploration

[docs/data_exploration/README.md](docs/data_exploration/README.md) has the figures and findings for the report: class imbalance, crops and diseases, the 10 unseen compositions, source datasets and image sizes. Regenerate it with `python scripts/explore_dataset.py`. The augmentation example image is only made when PyTorch is installed.

Findings that affect model design:
- Training images per composition range from 1 to 544, with a median of 7. 40 compositions have a single training image.
- Every unseen crop and disease appears in training, never together. `cherry_aphis_spp` is the hardest case: aphis_spp has only 7 training images.
- 36 of the 42 source datasets contain one crop, so a model can tell the crop from the photo style. Disease recognition is the real difficulty.
- Unseen test images are smaller (median shorter side 256 px, against 448 px in train).

## Preprocessing and augmentation

All models use [preprocessing.py](preprocessing.py), so differences in results come from the models, not the input pipeline. It needs `torch` and `torchvision`.

| Step | Training | Validation, test and UI |
|---|---|---|
| Loading | RGB, EXIF orientation applied; large JPEGs decoded at reduced size (shorter side at least 256) | same |
| Resize | `RandomResizedCrop(224, scale=(0.5, 1.0))` | resize shorter side to 256, centre-crop 224 |
| Flips | horizontal and vertical, p = 0.5 each | none |
| Colour | `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)` | none |
| Normalise | ImageNet mean and std | same |

Hue jitter is kept small because leaf colour (yellowing, rust, mildew) is itself a disease symptom.

```python
from torch.utils.data import DataLoader
from preprocessing import build_datasets, balanced_sampler

datasets = build_datasets()   # train, val, test_seen, test_unseen
train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True, num_workers=4)
for batch in train_loader:
    images, crops, diseases = batch['image'], batch['crop'], batch['disease']
```

Each item also has `composition` and `filename`; pass the filenames to `save_predictions`. Other options:
- `build_datasets(image_size=...)` changes the input size for a backbone that needs another resolution. Report the size used.
- `build_datasets(use_validation=False)` gives the full 8,000 training images for a final retrain after model selection.
- `balanced_sampler(datasets['train'].records)` draws rare compositions more often (square-root sampling). Pass it as `sampler=` with `shuffle=False`. Whether to use it is a modelling choice; document it.
- `eval_transform()` and `load_image()` give the same preprocessing for the user interface.

## Evaluation

All models are scored by `evaluate.py`, so results are comparable. It needs `numpy` and `matplotlib` (plots are skipped without matplotlib). Each model saves its scores for each split. Columns are indexed by the official IDs: 55 crops, 175 diseases, 333 compositions. Raw logits or softmax probabilities are both accepted.

```python
from evaluate import save_predictions
# Baselines (independent or multi-task): crop and disease heads.
save_predictions('predictions/resnet50_multitask_test_seen.npz', filenames, crop=crop_logits, disease=disease_logits)
# Models that score compositions directly, for example a vision-language model.
save_predictions('predictions/clip_test_unseen.npz', filenames, composition=composition_logits)
```

Then run:

```text
python evaluate.py run --name resnet50_multitask --seen predictions/resnet50_multitask_test_seen.npz --unseen predictions/resnet50_multitask_test_unseen.npz
python evaluate.py compare results/*/metrics.json
```

`results/<name>/` receives `metrics.json`, `summary.md` and the seen-test confusion matrices (CSV and PNG) at composition, crop and disease level. `compare` writes the report table to `results/comparison.md`. Use `--val` instead of the test flags while developing.

Protocol:
- **Generalised zero-shot:** seen and unseen test images are ranked against all 333 compositions. The model is not told which split an image comes from. Closed-set accuracy over only the 10 unseen compositions is reported separately as `composition_unseen_only`.
- **Composition scores from two heads:** P(composition) = P(crop) × P(disease), renormalised over the 333 valid pairs. If a model gives only composition scores, crop and disease scores are its summed composition probabilities.
- **Primary metric:** per-class mean accuracy. Seen-test classes range from 1 to 124 images, and `tomato_early_blight` is 33% of the unseen test set, so per-image accuracy would mostly measure a few large classes. Per-image accuracy is reported too. The harmonic mean is H = 2·S·U / (S + U), computed from seen and unseen composition accuracy, for Top-1 and Top-5.

## Git and checks

Share this project folder's scripts, requirements, README and `dataset/metadata`. Do not share `.venv-dataset`, archives, extracted images, generated prepared files or the machine-local setup state in Git. They are covered by this folder's `.gitignore`.

Run the focused verification tests (dataset setup, evaluation and preprocessing; the preprocessing tests are skipped without PyTorch) with:

```bat
.venv-dataset\Scripts\python.exe -m unittest discover -s tests -v
```
