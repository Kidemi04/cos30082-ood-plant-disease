# Plant disease UI

Gradio app for the COS30082 OOD plant disease project. Upload a leaf photo and it shows:

- the predicted **crop**, **disease** and **crop–disease pair**, with a badge for pairs never seen in training;
- the **Top 5** pairs;
- the **most similar training images**: the closest photos of the predicted pair, or for an unseen pair,
  the closest same-crop and same-disease photos;
- a **Compare models** tab that runs both baselines and the proposed approaches (BioCLIP heads with standard and with
  strong augmentation, Disentangled v2, CSP + WiSE-FT) on the same photo.

Scores use the same calibration (gamma) as the reported results, so the UI agrees with `results/`.

## Run

Double-click `ui\run_ui.bat`, or from the repo root:

```
python ui\app.py
```

It opens http://127.0.0.1:7860. The first launch loads the proposed model (a few seconds).

## Files

| File | What it does |
|---|---|
| `config.json` | Which run is the proposed model, the baseline checkpoints, and their gamma values |
| `predictor.py` | Loads the models and scores an image |
| `views.py`, `style.css` | The page layout and look |
| `app.py` | The Gradio app |
| `build_index.py` | One-off: builds `assets/` (thumbnails, embeddings, examples) from the dataset |
| `check_predictor.py` | Checks the UI's scores against the saved calibrated test predictions |

`assets/` (about 110 MB) holds training-image thumbnails, so it is not committed. Rebuild it with
`python ui\build_index.py` on a machine that has the dataset, or copy the folder.

## Changing models

- **Different proposed run:** change `models.proposed.run` in `config.json`, then rebuild `assets/`
  (the similar-images search uses the proposed model's image encoder).
- **Gamma values:** each model's `gamma` is in `config.json` (a number, or the run's `calibration.json`).
  A model with `"gamma": null` runs uncalibrated and the UI says so.
- **Results table:** each model's `results` entry holds its calibrated test numbers and their source file.

## Needed on the demo machine

- The repo, including `src/` (with `train_disentangled_v2.py` and `csp_model.py` from main), `evaluate.py`,
  `preprocessing.py` and `dataset_access.py`
- The model folders in `runs/` (not in git):
  - `clip_bioclip_heads_strong_final/` (BioCLIP heads + strong augmentation)
  - `clip_bioclip_heads_final/` (BioCLIP heads, standard augmentation)
  - `disentangled_v2_final_seed42/` (Disentangled v2)
  - `csp_wiseft_vit_b16_final_seed42/` (CSP + WiSE-FT)
  - `baseline1_crop_convnext_tiny_balanced_final_seed42/`, `baseline1_disease_convnext_tiny_balanced_final_seed42/`
  - `baseline2_final_seed42/`
- `ui/assets/`
- Python with `requirements-ui.txt` installed
- Two base-model downloads in `%USERPROFILE%\.cache\huggingface\hub\` (fetched automatically on a first run
  with internet, or copied from another machine):
  - `models--imageomics--bioclip` (BioCLIP heads)
  - `models--timm--vit_base_patch16_clip_224.openai` (CSP + WiSE-FT)

With all six models loaded the app uses about 2.3 GB of GPU memory (measured), which fits the 4 GB RTX 2050.
