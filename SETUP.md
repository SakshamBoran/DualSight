# DualSight: local setup (Windows, CPU-only laptop)

Satellite + listing-detail price estimates for Austin, TX, with a Kaggle notebook that trains two hybrid models (image + numbers), and real, model-computed explanations in the app.
Your laptop runs the app and every data step. Training happens on Kaggle's free GPU.

## 1. Two Python 3.10 environments

TensorFlow 2.10 and current Streamlit need conflicting versions of numpy and protobuf, so each gets its own environment.
Python 3.10 is required (`py -0` should list `-V:3.10`).

```bash
cd DualSight

py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate

py -3.10 -m venv venv_ui
venv_ui\Scripts\activate
pip install -r requirements_streamlit.txt
deactivate
```

## 2. Data and trained models (not in the repo)

**Data.** From https://www.kaggle.com/datasets/sanveersinghsuneja/austin-satellite-property-data:
- `austin_master_dataset.csv` goes into `data\`
- the PNGs go **directly** into `data\image_patches\` (the download nests them one level deeper; move them up)

The Kaggle CSV stores paths as `image_patches\property_0_rgb.png`. Rewrite them once so they include `data/`:

```bash
venv\Scripts\activate
copy data\austin_master_dataset.csv data\austin_master_dataset.backup.csv
python -c "import pandas as pd; d=pd.read_csv('data/austin_master_dataset.csv'); [d.__setitem__(c, d[c].str.replace('\\','/',regex=False).radd('data/')) for c in ('rgb_path','ndvi_path')]; d.to_csv('data/austin_master_dataset.csv',index=False)"
```

**Trained models.** Download `dualsight_trained_model.zip` from https://huggingface.co/SakshamBoran/dualsight (Files and versions tab), then import it:

```bash
venv\Scripts\activate
python pipeline/import_kaggle_model.py "%USERPROFILE%\Downloads\dualsight_trained_model.zip"
```

Or train your own with the Kaggle notebook (section 5).

## 3. Run the app

Either run `python main.py` from `venv` (it starts the backend, then Streamlit from `venv_ui`), or use two terminals:

```bash
:: Terminal 1 - backend
venv\Scripts\activate
python backend/main.py
```

```bash
:: Terminal 2 - web interface
venv_ui\Scripts\activate
streamlit run frontend/app.py
```

Open http://localhost:8501, pick an address, then click **Show prediction**.
Wait for `Active model` in terminal 1 before clicking (about 20–30 s on first start).

Command-line prediction: `python pipeline/predict_cmd.py 123`. Add `--explain` for the breakdown, or `--model neural_hybrid|boosting_hybrid` to pick a model.

## 4. Measure accuracy

```bash
venv\Scripts\activate
python pipeline/evaluate_model.py              # whole 20% test set
python pipeline/evaluate_model.py --sample 500 # faster
```

Scores both hybrid models on the same held-out properties, none of which were used in training.

## 5. Retrain on Kaggle: two hybrid models

1. On kaggle.com: **Create → New Notebook → File → Import Notebook** and upload `kaggle\dualsight_kaggle_training.ipynb`.
2. **Settings:** Accelerator **GPU T4 x2** (or P100), and **Internet on**.
3. **Add Input:** dataset `sanveersinghsuneja/austin-satellite-property-data`.
4. **Run All** (about 25–70 min). It trains and compares, on one test set:
   - **Model 1, neural hybrid:** ResNet50 image branch + numeric branch, trained end to end.
   - **Model 2, deep + boosting hybrid:** ResNet50 image features (PCA) + numbers, gradient boosting.
   - A numbers-only model, for comparison.

   Optional switches in the Settings cell: `USE_LATLON = True` adds latitude/longitude; `QUICK_TEST = True` does a short check run.
   (`kaggle\dualsight_kaggle_training_v1.ipynb` is the earlier, neural-only version.)
5. Download `dualsight_trained_model.zip` from the Output panel.
6. On your laptop:

```bash
venv\Scripts\activate
python pipeline/import_kaggle_model.py "%USERPROFILE%\Downloads\dualsight_trained_model.zip"
```

This:
- rebuilds model 1 in TensorFlow 2.10, loads the Kaggle weights by layer name, and checks its predictions match Kaggle's;
- retrains model 2's gradient boosting with your own scikit-learn from Kaggle's image features, and checks the test score matches;
- makes the better of the two the **active model**, then saves everything in `models\`.

Restart the backend to use it. To choose a model yourself, set `DUALSIGHT_MODEL` before starting the backend:

```bash
set DUALSIGHT_MODEL=neural_hybrid
python backend/main.py
```

(`neural_hybrid` or `boosting_hybrid`.) The first start with the boosting hybrid downloads ~94 MB of ImageNet weights, used for the image map.

## 6. Explanations (explainable AI)

**Explain this estimate** computes everything from the active model (a few seconds on a laptop CPU):

| Chart | Method |
| --- | --- |
| Why this price | Boosting hybrid: exact **TreeSHAP**. Neural hybrid: **integrated gradients** for each number, plus the image's effect vs. 16 typical patches. The steps add up exactly to the prediction. |
| Which parts of the satellite image mattered | **Occlusion map**: each square of an 8×8 grid is greyed out in turn and the price re-predicted. Green raised the price, red lowered it. |
| Predicted vs actual on held-out properties | Real test-set predictions for the active model, with this property starred. |

The notebook also saves a global SHAP importance chart (`global_importance.png`) for model 2.
The 1–10 ratings and "Inferred amenities" come from simple rule-based formulas, not model output, and the app labels them that way.
Speed knobs: `DUALSIGHT_OCCLUSION_GRID=4` (default 8) and `DUALSIGHT_IG_STEPS=8` (default 16).

## 7. Rebuilding the satellite data from scratch (optional)

Needs `data/austin_data.csv` (raw) and your **own** Sentinel Hub OAuth client. Never paste keys into code.

```bash
set SH_CLIENT_ID=your-client-id
set SH_CLIENT_SECRET=your-client-secret
python pipeline/clean_austin_data.py
python pipeline/compute_bbox_from_cleaned.py
python pipeline/download_sentinel.py LAT_MIN LON_MIN LAT_MAX LON_MAX
python pipeline/compute_ndvi.py
python pipeline/generate_image_patches.py
```
