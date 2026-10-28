"""Import a model trained with kaggle/aeroprice_kaggle_training.ipynb.

Run from the project root:
    python pipeline/import_kaggle_model.py path\\to\\aeroprice_trained_model.zip

It rebuilds the architecture from model_def.py in your local TensorFlow 2.10,
loads the Kaggle weights layer-by-layer by name, checks that the predictions
match the ones Kaggle computed, then saves:
    models/final_model_multimodal.h5
    models/model_meta.json
If the zip also contains the deep + boosting hybrid (combined notebook), it
retrains that gradient-boosting model here with your own scikit-learn, from the
image features Kaggle computed, so there are no library-version problems, and saves:
    models/hybrid_boosting.joblib
    models/hybrid_features.npz
    models/test_predictions.csv
The better of the two hybrids on the test set becomes the app's active model.
The backend uses these automatically once they exist. The original image-only
model (models/final_model_satellite.h5) is left untouched.
"""
import json
import os
import sys
import shutil
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

from model_def import build_multimodal_model
from sklearn.model_selection import train_test_split

EXPORT_DIR = "models/kaggle_export"
CSV_PATH = "data/austin_master_dataset.csv"
IMAGE_DIR = "data/image_patches"
OUT_MODEL = "models/final_model_multimodal.h5"
OUT_META = "models/model_meta.json"
OUT_BOOSTING = "models/hybrid_boosting.joblib"
OUT_FEATURES = "models/hybrid_features.npz"
OUT_TEST_PREDS = "models/test_predictions.csv"
TOLERANCE = 0.02  # max allowed difference in log(price) between Kaggle and here


def leaf_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            yield from leaf_layers(layer)
        else:
            yield layer


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pipeline/import_kaggle_model.py <aeroprice_trained_model.zip or folder>")
    src = sys.argv[1]
    if zipfile.is_zipfile(src):
        os.makedirs(EXPORT_DIR, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(EXPORT_DIR)
        folder = EXPORT_DIR
    else:
        folder = src

    with open(os.path.join(folder, "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("quick_test"):
        print("WARNING: this export came from a QUICK_TEST run and is not a properly trained model.")
    features = meta["numeric_features"]
    image_size = int(meta["image_size"])
    print(f"Features: {features}")
    print(f"Trained with: {meta.get('trained_with')}")

    # 1. Rebuild the architecture and load weights by layer name
    model = build_multimodal_model(num_tab_features=len(features), image_size=image_size, pretrained=False)
    saved = np.load(os.path.join(folder, "weights_by_layer.npz"))
    used = set()
    for layer in leaf_layers(model):
        current = layer.get_weights()
        if not current:
            continue
        new = []
        for i, w in enumerate(current):
            key = f"{layer.name}::{i}"
            if key not in saved:
                raise SystemExit(f"Missing weights for layer '{layer.name}'. Notebook and model_def.py differ.")
            if saved[key].shape != w.shape:
                raise SystemExit(f"Shape mismatch for {key}: Kaggle {saved[key].shape} vs local {w.shape}")
            new.append(saved[key])
            used.add(key)
        layer.set_weights(new)
    unused = set(saved.files) - used
    if unused:
        raise SystemExit(f"{len(unused)} Kaggle weight arrays were not used, e.g. {sorted(unused)[:3]}")
    print(f"Loaded {len(used)} weight arrays.")

    # 2. Check against Kaggle's reference predictions
    ref = pd.read_csv(os.path.join(folder, "reference_predictions.csv"))
    df = pd.read_csv(CSV_PATH)
    df["image_file"] = df["rgb_path"].astype(str).str.replace("\\", "/", regex=False).map(os.path.basename)
    df = df.set_index("image_file")
    med = np.array(meta["feature_median"], dtype="float32")
    mean = np.array(meta["feature_mean"], dtype="float32")
    scale = np.array(meta["feature_scale"], dtype="float32")
    imgs, tabs = [], []
    for name in ref["image_file"]:
        img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB").resize((image_size, image_size))
        imgs.append(preprocess_input(np.asarray(img, dtype="float32")))
        raw = pd.to_numeric(df.loc[name, features], errors="coerce").to_numpy("float32")
        raw = np.where(np.isnan(raw), med, raw)
        tabs.append((raw - mean) / scale)
    local = model.predict([np.stack(imgs), np.stack(tabs)], verbose=0).reshape(-1)
    diff = np.abs(local - ref["pred_log"].to_numpy())
    print(f"Reference check: max difference in log(price) = {diff.max():.5f} (tolerance {TOLERANCE})")
    if diff.max() > TOLERANCE:
        raise SystemExit("Predictions do not match Kaggle's. Not saving. Send this output to Claude.")

    # 3. Deep + boosting hybrid (only in exports from the combined notebook)
    features_npz = os.path.join(folder, "hybrid_features.npz")
    if os.path.exists(features_npz) and "boosting" in meta:
        local_med = import_boosting(folder, meta, features_npz)
        nn_med = next((r.get("median % error") for r in meta.get("test_results", []) if str(r.get("model", "")).startswith("1.")), None)
        meta["boosting"]["local_test_median_pct_error"] = local_med
        meta["active_model"] = "boosting_hybrid" if nn_med is None or local_med < nn_med else "neural_hybrid"
        print(f"Test median % error: neural hybrid {nn_med}%, deep + boosting hybrid {local_med:.1f}% "
              f"-> active model: {meta['active_model']}")
    else:
        meta["active_model"] = "neural_hybrid"
        print("No boosting hybrid in this export (older notebook). Active model: neural_hybrid")

    # 4. Save
    model.save(OUT_MODEL)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {OUT_MODEL} and {OUT_META}. Restart the backend to use the new model(s).")


def load_split_rows():
    """Same filtering and 80/20 + 10% validation split as the Kaggle notebook."""
    df = pd.read_csv(CSV_PATH)
    df["image_file"] = df["rgb_path"].astype(str).str.replace("\\", "/", regex=False).map(os.path.basename)
    available = set(os.listdir(IMAGE_DIR))
    df = df[df["rgb_path"].astype(str).str.len() > 0]
    df = df[df["image_file"].isin(available)].copy()
    df["latestPrice"] = pd.to_numeric(df["latestPrice"], errors="coerce")
    df = df.dropna(subset=["latestPrice"]).reset_index(drop=True)
    trainval_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, random_state=42)
    return df, train_idx, val_idx, test_idx


def import_boosting(folder, meta, features_npz):
    """Retrain the gradient-boosting hybrid locally and return its test median % error."""
    import joblib
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor

    store = np.load(features_npz)
    df, train_idx, val_idx, test_idx = load_split_rows()
    pos = {name: i for i, name in enumerate(store["image_file"])}
    missing = [f for f in df["image_file"] if f not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} local properties have no Kaggle image features (e.g. {missing[:3]}). "
                         "Was the notebook run on the same dataset?")
    rows = np.array([pos[f] for f in df["image_file"]])
    has = store["has_features"][rows]
    train_idx, val_idx, test_idx = (i[has[i]] for i in (train_idx, val_idx, test_idx))

    # The test properties must be the same ones Kaggle used
    kaggle_test = pd.read_csv(os.path.join(folder, "test_predictions.csv"))
    if list(kaggle_test["image_file"]) != list(df["image_file"].to_numpy()[test_idx]):
        raise SystemExit("The local test split does not match Kaggle's. Is data/austin_master_dataset.csv the same file?")

    feats = meta["numeric_features"]
    med = np.array(meta["feature_median"], dtype="float32")
    raw = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy("float32")
    raw = np.where(np.isnan(raw), med, raw)
    X = np.hstack([raw, store["img_pca"][rows]])
    y = df["latestPrice"].to_numpy("float64")

    params = dict(meta["boosting"]["params"])
    gbm = HistGradientBoostingRegressor(**params).fit(X[train_idx], np.log1p(y[train_idx]))
    val_pred = np.expm1(gbm.predict(X[val_idx]))
    price_scale = float(np.mean(np.exp(np.log1p(y[val_idx]) - np.log1p(val_pred))))
    test_pred = np.expm1(gbm.predict(X[test_idx])) * price_scale
    pct = np.abs(test_pred - y[test_idx]) / y[test_idx] * 100
    local_med = float(np.median(pct))
    kaggle_row = next((r for r in meta.get("test_results", []) if str(r.get("model", "")).startswith("2.")), {})
    print(f"Boosting hybrid rebuilt with scikit-learn {sklearn.__version__}: test median % error {local_med:.1f}% "
          f"(Kaggle: {kaggle_row.get('median % error')}%), MAE ${np.abs(test_pred - y[test_idx]).mean():,.0f} "
          f"(Kaggle: ${kaggle_row.get('MAE $')})")

    joblib.dump({"model": gbm, "features": feats + list(meta["boosting"]["image_features"]),
                 "numeric_features": feats, "price_scale": price_scale,
                 "sklearn_version": sklearn.__version__}, OUT_BOOSTING)
    shutil.copyfile(features_npz, OUT_FEATURES)
    meta["boosting"]["price_scale_local"] = price_scale

    preds = kaggle_test.copy()
    preds["boosting_hybrid"] = test_pred
    if os.path.exists(OUT_TEST_PREDS):  # keep an image_only column written by evaluate_model.py
        old = pd.read_csv(OUT_TEST_PREDS)
        if "image_only" in old.columns:
            preds = preds.merge(old[["image_file", "image_only"]], on="image_file", how="left")
    preds.to_csv(OUT_TEST_PREDS, index=False)
    print(f"Saved {OUT_BOOSTING}, {OUT_FEATURES} and {OUT_TEST_PREDS}")
    return local_med


if __name__ == "__main__":
    main()
