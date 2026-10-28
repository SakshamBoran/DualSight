"""Measure how accurate the model(s) really are on the held-out test set.

Run from the project root:
    python pipeline/evaluate_model.py            # whole test set (about 3,000 properties)
    python pipeline/evaluate_model.py --sample 500

Uses the same 80/20 split as training (random_state=42), so the properties
scored here were never trained on. Scores the neural hybrid and the deep +
boosting hybrid (whichever you have imported) on the same rows.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

CSV_PATH = "data/austin_master_dataset.csv"
IMAGE_DIR = "data/image_patches"
MULTIMODAL_MODEL = "models/final_model_multimodal.h5"
META_PATH = "models/model_meta.json"
BOOSTING_PATH = "models/hybrid_boosting.joblib"
FEATURES_PATH = "models/hybrid_features.npz"
IMAGE_SIZE = (128, 128)


def load_test_rows(sample):
    df = pd.read_csv(CSV_PATH)
    df["image_file"] = df["rgb_path"].astype(str).str.replace("\\", "/", regex=False).map(os.path.basename)
    available = set(os.listdir(IMAGE_DIR))
    df = df[df["rgb_path"].astype(str).str.len() > 0]
    df = df[df["image_file"].isin(available)].copy()
    df["latestPrice"] = pd.to_numeric(df["latestPrice"], errors="coerce")
    df = df.dropna(subset=["latestPrice"]).reset_index(drop=True)
    _, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    if sample and sample < len(test_idx):
        test_idx = np.random.default_rng(0).choice(test_idx, size=sample, replace=False)
    return df.iloc[test_idx].reset_index(drop=True)


def report(name, actual, pred):
    err = pred - actual
    pct = np.abs(err) / actual * 100
    la, lp = np.log1p(actual), np.log1p(np.maximum(pred, 0))
    r2 = 1 - np.sum((la - lp) ** 2) / np.sum((la - la.mean()) ** 2)
    print(f"\n=== {name} ===")
    print(f"Mean absolute error:          ${np.abs(err).mean():,.0f}")
    print(f"Median absolute error:        ${np.median(np.abs(err)):,.0f}")
    print(f"Median % error:               {np.median(pct):.1f}%")
    print(f"Within 20% of actual:         {(pct <= 20).mean() * 100:.1f}%")
    print(f"Average bias (pred - actual): ${err.mean():+,.0f}")
    print(f"R-squared (log price):        {r2:.3f}")
    print(f"Predicted price range:        ${pred.min():,.0f} to ${pred.max():,.0f}")
    print(f"Predictions above $5M:        {(pred > 5e6).sum()}")


def main():
    sample = None
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])
    rows = load_test_rows(sample)
    actual = rows["latestPrice"].to_numpy("float64")
    print(f"Loading {len(rows)} held-out images...")
    x_img = np.stack([
        preprocess_input(np.asarray(Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB").resize(IMAGE_SIZE), dtype="float32"))
        for f in rows["image_file"]
    ])

    print(f"Properties scored: {len(rows)}")
    print(f"Actual price range: ${actual.min():,.0f} to ${actual.max():,.0f}")
    print(f"Guess-the-median MAE: ${np.abs(actual - np.median(actual)).mean():,.0f}  (a useful model must beat this)")

    if os.path.exists(MULTIMODAL_MODEL) and os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        feats = meta["numeric_features"]
        raw = rows[feats].apply(pd.to_numeric, errors="coerce").to_numpy("float32")
        raw = np.where(np.isnan(raw), np.array(meta["feature_median"], dtype="float32"), raw)
        x_tab = (raw - np.array(meta["feature_mean"], dtype="float32")) / np.array(meta["feature_scale"], dtype="float32")
        model = tf.keras.models.load_model(MULTIMODAL_MODEL)
        pred = np.expm1(model.predict([x_img, x_tab], batch_size=32, verbose=0).reshape(-1).astype("float64"))
        report("Neural hybrid (image + numbers)", actual, pred * float(meta.get("price_scale", 1.0)))
    else:
        print("\n(No neural hybrid found. Import the trained models with pipeline/import_kaggle_model.py.)")

    if os.path.exists(BOOSTING_PATH) and os.path.exists(FEATURES_PATH):
        import joblib
        bundle = joblib.load(BOOSTING_PATH)
        store = np.load(FEATURES_PATH)
        pos = {name: i for i, name in enumerate(store["image_file"])}
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        raw = rows[bundle["numeric_features"]].apply(pd.to_numeric, errors="coerce").to_numpy("float32")
        raw = np.where(np.isnan(raw), np.array(meta["feature_median"], dtype="float32"), raw)
        X = np.hstack([raw, store["img_pca"][[pos[f] for f in rows["image_file"]]]])
        pred = np.expm1(bundle["model"].predict(X)) * float(bundle["price_scale"])
        report("Deep + boosting hybrid", actual, pred)


if __name__ == "__main__":
    main()
