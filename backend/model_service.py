"""Loads whichever DualSight model is active, predicts prices and explains them.

Two models can be active:
    neural_hybrid    image + numbers, trained end to end on Kaggle
    boosting_hybrid  ResNet50 image features (PCA) + numbers, gradient boosting

Which one is used: the DUALSIGHT_MODEL environment variable if set, otherwise
"active_model" in models/model_meta.json (chosen by import_kaggle_model.py),
otherwise the deep + boosting hybrid if it exists.

Every explanation is computed from the active model itself:
    boosting_hybrid  exact TreeSHAP values
    neural_hybrid    integrated gradients for the numbers + image effect vs. typical images
and an occlusion map shows which parts of the satellite patch raised or lowered
the price (grey out one square, re-predict, measure the change).
"""
import json
import os
import threading

import numpy as np
import pandas as pd
from PIL import Image

MODELS_DIR = "models"
NEURAL_PATH = os.path.join(MODELS_DIR, "final_model_multimodal.h5")
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")
BOOSTING_PATH = os.path.join(MODELS_DIR, "hybrid_boosting.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "hybrid_features.npz")
TEST_PREDS_PATH = os.path.join(MODELS_DIR, "test_predictions.csv")

IMAGE_SIZE = 128
CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")
OCCLUSION_GRID = int(os.environ.get("DUALSIGHT_OCCLUSION_GRID", "8"))
IG_STEPS = int(os.environ.get("DUALSIGHT_IG_STEPS", "16"))
N_BACKGROUND = 16

MODEL_LABELS = {
    "neural_hybrid": "Neural hybrid (image + numbers)",
    "boosting_hybrid": "Deep + boosting hybrid (image features + numbers)",
}
FEATURE_LABELS = {
    "livingAreaSqFt": "Living area", "yearBuilt": "Year built", "numOfBedrooms": "Bedrooms",
    "numOfBathrooms": "Bathrooms", "avgSchoolRating": "School rating", "lotSizeSqFt": "Lot size",
    "latitude": "Latitude", "longitude": "Longitude",
}


def caffe_preprocess(rgb):
    """keras.applications.resnet50.preprocess_input, written out so it needs no TF import."""
    return rgb[..., ::-1].astype("float32") - CAFFE_MEAN_BGR


def pil_image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype="float32")


def to_dollar_steps(base_log, contributions, price_scale):
    """Turn log-price contributions into dollar steps that add up exactly.

    Applied one at a time (largest first): each step is the dollar change from
    adding that contribution to everything before it.
    """
    order = sorted(contributions, key=lambda kv: -abs(kv[1]))
    running = base_log
    steps = []
    for label, c in order:
        before = np.expm1(running) * price_scale
        running += c
        steps.append((label, float(np.expm1(running) * price_scale - before), float(c)))
    return float(np.expm1(base_log) * price_scale), steps, float(np.expm1(running) * price_scale)


class ModelService:
    def __init__(self, dataset_df):
        self.df = dataset_df
        self.meta = None
        if os.path.exists(META_PATH):
            with open(META_PATH, encoding="utf-8") as f:
                self.meta = json.load(f)
        self.available = [k for k, ok in (
            ("neural_hybrid", os.path.exists(NEURAL_PATH) and self.meta is not None),
            ("boosting_hybrid", os.path.exists(BOOSTING_PATH) and os.path.exists(FEATURES_PATH) and self.meta is not None),
        ) if ok]
        wanted = os.environ.get("DUALSIGHT_MODEL") or (self.meta or {}).get("active_model")
        if wanted not in self.available:
            if wanted:
                print(f"Requested model '{wanted}' is not available; using the newest one instead.")
            wanted = self.available[-1] if self.available else None
        if wanted is None:
            raise RuntimeError("No trained model found in models/. Import dualsight_trained_model.zip first (see README.md).")
        self.kind = wanted
        self.label = MODEL_LABELS[wanted]

        self.keras_model = None
        self.price_scale = 1.0
        if self.kind == "neural_hybrid":
            import tensorflow as tf
            self.tf = tf
            self.keras_model = tf.keras.models.load_model(NEURAL_PATH)
            self.price_scale = float(self.meta.get("price_scale", 1.0))
        else:
            import joblib
            bundle = joblib.load(BOOSTING_PATH)
            self.gbm = bundle["model"]
            self.gb_numeric = bundle["numeric_features"]
            self.price_scale = float(bundle["price_scale"])
            store = np.load(FEATURES_PATH)
            self.img_pca = store["img_pca"]
            self.pca_mean = store["pca_mean"]
            self.pca_components = store["pca_components"]
            self.feature_row = {name: i for i, name in enumerate(store["image_file"])}
            self._tree_explainer = None
            self._extractor = None
            self._extractor_lock = threading.Lock()
            # Load the ImageNet feature extractor in the background (first run downloads ~94 MB)
            threading.Thread(target=self._get_extractor, daemon=True).start()

        if self.meta is not None:
            self.numeric = self.meta["numeric_features"]
            self.med = np.array(self.meta["feature_median"], dtype="float32")
            self.mean = np.array(self.meta["feature_mean"], dtype="float32")
            self.scale = np.array(self.meta["feature_scale"], dtype="float32")
        self._background = None
        print(f"--> Active model: {self.label}  (available: {', '.join(self.available)})")

    # ------------------------------------------------------------------ inputs
    def row(self, i):
        return self.df.iloc[i]

    def image_file(self, i):
        return os.path.basename(str(self.row(i)["rgb_path"]).replace("\\", "/"))

    def raw_numbers(self, i):
        vals = pd.to_numeric(self.row(i)[self.numeric], errors="coerce").to_numpy("float32")
        return np.where(np.isnan(vals), self.med, vals)

    def scaled_numbers(self, i):
        return (self.raw_numbers(i) - self.mean) / self.scale

    def _neural_log(self, imgs, tabs):
        return self.keras_model.predict([imgs, tabs], batch_size=32, verbose=0).reshape(-1).astype("float64")

    def _boosting_x(self, i, img_features=None):
        if img_features is None:
            j = self.feature_row.get(self.image_file(i))
            if j is None:
                raise ValueError("No precomputed image features for this property.")
            img_features = self.img_pca[j]
        return np.concatenate([self.raw_numbers(i), img_features]).reshape(1, -1)

    # ------------------------------------------------------------------ predict
    def predict_index(self, i):
        if self.kind == "boosting_hybrid":
            log = float(self.gbm.predict(self._boosting_x(i))[0])
        else:
            img = caffe_preprocess(pil_image(self.row(i)["rgb_path"]))[None]
            tab = self.scaled_numbers(i)[None]
            log = float(self._neural_log(img, tab)[0])
        return max(0.0, float(np.expm1(log)) * self.price_scale)

    # ------------------------------------------------------------------ explain
    def _background_images(self):
        """Typical satellite patches used as the 'average image' reference."""
        if self._background is None:
            sample = self.df.sample(n=min(N_BACKGROUND, len(self.df)), random_state=7)
            self._background = np.stack([caffe_preprocess(pil_image(p)) for p in sample["rgb_path"]])
        return self._background

    def _get_extractor(self):
        with self._extractor_lock:
            if self._extractor is None:
                try:
                    import tensorflow as tf
                    self.tf = tf
                    self._extractor = tf.keras.applications.ResNet50(include_top=False, pooling="avg", weights="imagenet",
                                                                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
                except Exception as e:  # e.g. offline: heatmap is skipped, everything else works
                    print(f"Image feature extractor unavailable ({e}); occlusion maps disabled.")
                    self._extractor = False
        return self._extractor

    def explain_index(self, i):
        """Returns base price, dollar steps per factor, final price, method notes and an occlusion grid."""
        if self.kind == "boosting_hybrid":
            base_log, contribs, method = self._explain_boosting(i)
        else:
            base_log, contribs, method = self._explain_neural(i)
        base_price, steps, final_price = to_dollar_steps(base_log, contribs, self.price_scale)
        grid, grid_note = self._occlusion(i)
        return {"model": self.label, "model_kind": self.kind, "method": method,
                "base_price": base_price, "steps": steps, "predicted_price": max(0.0, final_price),
                "occlusion": grid, "occlusion_note": grid_note}

    def _explain_boosting(self, i):
        import shap
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self.gbm)
        x = self._boosting_x(i)
        sv = np.asarray(self._tree_explainer.shap_values(x)).reshape(-1)
        base = float(np.asarray(self._tree_explainer.expected_value).reshape(-1)[0])
        k = len(self.gb_numeric)
        contribs = [(FEATURE_LABELS.get(f, f), float(v)) for f, v in zip(self.gb_numeric, sv[:k])]
        contribs.append(("Satellite image", float(sv[k:].sum())))
        return base, contribs, ("Exact TreeSHAP values from the gradient-boosting model. The starting point is the "
                                "model's average prediction over its training data.")

    def _explain_neural(self, i):
        tf = self.tf
        img = caffe_preprocess(pil_image(self.row(i)["rgb_path"]))[None]
        tab = self.scaled_numbers(i)[None]
        tab_ref = ((self.med - self.mean) / self.scale)[None]
        bg = self._background_images()

        # Reference: typical images with typical (median) numbers
        base = float(self._neural_log(bg, np.repeat(tab_ref, len(bg), axis=0)).mean())
        with_img = float(self._neural_log(img, tab_ref)[0])

        # Integrated gradients for the numbers, image held fixed
        alphas = tf.reshape(tf.linspace(0.0, 1.0, IG_STEPS + 1), (-1, 1))
        path = tf.constant(tab_ref) + alphas * tf.constant(tab - tab_ref)
        imgs = tf.repeat(tf.constant(img), IG_STEPS + 1, axis=0)
        with tf.GradientTape() as tape:
            tape.watch(path)
            out = self.keras_model([imgs, path], training=False)
        grads = tape.gradient(out, path).numpy()
        avg_grads = (grads[:-1] + grads[1:]).mean(axis=0) / 2.0
        ig = ((tab - tab_ref) * avg_grads).reshape(-1)
        full = float(self._neural_log(img, tab)[0])  # same call as predict_index, so both show one price

        # Integrated gradients is a numerical integral, so it can miss the exact total slightly.
        # Tiny gaps are spread across the numbers in proportion to their size; a visible gap gets its own row.
        gap = (full - with_img) - float(ig.sum())
        if abs(gap) <= 0.002 and np.abs(ig).sum() > 0:
            ig = ig + gap * np.abs(ig) / np.abs(ig).sum()
            gap = 0.0
        contribs = [("Satellite image", with_img - base)]
        contribs += [(FEATURE_LABELS.get(f, f), float(v)) for f, v in zip(self.numeric, ig)]
        if gap != 0.0:
            contribs.append(("Approximation error", gap))
        return base, contribs, ("Integrated gradients for each number (from the median property to this one, image "
                                "fixed). The image's effect is measured against 16 typical satellite patches.")

    def _occlusion(self, i):
        """Change in log(price) when each square of the patch is greyed out (positive = that area raises the price)."""
        g = OCCLUSION_GRID
        cell = IMAGE_SIZE // g
        if self.kind == "boosting_hybrid":
            extractor = self._get_extractor()
            if not extractor:
                return None, "Occlusion map unavailable: the ImageNet feature extractor could not be loaded."
            tf = self.tf
            raw = tf.io.decode_png(tf.io.read_file(str(self.row(i)["rgb_path"])), channels=3)
            img = tf.clip_by_value(tf.image.resize(raw, [IMAGE_SIZE, IMAGE_SIZE], antialias=True), 0, 255).numpy()
            img = caffe_preprocess(img)
        else:
            img = caffe_preprocess(pil_image(self.row(i)["rgb_path"]))
        batch = [img]
        for r in range(g):
            for c in range(g):
                o = img.copy()
                o[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell, :] = 0.0  # ImageNet mean colour
                batch.append(o)
        batch = np.stack(batch)

        note = f"{g}x{g} grid; each square is greyed out in turn and the price re-predicted."
        if self.kind == "boosting_hybrid":
            deep = extractor.predict(batch, batch_size=32, verbose=0)
            pcs = (deep - self.pca_mean) @ self.pca_components.T
            stored = self.img_pca[self.feature_row[self.image_file(i)]]
            drift = float(np.abs(pcs[0] - stored).max() / (np.abs(stored).max() + 1e-6))
            if drift > 0.05:
                note += f" Note: live image features differ from Kaggle's by {drift * 100:.0f}%, so the map is approximate."
            x = np.hstack([np.repeat(self.raw_numbers(i)[None], len(batch), axis=0), pcs])
            logs = self.gbm.predict(x)
        else:
            tabs = np.repeat(self.scaled_numbers(i)[None], len(batch), axis=0)
            logs = self._neural_log(batch, tabs)
        grid = (logs[0] - logs[1:]).reshape(g, g)
        return grid, note

    # ------------------------------------------------------------------ held-out results
    def held_out_predictions(self):
        """(actual, predicted, image_file) for the test set, if available for the active model."""
        if not os.path.exists(TEST_PREDS_PATH):
            return None
        preds = pd.read_csv(TEST_PREDS_PATH)
        if self.kind not in preds.columns:
            return None
        preds = preds.dropna(subset=[self.kind])
        return preds["actual"].to_numpy(), preds[self.kind].to_numpy(), preds["image_file"].to_numpy()
