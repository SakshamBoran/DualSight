from pathlib import Path
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

CSV_PATH = Path("data/austin_master_dataset.csv")

TARGET_COLUMN = "latestPrice"

def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    df = df[df["rgb_path"].astype(str).str.len() > 0].copy()
    df = df[df["rgb_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    x_paths = df["rgb_path"].astype(str).values
    y = df[TARGET_COLUMN].astype(np.float32).values

    from sklearn.model_selection import train_test_split
    x_paths_train, x_paths_val, y_train, y_val = train_test_split(
        x_paths, y, test_size=0.2, random_state=42
    )
    return (x_paths_train, y_train), (x_paths_val, y_val)

def make_datasets(train_tuple, val_tuple, batch_size=32):
    x_paths_train, y_train = train_tuple
    x_paths_val, y_val = val_tuple

    def _read_preprocess_image(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, [128, 128])
        img = preprocess_input(img)
        return img

    def _augment_image(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        return img

    def map_train(path, label):
        img = _read_preprocess_image(path)
        img = _augment_image(img)
        return img, tf.math.log1p(label)

    def map_val(path, label):
        img = _read_preprocess_image(path)
        return img, tf.math.log1p(label)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((x_paths_train, y_train))
    train_ds = train_ds.shuffle(len(x_paths_train)).map(map_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_paths_val, y_val))
    val_ds = val_ds.map(map_val, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds
