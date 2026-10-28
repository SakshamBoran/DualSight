import math
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

INPUT_CSV = Path("data/austin_properties_cleaned.csv")
OUTPUT_DIR = Path("data/image_patches")
RGB_GLOB = "data/outputs/sentinel2_rgb_*.png"
NDVI_PATH = Path("data/outputs/austin_ndvi.tif")

def lonlat_to_pixel(dataset: rasterio.io.DatasetReader, lon: float, lat: float) -> Tuple[int, int]:
    """Convert (lon, lat) to pixel coordinates in the raster."""
    x, y = dataset.index(lon, lat)
    return int(x), int(y)

def crop_window(center_x: int, center_y: int, width: int, height: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Compute a valid crop window centered on (x, y)."""
    half_w, half_h = width // 2, height // 2
    x0 = max(0, center_x - half_w)
    y0 = max(0, center_y - half_h)
    x1 = min(img_w, x0 + width)
    y1 = min(img_h, y0 + height)
    if x1 - x0 < width:
        x0 = max(0, x1 - width)
    if y1 - y0 < height:
        y0 = max(0, y1 - height)
    return x0, y0, x1, y1

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit("❌ Missing austin_properties_cleaned.csv. Run clean_austin_data.py first.")

    df = pd.read_csv(INPUT_CSV)
    print(f"📄 Loaded {len(df)} cleaned property rows")

    rgb_candidates = sorted(Path().glob(RGB_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    if not rgb_candidates:
        raise SystemExit("❌ No RGB Sentinel-2 image found in 'outputs/'. Run download_sentinel.py first.")
    rgb_path = rgb_candidates[0]
    print(f"🛰️ Using RGB image: {rgb_path.name}")

    if not NDVI_PATH.exists():
        raise SystemExit("❌ NDVI GeoTIFF not found in outputs/. Run compute_ndvi.py first.")
    print(f"🌿 Using NDVI file: {NDVI_PATH.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(rgb_path) as rgb_ds, rasterio.open(NDVI_PATH) as ndvi_ds:
        img_w, img_h = rgb_ds.width, rgb_ds.height
        if (img_w, img_h) != (ndvi_ds.width, ndvi_ds.height):
            raise SystemExit("❌ RGB and NDVI dimensions do not match!")

        rgb_paths, ndvi_paths = [], []

        for idx, row in df.iterrows():
            lat = float(row["latitude"]) if not pd.isna(row["latitude"]) else None
            lon = float(row["longitude"]) if not pd.isna(row["longitude"]) else None
            if lat is None or lon is None:
                rgb_paths.append("")
                ndvi_paths.append("")
                continue

            px, py = lonlat_to_pixel(ndvi_ds, lon, lat)
            x0, y0, x1, y1 = crop_window(px, py, 256, 256, img_w, img_h)
            if x1 - x0 < 256 or y1 - y0 < 256:
                rgb_paths.append("")
                ndvi_paths.append("")
                continue

            rgb_window = rasterio.windows.Window.from_slices((y0, y1), (x0, x1))
            rgb_arr = rgb_ds.read(indexes=(1, 2, 3), window=rgb_window)
            rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
            rgb_img = Image.fromarray(rgb_arr.astype(np.uint8))

            ndvi_arr = ndvi_ds.read(1, window=rgb_window)
            ndvi_vis = ((ndvi_arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            ndvi_img = Image.fromarray(ndvi_vis, mode="L")

            rgb_out = OUTPUT_DIR / f"property_{idx}_rgb.png"
            ndvi_out = OUTPUT_DIR / f"property_{idx}_ndvi.png"
            rgb_img.save(rgb_out)
            ndvi_img.save(ndvi_out)

            rgb_paths.append(str(rgb_out))
            ndvi_paths.append(str(ndvi_out))

    df["rgb_path"] = rgb_paths
    df["ndvi_path"] = ndvi_paths
    df.to_csv("data/austin_master_dataset.csv", index=False)
    print(f"✅ Saved austin_master_dataset.csv with {len(df)} rows")
    print(f"🖼️ Image patches saved in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
