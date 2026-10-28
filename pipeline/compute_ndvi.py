import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine


OUTPUTS_DIR = Path("data/outputs")


def find_most_recent(pattern: str) -> Optional[Path]:
    files = sorted(OUTPUTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_metadata() -> Tuple[Tuple[float, float, float, float], Tuple[int, int]]:
    meta_file = find_most_recent("sentinel2_metadata_*.json")
    if not meta_file:
        raise SystemExit("No metadata JSON found in 'outputs'. Run the download script first.")
    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    bbox = meta.get("bbox_wgs84")
    size = meta.get("size_px") or {}
    if not bbox or len(bbox) != 4:
        raise SystemExit("Metadata missing 'bbox_wgs84' with 4 values.")
    width, height = int(size.get("width", 0)), int(size.get("height", 0))
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), (width, height)


def compute_transform(bbox: Tuple[float, float, float, float], width: int, height: int) -> Affine:
    lon_min, lat_min, lon_max, lat_max = bbox
    if width <= 0 or height <= 0:
        raise SystemExit("Invalid width/height in metadata.")
    x_res = (lon_max - lon_min) / float(width)
    y_res = (lat_max - lat_min) / float(height)
    # GDAL Affine: from upper-left corner; y pixel size must be negative for north-up images
    return Affine.translation(lon_min, lat_max) * Affine.scale(x_res, -y_res)


def read_band_from_png(path: Path, band_index_1_based: int) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(band_index_1_based)
    return band


def main() -> None:
    if not OUTPUTS_DIR.exists():
        raise SystemExit("'outputs' directory not found.")

    rgb_path = find_most_recent("sentinel2_rgb_*.png")
    nir_path = find_most_recent("sentinel2_nir_*.png")
    if not rgb_path or not nir_path:
        raise SystemExit("RGB or NIR PNG not found in 'outputs'. Run the download script first.")

    # Read Red from RGB (channel 1 in 1-based indexing; saved as [R,G,B])
    red = read_band_from_png(rgb_path, 1).astype("float32")
    nir = read_band_from_png(nir_path, 1).astype("float32")

    if red.shape != nir.shape:
        raise SystemExit(f"Mismatched shapes: red {red.shape} vs nir {nir.shape}")

    # Scale from 0..255 to 0..1
    red /= 255.0
    nir /= 255.0

    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / denom
    ndvi = np.where(denom == 0.0, 0.0, ndvi).astype("float32")

    # Georeference using metadata (bbox in EPSG:4326 and pixel size)
    bbox, (width_m, height_m) = load_metadata()
    transform = compute_transform(bbox, ndvi.shape[1], ndvi.shape[0])

    out_path = OUTPUTS_DIR / "austin_ndvi.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": ndvi.shape[0],
        "width": ndvi.shape[1],
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ndvi, 1)

    print(out_path)


if __name__ == "__main__":
    main()


