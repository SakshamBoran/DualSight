import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

try:
    from sentinelhub import (
        BBox,
        CRS,
        MimeType,
        SentinelHubRequest,
        DataCollection,
        bbox_to_dimensions,
        SHConfig,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'sentinelhub'. Install requirements first: pip install -r requirements.txt"
    ) from exc


RGB_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "CLM", "dataMask"] }],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  // True color in RGB order (B04=R, B03=G, B02=B)
  return [sample.B04, sample.B03, sample.B02];
}
"""

NIR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B08", "CLM", "dataMask"] }],
    output: { bands: 1 }
  };
}
function evaluatePixel(sample) {
  // NIR channel grayscale
  return [sample.B08];
}
"""


def parse_bbox(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> Tuple[BBox, Tuple[int, int]]:
    """Create WGS84 bbox (lon/lat order) and compute image size in pixels.

    We accept inputs in (lat_min, lon_min, lat_max, lon_max) and convert to (lon_min, lat_min, lon_max, lat_max).
    """
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Invalid bbox: ensure lat_min < lat_max and lon_min < lon_max")

    wgs84_bbox = BBox(bbox=(lon_min, lat_min, lon_max, lat_max), crs=CRS.WGS84)
    # Choose a reasonable resolution: 10 m per pixel for Sentinel-2 bands at 10 m
    width, height = bbox_to_dimensions(wgs84_bbox, resolution=22)
    # Guard against extremely large downloads
    if width * height > 4000 * 4000:
        raise ValueError(
            f"Requested area is too large at 10 m resolution ({width}x{height} px). Use a smaller bbox or lower resolution."
        )
    return wgs84_bbox, (width, height)


def get_time_interval(months: int) -> Tuple[str, str]:
    # Fixed interval for full year 2020, ignoring 'months' argument per request
    return ("2020-01-01", "2020-12-31")


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def configure_sentinelhub() -> SHConfig:
    cfg = SHConfig()

    # Credentials come from environment variables (never hardcode them).
    #   Windows (PowerShell): $env:SH_CLIENT_ID="..."; $env:SH_CLIENT_SECRET="..."
    #   macOS/Linux:          export SH_CLIENT_ID=...; export SH_CLIENT_SECRET=...
    cfg.sh_client_id = os.environ.get("SH_CLIENT_ID", "")
    cfg.sh_client_secret = os.environ.get("SH_CLIENT_SECRET", "")

    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise SystemExit(
            "Sentinel Hub credentials are not set. "
            "Set the SH_CLIENT_ID and SH_CLIENT_SECRET environment variables "
            "with your own OAuth client from your Sentinel Hub dashboard."
        )
    return cfg


def request_image(
    bbox: BBox,
    size: Tuple[int, int],
    time_interval: Tuple[str, str],
    evalscript: str,
    max_cloud: int,
    config: SHConfig,
):
    return SentinelHubRequest(
        data_folder=None,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="mostRecent",
                other_args={
                    "processing": {},
                    "dataFilter": {
                        # Filter scenes by cloud coverage at catalog level
                        "maxCloudCoverage": max_cloud,
                    },
                },
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )


def save_raster(array: np.ndarray, path: Path) -> None:
    # Expect shapes: HxWx3 (RGB) or HxWx1 (NIR). Convert to uint16 and save as PNG for convenience.
    arr = array
    # Normalize assuming Sentinel-2 reflectance scaled 0..1 in Process API; scale to 0..10000 if needed
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray(arr)
    img.save(path)


def write_metadata(path: Path, meta: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Sentinel-2 true-color (RGB) and NIR imagery using Sentinel Hub for a given lat/lon bbox."
        )
    )
    parser.add_argument("lat_min", type=float, help="Minimum latitude")
    parser.add_argument("lon_min", type=float, help="Minimum longitude")
    parser.add_argument("lat_max", type=float, help="Maximum latitude")
    parser.add_argument("lon_max", type=float, help="Maximum longitude")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months back from today to search (default: 12)",
    )
    parser.add_argument(
        "--max-cloud",
        type=int,
        default=10,
        help="Maximum cloud coverage percentage per scene (default: 10)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/outputs",
        help="Output directory (default: data/outputs)",
    )
    args = parser.parse_args()

    bbox, size = parse_bbox(args.lat_min, args.lon_min, args.lat_max, args.lon_max)
    time_interval = get_time_interval(args.months)
    out_dir = ensure_output_dir(args.out)

    config = configure_sentinelhub()

    # Build requests
    rgb_request = request_image(
        bbox=bbox,
        size=size,
        time_interval=time_interval,
        evalscript=RGB_EVALSCRIPT,
        max_cloud=args.max_cloud,
        config=config,
    )
    nir_request = request_image(
        bbox=bbox,
        size=size,
        time_interval=time_interval,
        evalscript=NIR_EVALSCRIPT,
        max_cloud=args.max_cloud,
        config=config,
    )

    # Download
    print("Requesting RGB image ...")
    rgb_data = rgb_request.get_data()[0]  # HxWx3 float in [0,1]
    print("Requesting NIR image ...")
    nir_data = nir_request.get_data()[0]  # HxWx1 float in [0,1]

    # Save images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb_path = out_dir / f"sentinel2_rgb_{timestamp}.png"
    nir_path = out_dir / f"sentinel2_nir_{timestamp}.png"
    save_raster(rgb_data, rgb_path)
    save_raster(nir_data, nir_path)

    # Save metadata
    meta = {
        "bbox_wgs84": list(bbox),
        "size_px": {"width": size[0], "height": size[1]},
        "time_interval": {"from": time_interval[0], "to": time_interval[1]},
        "data_collection": "SENTINEL2_L2A",
        "max_cloud": args.max_cloud,
        "notes": "Images are mosaicked by most recent acquisition within the time interval.",
    }
    write_metadata(out_dir / f"sentinel2_metadata_{timestamp}.json", meta)

    print("Saved:")
    print(f" - RGB: {rgb_path}")
    print(f" - NIR: {nir_path}")


if __name__ == "__main__":
    main()


