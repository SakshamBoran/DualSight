import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("data/austin_properties_cleaned.csv")
    if not input_path.exists():
        raise SystemExit("data/austin_properties_cleaned.csv not found. Run clean_austin_data.py first.")

    df = pd.read_csv(input_path, usecols=["latitude", "longitude"])
    if df.empty:
        raise SystemExit("Input CSV has no rows after reading required columns.")

    # Ensure numeric and drop non-numeric rows if any slipped through
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No valid numeric latitude/longitude values found.")

    lat_min = float(df["latitude"].min())
    lat_max = float(df["latitude"].max())
    lon_min = float(df["longitude"].min())
    lon_max = float(df["longitude"].max())

    buffer_deg = 0.01
    lat_min_b = max(-90.0, lat_min - buffer_deg)
    lon_min_b = max(-180.0, lon_min - buffer_deg)
    lat_max_b = min(90.0, lat_max + buffer_deg)
    lon_max_b = min(180.0, lon_max + buffer_deg)

    # Print in the exact order required by download_sentinel.py: lat_min lon_min lat_max lon_max
    # Single line, space-separated for easy copy-paste
    print(f"{lat_min_b} {lon_min_b} {lat_max_b} {lon_max_b}")


if __name__ == "__main__":
    main()


