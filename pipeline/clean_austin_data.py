import os
import pandas as pd
from pathlib import Path

# Essential columns for the dataset
ESSENTIAL_COLUMNS = [
    "latitude", "longitude", "latestPrice", "livingAreaSqFt", 
    "yearBuilt", "numOfBedrooms", "numOfBathrooms", "streetAddress", 
    "zipcode", "avgSchoolRating", "lotSizeSqFt"
]

# Required numeric features for the model
NUMERIC_FEATURES = [
    "livingAreaSqFt", "yearBuilt", "numOfBedrooms", 
    "numOfBathrooms", "avgSchoolRating", "lotSizeSqFt"
]

def main() -> None:
    print("📥 Loading Austin raw data...")
    df = pd.read_csv("data/austin_data.csv")

    # Keep only essential columns
    print("📊 Filtering to essential columns...")
    df = df[[c for c in ESSENTIAL_COLUMNS if c in df.columns]]

    # Clean data
    print("🧹 Cleaning data...")
    df = df.dropna()
    df = df[(df["latestPrice"] <= 2_000_000) & (df["livingAreaSqFt"] >= 500)]

    # Save the cleaned dataset for satellite patch generation
    output_file = "data/austin_properties_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned dataset as {output_file} with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
