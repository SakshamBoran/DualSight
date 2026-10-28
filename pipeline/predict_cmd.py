"""Predict (and optionally explain) one property's price from the terminal.

Run from the project root:
    python pipeline/predict_cmd.py 123                  # active model
    python pipeline/predict_cmd.py 123 --explain        # plus a breakdown of why
    python pipeline/predict_cmd.py 123 --model neural_hybrid
Models: neural_hybrid, boosting_hybrid (whichever you have in models/).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

CSV_PATH = "data/austin_master_dataset.csv"


def main():
    args = sys.argv[1:]
    if "--model" in args:
        os.environ["DUALSIGHT_MODEL"] = args[args.index("--model") + 1]
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--model")]
    if not positional:
        print(__doc__)
        sys.exit(1)
    try:
        idx = int(positional[0])
    except ValueError:
        print("First argument must be an integer row index")
        sys.exit(1)

    from model_service import ModelService  # imported late so DUALSIGHT_MODEL is respected

    df = pd.read_csv(CSV_PATH)
    if idx < 0 or idx >= len(df):
        print(f"Index {idx} out of range (dataset has {len(df)} rows).")
        sys.exit(1)
    if not os.path.exists(str(df.iloc[idx]["rgb_path"])):
        print(f"Image not found: {df.iloc[idx]['rgb_path']}")
        sys.exit(1)

    service = ModelService(df)
    row = df.iloc[idx]
    price = service.predict_index(idx)
    actual = row.get("latestPrice")
    print(f"\n=== {service.label}: property {idx}, {row.get('streetAddress', '')} {row.get('zipcode', '')} ===")
    print(f"Predicted price: ${price:,.0f}")
    if actual is not None and not pd.isna(actual):
        print(f"Actual price:    ${float(actual):,.0f}   (error {price - float(actual):+,.0f}, "
              f"{(price - float(actual)) / float(actual) * 100:+.1f}%)")

    if "--explain" in args:
        exp = service.explain_index(idx)
        print(f"\nWhy: starting from a typical property at ${exp['base_price']:,.0f}")
        for label, dollars, _ in exp["steps"]:
            print(f"  {label:<20s} {'+' if dollars >= 0 else '-'}${abs(dollars):>12,.0f}")
        print(f"  {'= prediction':<20s}  ${exp['predicted_price']:>12,.0f}")
        print(f"Method: {exp['method']}")
        if exp["occlusion"] is not None:
            g = exp["occlusion"]
            r, c = divmod(int(abs(g).argmax()), g.shape[1])
            print(f"Most influential image area: row {r + 1}, column {c + 1} of {g.shape[0]}x{g.shape[1]} grid "
                  f"({'raises' if g[r, c] > 0 else 'lowers'} the price by {abs((2.718281828 ** g[r, c] - 1) * 100):.1f}%)")
    print()


if __name__ == "__main__":
    main()
