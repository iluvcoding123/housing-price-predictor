#!/usr/bin/env python3
import json
import argparse
import sys
import pandas as pd
import joblib

# --- load artifacts once ---
MODEL_PATH = "../models/xgboost_model.pkl"
SCALER_PATH = "../models/scaler.pkl"          # keep if XGB was trained on scaled features
FEATS_PATH  = "../data/feature_names.pkl"     # saved during training

model = joblib.load(MODEL_PATH)
try:
    scaler = joblib.load(SCALER_PATH)      # remove try/except if you always scale
except Exception:
    scaler = None
feature_names = joblib.load(FEATS_PATH)    # do not hardcode

REQUIRED_RAW = ["Gr Liv Area", "Total Bsmt SF", "Garage Cars", "Overall Qual", "Year Built", "Yr Sold"]

def validate_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # basic type coercion
    numeric_cols = ["Gr Liv Area", "Total Bsmt SF", "Garage Cars", "Overall Qual", "Year Built", "Yr Sold"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[numeric_cols].isna().any().any():
        bad = df.index[df[numeric_cols].isna().any(axis=1)].tolist()
        raise ValueError(f"Found non-numeric values in numeric fields at rows: {bad}")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_and_cast(df.copy())
    # feature engineering (must match training)
    df["TotalSF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
    X = df[feature_names]                      # correct order/columns
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return X

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    X = preprocess(df)
    preds = model.predict(X)
    out = df.copy()
    out["Predicted Sale Price"] = preds
    return out[["Predicted Sale Price"]]

def main():
    ap = argparse.ArgumentParser(description="Predict house SalePrice from JSON/CSV input.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="Path to a JSON file containing a list of rows or a single object.")
    src.add_argument("--csv", help="Path to a CSV file.")
    src.add_argument("--row", help="Inline JSON for a single row.")
    ap.add_argument("--save", help="Optional path to save predictions CSV.")
    args = ap.parse_args()

    # load input
    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = pd.DataFrame([json.loads(args.row)])

    # predict
    try:
        preds = predict_df(df)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # output
    if args.save:
        preds.to_csv(args.save, index=False)
        print(f"Saved predictions -> {args.save}")
    else:
        # pretty print
        for i, v in enumerate(preds["Predicted Sale Price"].tolist()):
            print(f"Row {i}: ${v:,.2f}")

if __name__ == "__main__":
    main()