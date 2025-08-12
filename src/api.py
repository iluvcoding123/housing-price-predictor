"""
FastAPI app for serving the Ames Housing price predictor.

Endpoints
- GET  /health   -> quick liveness check
- POST /predict  -> JSON body with raw house features; returns predicted sale prices

How to run (from project root):
    pip install fastapi uvicorn
    uvicorn src.api:app --reload --port 8000

Example request:
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
            "data": [
              {"Gr Liv Area": 1800, "Total Bsmt SF": 900, "Garage Cars": 2, "Overall Qual": 7, "Year Built": 2005, "Yr Sold": 2010}
            ]
          }'
"""
from __future__ import annotations

from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# ----------------------
# Configurable artifact paths (relative to repo root / api file)
# Adjust if your layout differs
# ----------------------
MODEL_PATH = "models/xgboost_model.pkl"
SCALER_PATH = "models/scaler.pkl"  # keep if model trained on scaled features
FEATS_PATH = "data/feature_names.pkl"

# ----------------------
# Pydantic request schema
# ----------------------
class HouseRow(BaseModel):
    Gr_Liv_Area: float = Field(..., alias="Gr Liv Area")
    Total_Bsmt_SF: float = Field(..., alias="Total Bsmt SF")
    Garage_Cars: float = Field(..., alias="Garage Cars")
    Overall_Qual: float = Field(..., alias="Overall Qual")
    Year_Built: int = Field(..., alias="Year Built")
    Yr_Sold: int = Field(..., alias="Yr Sold")

    # Optional sanity checks
    @validator("Garage_Cars")
    def non_negative_garage(cls, v):
        if v < 0:
            raise ValueError("Garage Cars must be >= 0")
        return v

    @validator("Overall_Qual")
    def qual_range(cls, v):
        # Ames overall quality is typically 1..10
        if not (1 <= v <= 10):
            raise ValueError("Overall Qual must be between 1 and 10")
        return v

class PredictRequest(BaseModel):
    data: List[HouseRow]

class PredictResponseRow(BaseModel):
    predicted_sale_price: float

class PredictResponse(BaseModel):
    predictions: List[PredictResponseRow]

# ----------------------
# App init & artifact loading
# ----------------------
app = FastAPI(title="Ames Housing Price Predictor", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Raise at startup so misconfigured deploys fail fast
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

try:
    # Scaler may or may not exist depending on training choices
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

try:
    feature_names: List[str] = joblib.load(FEATS_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load feature names at {FEATS_PATH}: {e}")

RAW_COLS = [
    "Gr Liv Area",
    "Total Bsmt SF",
    "Garage Cars",
    "Overall Qual",
    "Year Built",
    "Yr Sold",
]

# ----------------------
# Helpers
# ----------------------

def _rows_to_dataframe(rows: List[HouseRow]) -> pd.DataFrame:
    # Convert validated Pydantic objects back to a DataFrame with original column names
    raw_dicts = []
    for r in rows:
        raw_dicts.append(
            {
                "Gr Liv Area": r.Gr_Liv_Area,
                "Total Bsmt SF": r.Total_Bsmt_SF,
                "Garage Cars": r.Garage_Cars,
                "Overall Qual": r.Overall_Qual,
                "Year Built": r.Year_Built,
                "Yr Sold": r.Yr_Sold,
            }
        )
    return pd.DataFrame(raw_dicts, columns=RAW_COLS)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering (must match training exactly)
    df = df.copy()
    df["TotalSF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]

    # Select columns in the exact order used during training
    try:
        X = df[feature_names]
    except KeyError as e:
        missing = [c for c in feature_names if c not in df.columns]
        raise HTTPException(status_code=400, detail=f"Missing engineered columns: {missing}") from e

    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_names)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Scaling failed: {e}") from e

    return X


# ----------------------
# Routes
# ----------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True, "scaler": scaler is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        df_raw = _rows_to_dataframe(req.data)
        X = _preprocess(df_raw)
        preds = model.predict(X)
    except HTTPException:
        # re-raise structured API errors
        raise
    except Exception as e:
        # Convert any other exception into a 400 for client clarity
        raise HTTPException(status_code=400, detail=str(e))

    # Build response (float conversion for JSON serializability)
    out = [PredictResponseRow(predicted_sale_price=float(p)) for p in preds]
    return PredictResponse(predictions=out)
