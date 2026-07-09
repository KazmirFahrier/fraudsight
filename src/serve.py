"""Phase 9 — model serving with FastAPI.

Loads the FULL pipeline artifact (preprocessing + model together, so there is no
train/serve skew) and exposes a /predict endpoint returning a calibrated-style
probability plus a decision at the deployed operating threshold.

Run:  uvicorn src.serve:app --reload
"""
from __future__ import annotations
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.train import ROOT, FEATURES

# Deployment operating threshold, taken from the cost-based analysis in
# reports/evaluation_report.md (~0.42). Override per environment with FRAUD_THRESHOLD.
THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.42"))

app = FastAPI(title="FraudSight")
_model = None


def get_model():
    global _model
    if _model is None:
        path = ROOT / "models/fraudsight_pipeline.joblib"
        if not path.exists():
            raise RuntimeError("Train first: python -m src.train")
        _model = joblib.load(path)
    return _model


class Transaction(BaseModel):
    amount: float
    log_amount: float
    hour: int
    spend_prior: float
    count_prior: float
    txns_prior_lifetime: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    X = pd.DataFrame([[getattr(txn, f) for f in FEATURES]], columns=FEATURES)
    proba = float(get_model().predict_proba(X)[:, 1][0])
    return {"fraud_probability": proba,
            "decision": "review" if proba >= THRESHOLD else "approve",
            "threshold": THRESHOLD}
