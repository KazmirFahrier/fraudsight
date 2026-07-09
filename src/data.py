"""Data loading for FraudSight.

Two entry points:
- load_real(): read the Kaggle creditcard.csv if present.
- make_synthetic(): generate a realistic, time-ordered, heavily imbalanced
  transaction stream so the whole pipeline runs WITHOUT any download. Fraud is
  injected with signal (unusual amount + elevated prior-velocity) plus noise,
  so the modeling task is non-trivial but learnable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic(n: int = 60_000, n_cards: int = 3_000,
                   fraud_rate: float = 0.006, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    card_id = rng.integers(0, n_cards, n)
    # transactions arrive over ~30 days, sorted by time
    ts = np.sort(rng.uniform(0, 30 * 24 * 3600, n))
    base = pd.Timestamp("2026-01-01")
    ts = base + pd.to_timedelta(ts, unit="s")
    amount = rng.lognormal(mean=3.2, sigma=1.0, size=n)  # skewed, like real spend

    df = pd.DataFrame({"card_id": card_id, "ts": ts, "amount": amount})
    df = df.sort_values("ts").reset_index(drop=True)

    # a latent velocity driver: how many prior txns this card has already made
    prior_txns = df.groupby("card_id").cumcount().to_numpy()

    # Fraud propensity with STRONG, learnable structure (large amount at night on
    # high-velocity cards) plus noise so it is not perfectly separable.
    hour = df["ts"].dt.hour.to_numpy()
    amt = df["amount"].to_numpy()
    signal = (4.0 * (amt > np.quantile(amt, 0.95)).astype(float)
              + 2.5 * ((hour < 5) | (hour > 22)).astype(float)
              + 2.0 * (prior_txns > 5).astype(float)
              + rng.normal(0, 0.5, n))

    # Calibrate the INTERCEPT (not a multiplicative squash) so the base rate hits
    # target while preserving the signal's discriminative power.
    def _rate(b):
        return (1 / (1 + np.exp(-(signal + b)))).mean()
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _rate(mid) > fraud_rate:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(signal + mid)))
    df["is_fraud"] = (rng.random(n) < p).astype(int)
    return df


def load_real(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the Kaggle credit-card dataset and normalize to the FraudSight schema."""
    df = pd.read_csv(path)
    # Kaggle: Time (sec since first txn), Amount, V1..V28, Class
    df = df.rename(columns={"Class": "is_fraud", "Amount": "amount"})
    df["ts"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df["Time"], unit="s")
    df["card_id"] = 0  # dataset has no card id; velocity features are global
    return df


def get_data(prefer_real: bool = True):
    import os
    if prefer_real and os.path.exists("data/creditcard.csv"):
        return load_real(), "kaggle"
    return make_synthetic(), "synthetic"
