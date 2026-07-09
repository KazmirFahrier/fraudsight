"""Leakage-safe, point-in-time feature engineering.

The cardinal rule: every rolling/aggregate feature for a transaction may use
ONLY rows strictly before that transaction. Scaling/encoding statistics are fit
on training data only, inside a sklearn Pipeline, never on the full frame.
"""
from __future__ import annotations
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def add_point_in_time_features(
    df: pd.DataFrame,
    id_col: str = "card_id",
    ts_col: str = "ts",
    amount_col: str = "amount",
    window: str = "24h",
) -> pd.DataFrame:
    """Prior-window spend/count per entity, EXCLUDING the current row.

    Uses a closed='left' rolling window so the current transaction never sees
    itself or the future. This is the pandas equivalent of the SQL predicate
    `p.ts < t.ts` and the Spark window `rangeBetween(-N, -1)`.
    """
    df = df.sort_values([id_col, ts_col]).copy()

    # Iterate groups explicitly (avoids the groupby.apply-on-grouping-column
    # deprecation and preserves all columns, including id_col).
    parts = []
    for _, g in df.groupby(id_col, sort=False):
        g = g.copy()
        r = g.rolling(window, on=ts_col, closed="left")
        g["spend_prior"] = r[amount_col].sum()
        g["count_prior"] = r[amount_col].count()
        parts.append(g)
    df = pd.concat(parts) if parts else df
    df[["spend_prior", "count_prior"]] = df[["spend_prior", "count_prior"]].fillna(0)
    return df


def build_preprocessor(numeric_features: list[str]) -> Pipeline:
    """Preprocessing that is fit on TRAIN ONLY when placed in a Pipeline.

    Never fit a scaler on the full dataset before splitting — that leaks test
    statistics into training. Keep preprocessing and model in ONE artifact so
    production applies identical transforms (no train/serve skew).
    """
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
