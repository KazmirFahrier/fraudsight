"""Phase 9 — the unit test that guards against look-ahead leakage.

A scoring row's features must NOT change when future transactions are appended.
If this test fails, a feature is peeking into the future.
"""
import pandas as pd
from src.features import add_point_in_time_features


def _frame(extra=False):
    rows = [
        ("c1", "2026-01-01 10:00", 100.0),
        ("c1", "2026-01-01 11:00", 50.0),   # <-- scoring row
    ]
    if extra:
        rows.append(("c1", "2026-01-01 12:00", 999.0))  # future txn
    df = pd.DataFrame(rows, columns=["card_id", "ts", "amount"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def test_future_rows_do_not_change_past_features():
    base = add_point_in_time_features(_frame(extra=False))
    withf = add_point_in_time_features(_frame(extra=True))
    row_base = base[base["ts"] == pd.Timestamp("2026-01-01 11:00")].iloc[0]
    row_withf = withf[withf["ts"] == pd.Timestamp("2026-01-01 11:00")].iloc[0]
    assert row_base["spend_prior"] == row_withf["spend_prior"]
    assert row_base["count_prior"] == row_withf["count_prior"]


def test_current_row_excluded():
    # At 11:00 only the 10:00 txn (100.0) is prior; the current 50.0 is excluded.
    out = add_point_in_time_features(_frame(extra=False))
    row = out[out["ts"] == pd.Timestamp("2026-01-01 11:00")].iloc[0]
    assert row["spend_prior"] == 100.0
    assert row["count_prior"] == 1
