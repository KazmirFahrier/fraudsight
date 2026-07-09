"""Phase 1-3+7 end-to-end: data -> leakage-safe features -> temporal split ->
model -> honest evaluation -> saved artifact + written report.

Run:  python -m src.train
Produces:  models/fraudsight_pipeline.joblib  and  reports/evaluation_report.md

Design guarantees (the things an auditor checks):
- Temporal split: the test set is strictly LATER in time than the train set.
- Preprocessing (impute+scale) is fit on TRAIN ONLY, inside a Pipeline, and
  saved WITH the model — so production applies identical transforms (no skew).
- Velocity features are point-in-time (current row excluded).
- Metrics are imbalance-appropriate (PR-AUC) and reported with a bootstrap CI.
- The operating threshold comes from a cost matrix, not the default 0.5.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from src.seed import seed_everything
from src.data import get_data
from src.features import add_point_in_time_features
from src.evaluate import bootstrap_pr_auc_ci, threshold_from_cost

ROOT = Path(__file__).resolve().parents[1]   # repo root, so paths work from anywhere

FEATURES = ["amount", "log_amount", "hour", "spend_prior", "count_prior",
            "txns_prior_lifetime"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_point_in_time_features(df, window="24h").sort_values("ts").reset_index(drop=True)
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["ts"].dt.hour
    # point-in-time-safe lifetime velocity: counts only strictly-prior rows
    df["txns_prior_lifetime"] = df.groupby("card_id").cumcount()
    return df


def temporal_split(df: pd.DataFrame, frac_train: float = 0.8):
    cut = int(len(df) * frac_train)
    return df.iloc[:cut], df.iloc[cut:]


def build_model(kind: str = "gbm") -> Pipeline:
    pre = ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), FEATURES)])
    if kind == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=300,
            l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> dict:
    seed_everything(42)
    raw, source = get_data(prefer_real=True)
    df = engineer(raw)
    train, test = temporal_split(df)
    assert train["ts"].max() <= test["ts"].min(), "temporal leak: train overlaps test in time"

    Xtr, ytr = train[FEATURES], train["is_fraud"].to_numpy()
    Xte, yte = test[FEATURES], test["is_fraud"].to_numpy()

    results = {}
    fitted = {}
    for kind in ["logistic", "gbm"]:
        model = build_model(kind)
        sw = compute_sample_weight("balanced", ytr)  # cost-sensitive for the GBM
        if kind == "gbm":
            model.fit(Xtr, ytr, clf__sample_weight=sw)
        else:
            model.fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        pr = average_precision_score(yte, score)
        mean, lo, hi = bootstrap_pr_auc_ci(yte, score, n_boot=1000)
        results[kind] = {"pr_auc": float(pr), "pr_auc_ci": [lo, hi]}
        fitted[kind] = (model, score)

    # trivial baseline for context (always-negative == base rate PR-AUC)
    base_rate = float(yte.mean())
    results["baseline_always_negative"] = {"pr_auc": base_rate}

    # pick best model, choose an operating threshold from a cost matrix
    best_kind = max(["logistic", "gbm"], key=lambda k: results[k]["pr_auc"])
    best_model, best_score = fitted[best_kind]
    t, cost = threshold_from_cost(yte, best_score, c_fn=500, c_fp=5)
    pred = (best_score >= t).astype(int)
    results["operating_point"] = {
        "model": best_kind, "threshold": round(t, 4),
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "expected_cost": float(cost),
    }
    results["meta"] = {
        "data_source": source, "n_train": len(train), "n_test": len(test),
        "train_fraud_rate": round(float(ytr.mean()), 5),
        "test_fraud_rate": round(base_rate, 5),
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best_model, ROOT / "models/fraudsight_pipeline.joblib")  # pipeline, not bare model
    write_report(results)
    print(json.dumps(results, indent=2))
    return results


def write_report(r: dict) -> None:
    m = r["meta"]; op = r["operating_point"]
    lines = [
        "# Evaluation report (auto-generated by src/train.py)",
        "",
        f"**Data source:** {m['data_source']}  ",
        f"**Train / test:** {m['n_train']:,} / {m['n_test']:,} transactions "
        f"(temporal split — test is strictly later in time)  ",
        f"**Fraud rate:** train {m['train_fraud_rate']:.3%}, test {m['test_fraud_rate']:.3%}",
        "",
        "## Headline metrics (held-out temporal test)",
        "",
        "| Model | PR-AUC | 95% bootstrap CI |",
        "|---|---|---|",
    ]
    for k in ["logistic", "gbm"]:
        ci = r[k]["pr_auc_ci"]
        lines.append(f"| {k} | {r[k]['pr_auc']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += [
        f"| always-negative baseline | {r['baseline_always_negative']['pr_auc']:.3f} | — |",
        "",
        "PR-AUC is the primary metric because the positive class is tiny; accuracy "
        "would exceed 99% for a model that never predicts fraud and is therefore "
        "meaningless here. Each interval is a 1,000-sample percentile bootstrap.",
        "",
        "## Operating point (cost-based threshold)",
        "",
        f"Chosen on the **{op['model']}** model with cost(missed fraud)=500, "
        f"cost(false block)=5:",
        "",
        f"- Threshold: **{op['threshold']}** (not the default 0.5)",
        f"- Precision: **{op['precision']:.3f}**, Recall: **{op['recall']:.3f}**",
        f"- Expected cost on test: **{op['expected_cost']:.0f}**",
        "",
        "## Limitations",
        "",
        "- Synthetic data (unless creditcard.csv present) — absolute numbers are "
        "illustrative; the *methodology* is what transfers.",
        "- No concept-drift handling yet; add rolling PR-AUC monitoring (Phase 9).",
        "- Seed-variance band not yet reported; re-run across seeds before claiming "
        "one model beats another (compute a paired bootstrap on the difference).",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/evaluation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
