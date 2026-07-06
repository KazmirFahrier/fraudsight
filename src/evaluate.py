"""Phase 2 & 7 — evaluation, metrics under imbalance, and honest inference.

A point estimate is not a conclusion. Report PR-AUC with a bootstrap CI, choose
the operating threshold from a cost matrix (not 0.5), and check calibration.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def core_metrics(y_true, y_score) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),  # primary
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def bootstrap_pr_auc_ci(y_true, y_score, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap 95% CI for PR-AUC. A single number without this is not a result."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:      # skip degenerate resamples with no positives
            continue
        stats.append(average_precision_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


def threshold_from_cost(y_true, y_score, c_fn: float, c_fp: float):
    """Pick the probability threshold that minimizes expected cost.

    c_fn = cost of a missed fraud (false negative);
    c_fp = cost of blocking a legit customer (false positive).
    """
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    thresholds = np.unique(y_score)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        pred = y_score >= t
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t, best_cost
