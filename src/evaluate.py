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
    if y_true.sum() == 0:
        raise ValueError("Cannot bootstrap PR-AUC: the sample contains no positive labels.")
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:      # skip degenerate resamples with no positives
            continue
        stats.append(average_precision_score(y_true[idx], y_score[idx]))
    if not stats:
        raise ValueError("All bootstrap resamples were degenerate (too few positives).")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


def threshold_from_cost(y_true, y_score, c_fn: float, c_fp: float):
    """Pick the probability threshold that minimizes expected cost.

    c_fn = cost of a missed fraud (false negative);
    c_fp = cost of blocking a legit customer (false positive).

    Vectorized O(n log n): sweeps every candidate threshold (each unique score,
    prediction = score >= t) via cumulative counts instead of an O(n^2) loop, so
    it stays fast on large test sets (e.g. the 284k-row Kaggle data).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(y_score)
    s_sorted = y_score[order]
    y_sorted = y_true[order]
    N = int((y_true == 0).sum())
    cum_pos = np.cumsum(y_sorted)            # positives with score <= s_sorted[i]
    cum_neg = np.cumsum(1 - y_sorted)        # negatives with score <= s_sorted[i]

    uniq = np.unique(s_sorted)
    i = np.searchsorted(s_sorted, uniq, side="left")   # first index with score >= t
    prev = i - 1
    fn = np.where(prev >= 0, cum_pos[np.clip(prev, 0, None)], 0)   # positives strictly below t
    neg_below = np.where(prev >= 0, cum_neg[np.clip(prev, 0, None)], 0)
    fp = N - neg_below                                            # negatives at or above t
    cost = c_fn * fn + c_fp * fp
    k = int(np.argmin(cost))
    return float(uniq[k]), float(cost[k])
