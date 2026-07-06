"""Sanity tests for evaluation utilities."""
import numpy as np
from src.evaluate import core_metrics, bootstrap_pr_auc_ci, threshold_from_cost


def test_core_metrics_perfect_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = core_metrics(y, s)
    assert m["pr_auc"] > 0.99 and m["roc_auc"] > 0.99


def test_bootstrap_ci_orders():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    mean, lo, hi = bootstrap_pr_auc_ci(y, s, n_boot=200)
    assert lo <= mean <= hi


def test_cost_threshold_prefers_recall_when_fn_expensive():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.4, 0.45, 0.6, 0.5])
    # Missed fraud very expensive -> threshold should be low enough to catch positives.
    t, cost = threshold_from_cost(y, s, c_fn=1000, c_fp=1)
    assert t <= 0.45
