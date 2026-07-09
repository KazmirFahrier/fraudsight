"""Phase 9 — drift monitoring.

Data drift  = the input distribution shifts (feature means/shape move).
Concept drift = the X->y relationship changes (fraud patterns evolve) even if
                inputs look the same.

Population Stability Index (PSI) is the standard input-drift metric:
  PSI < 0.1  : no significant shift
  0.1-0.25   : moderate shift — investigate
  > 0.25     : major shift — retrain
"""
from __future__ import annotations
import numpy as np


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between a reference (expected) and new (actual) sample."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    # fixed bin edges from the reference distribution's quantiles
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def psi_verdict(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate shift — investigate"
    return "major shift — retrain"
