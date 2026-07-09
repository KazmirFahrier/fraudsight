"""Drift-metric unit tests (Phase 9)."""
import numpy as np
from src.monitor import psi, psi_verdict


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    assert psi(x, y) < 0.1
    assert psi_verdict(psi(x, y)) == "stable"


def test_psi_flags_large_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(3, 1, 5000)          # mean shifted by 3 sigma
    assert psi(x, y) > 0.25
    assert psi_verdict(psi(x, y)) == "major shift — retrain"
