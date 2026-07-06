# FraudSight

**An end-to-end fraud-detection project engineered to demonstrate every competency of a senior data-science / AI-content-specialist role** — leakage-safe feature engineering, correct cross-validation, class-imbalance handling, Bayesian hyperparameter optimization, deep learning in **both** PyTorch and TensorFlow, statistical + Bayesian inference, light Spark, and MLOps — with an **AI-code-audit** track at its center.

> Companion to the *FraudSight Project Guide* (PDF). Each `src` module and notebook corresponds to a phase in that guide.

## Why this repo exists

The hard part of production ML is not fitting a model — it is avoiding the silent failure modes that inflate offline metrics and collapse in production. This repo is organized around catching five families of them:

1. **Data leakage** — target leakage, train/test contamination, resampling before splitting, non-point-in-time features.
2. **Overfitting** — tuning on a fixed validation set, no early stopping, reporting inner-loop CV scores.
3. **Class-imbalance mishandling** — accuracy as a metric, default 0.5 threshold, resampling the test set.
4. **Invalid statistical conclusions** — point estimates with no interval, confidence vs credible confusion, uncalibrated probabilities.
5. **Reproducibility / engineering** — unseeded randomness, train/serve skew, serializing a model without its preprocessing.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # runs the leakage + metric unit tests
```

Download the dataset (not committed) into `data/` — see [`data/README.md`](data/README.md).

## Layout

```
fraudsight/
├── README.md
├── requirements.txt
├── data/                 # dataset lives here (gitignored); see data/README.md
├── src/
│   ├── seed.py           # reproducibility helper
│   ├── features.py       # leakage-safe, point-in-time feature pipeline
│   ├── cv.py             # temporal + nested cross-validation helpers
│   ├── models_torch.py   # PyTorch MLP (Phase 6a)
│   ├── models_tf.py      # TensorFlow/Keras MLP (Phase 6b)
│   └── evaluate.py       # PR-AUC, bootstrap CIs, calibration
├── tests/
│   ├── test_no_leakage.py
│   └── test_metrics.py
├── notebooks/            # one stub per guide phase
├── reports/             # evaluation report + audit log go here
└── .github/workflows/ci.yml
```

## Phase → file map

| Phase | Topic | Code |
|------|-------|------|
| 1 | Leakage-safe features (SQL/pandas) | `src/features.py` |
| 2 | Metrics under imbalance | `src/evaluate.py` |
| 3 | Temporal + nested CV | `src/cv.py` |
| 4 | Dimensionality reduction | `notebooks/04_dimensionality_reduction.ipynb` |
| 5 | Bayesian HPO (Optuna) | `notebooks/05_bayesian_hpo.ipynb` |
| 6 | Deep learning (PyTorch + TF) | `src/models_torch.py`, `src/models_tf.py` |
| 7 | Statistical + Bayesian inference | `src/evaluate.py`, `notebooks/07_inference.ipynb` |
| 8 | PySpark features | `notebooks/08_spark.ipynb` |
| 9 | MLOps / CI | `.github/workflows/ci.yml`, `tests/` |
| 10 | AI-code-audit capstone | `reports/audit_log.md` |

## License

MIT © 2026 Kazmir Fahrier
