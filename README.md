# FraudSight

**An end-to-end fraud-detection project engineered to demonstrate every competency of a senior data-science / AI-content-specialist role** — leakage-safe feature engineering, correct cross-validation, class-imbalance handling, Bayesian hyperparameter optimization, deep learning in **both** PyTorch and TensorFlow, statistical + Bayesian inference, light Spark, and MLOps — with an **AI-code-audit** track at its center.

> Companion to the [FraudSight Project Guide](docs/FraudSight_Project_Guide.pdf). Each `src` module and notebook corresponds to a phase in that guide.

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

## Run the whole pipeline (no download needed)

The project ships with a synthetic, time-ordered, heavily imbalanced transaction
generator, so the end-to-end pipeline runs out of the box. If you drop the real
Kaggle `creditcard.csv` into `data/`, it is used automatically instead.

```bash
python -m src.train          # data -> features -> temporal CV -> model -> report + artifact
python -m src.train_torch    # same features, PyTorch MLP (needs torch)
```

`src.train` writes `models/fraudsight_pipeline.joblib` (the full preprocessing+model
pipeline — no train/serve skew) and `reports/evaluation_report.md`.

### Latest results (synthetic data, held-out temporal test)

| Model | PR-AUC | 95% bootstrap CI |
|---|---|---|
| Logistic (balanced) | 0.117 | [0.078, 0.179] |
| HistGradientBoosting | 0.152 | [0.111, 0.213] |
| PyTorch MLP | 0.160 | [0.105, 0.233] |
| Always-negative baseline | 0.008 | — |

At a cost-based threshold (missed-fraud=500, false-block=5) the GBM catches ~87% of
fraud. Numbers are illustrative on synthetic data — the *leakage-safe methodology*
is what transfers to real data.

**Unsupervised detectors (Phase 4, no labels seen in training):** PCA
reconstruction-error PR-AUC 0.033 and autoencoder 0.019 — both beat the 0.008
base rate but sit well below the supervised models, as expected.

```bash
python -m src.tune       # Optuna TPE vs random search, PR-AUC under temporal CV
python -m src.anomaly    # PCA + autoencoder reconstruction-error detectors
```

**Big-data & MLOps (Phases 8-9):**

```bash
SPARK_LOCAL_IP=127.0.0.1 python -m src.spark_features   # Spark window features == pandas; MLlib GBT
python -m src.track                                     # MLflow: log params/metrics/artifact
uvicorn src.serve:app --reload                          # FastAPI /predict over the saved pipeline
pytest tests/test_monitor.py                            # PSI drift-metric tests
```

Phase 8 proves the Spark `rangeBetween(-86400, -1)` window reproduces the tested
pandas prior-counts exactly (leakage-safe boundary survives distribution). Phase 9
serves the *full pipeline* artifact (no train/serve skew) and monitors input drift
via PSI; see `reports/monitoring_runbook.md` for the retraining policy.

## Layout

```
fraudsight/
├── README.md
├── requirements.txt
├── data/                 # dataset lives here (gitignored); see data/README.md
├── sql/features.sql      # point-in-time feature engineering (DuckDB/SQLite)
├── R/inference.R         # bootstrap CI, paired model test, calibration (R)
├── src/
│   ├── seed.py           # reproducibility helper
│   ├── data.py           # synthetic generator + real Kaggle loader
│   ├── features.py       # leakage-safe, point-in-time feature pipeline
│   ├── cv.py             # temporal + nested cross-validation helpers
│   ├── train.py          # end-to-end training -> artifact + report
│   ├── train_torch.py    # end-to-end PyTorch training (Phase 6a)
│   ├── models_torch.py   # PyTorch MLP definition
│   ├── models_tf.py      # TensorFlow/Keras MLP definition (Phase 6b)
│   ├── evaluate.py       # PR-AUC, bootstrap CIs, cost-based threshold
│   ├── anomaly.py        # PCA + autoencoder anomaly detection (Phase 4)
│   ├── tune.py           # Optuna Bayesian HPO vs random search (Phase 5)
│   ├── spark_features.py # PySpark window features + MLlib GBT (Phase 8)
│   ├── track.py          # MLflow experiment tracking (Phase 9)
│   ├── serve.py          # FastAPI scoring endpoint (Phase 9)
│   └── monitor.py        # PSI drift monitoring (Phase 9)
├── models/               # saved pipeline artifact (gitignored)
├── tests/
│   ├── test_no_leakage.py
│   └── test_metrics.py
├── notebooks/            # one per guide phase
├── reports/              # auto-generated evaluation report + audit log
└── .github/workflows/ci.yml
```

## Phase → file map

| Phase | Topic | Code |
|------|-------|------|
| 1 | Leakage-safe features (SQL/pandas) | `sql/features.sql`, `src/features.py` |
| 2 | Metrics under imbalance | `src/evaluate.py`, `src/train.py` |
| 3 | Temporal + nested CV | `src/cv.py`, `src/train.py` |
| 4 | Dimensionality reduction + anomaly detection | `src/anomaly.py`, `notebooks/04_dimensionality_reduction.ipynb` |
| 5 | Bayesian HPO (Optuna) | `src/tune.py`, `notebooks/05_bayesian_hpo.ipynb` |
| 6 | Deep learning (PyTorch + TF) | `src/train_torch.py`, `src/models_torch.py`, `src/models_tf.py` |
| 7 | Statistical + Bayesian inference | `src/evaluate.py`, `R/inference.R`, `notebooks/07_inference.ipynb` |
| 8 | PySpark features + MLlib | `src/spark_features.py`, `notebooks/08_spark.ipynb` |
| 9 | MLOps: tracking, serving, drift, CI | `src/track.py`, `src/serve.py`, `src/monitor.py`, `reports/monitoring_runbook.md`, `.github/workflows/ci.yml` |
| 10 | AI-code-audit capstone | `reports/audit_log.md` |

## License

MIT © 2026 Kazmir Fahrier
