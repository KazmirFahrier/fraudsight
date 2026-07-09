"""Phase 9 — experiment tracking with MLflow.

Wraps the end-to-end training run and logs params, metrics, and the pipeline
artifact to a local MLflow store (./mlruns). Every run is reproducible and
comparable.

Run:  python -m src.track
"""
from __future__ import annotations
import os
from pathlib import Path
import mlflow
from src.train import main as train_main, ROOT


def run():
    # SQLite backend (the file store is deprecated in MLflow 3.x). Defaults to the
    # repo root; override with MLFLOW_DB_PATH if your filesystem lacks SQLite
    # locking (e.g. some network/overlay mounts).
    db = os.environ.get("MLFLOW_DB_PATH", str(ROOT / "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment("fraudsight")
    with mlflow.start_run():
        results = train_main()
        mlflow.log_metric("gbm_pr_auc", results["gbm"]["pr_auc"])
        mlflow.log_metric("logistic_pr_auc", results["logistic"]["pr_auc"])
        mlflow.log_param("data_source", results["meta"]["data_source"])
        mlflow.log_param("n_train", results["meta"]["n_train"])
        art = ROOT / "models/fraudsight_pipeline.joblib"
        if art.exists():
            mlflow.log_artifact(str(art))
        report = ROOT / "reports/evaluation_report.md"
        if report.exists():
            mlflow.log_artifact(str(report))
    print("Logged run to", mlflow.get_tracking_uri())
    return results


if __name__ == "__main__":
    run()
