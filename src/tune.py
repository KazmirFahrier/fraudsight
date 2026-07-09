"""Phase 5 — Bayesian hyperparameter optimization with Optuna (TPE).

Tunes the HistGradientBoosting fraud model against PR-AUC under TEMPORAL
cross-validation, and contrasts Bayesian (TPE) search with random search at an
equal trial budget. The point: TPE models the objective with a surrogate and
spends trials where expected improvement is highest, so it typically reaches a
better score for the same budget.

Run:  python -m src.tune            (requires optuna)
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from src.seed import seed_everything
from src.data import get_data
from src.train import engineer, temporal_split, FEATURES
from src.cv import temporal_cv


def _score(params, X, y, n_splits=4):
    model = HistGradientBoostingClassifier(random_state=42, **params)
    sw = compute_sample_weight("balanced", y)
    # sample_weight passed through to each fold's fit via fit_params
    return cross_val_score(model, X, y, cv=temporal_cv(n_splits),
                           scoring="average_precision",
                           params={"sample_weight": sw}).mean()


def _space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 600),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
    }


def run(n_trials: int = 30, max_rows: int | None = None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed_everything(42)
    df = engineer(get_data()[0])
    train, _ = temporal_split(df)
    if max_rows:                       # optional cap keeps tuning fast (temporal order preserved)
        train = train.iloc[:max_rows]
    X, y = train[FEATURES], train["is_fraud"].to_numpy()

    def objective(trial):
        return _score(_space(trial), X, y)

    tpe = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
    tpe.optimize(objective, n_trials=n_trials)

    rnd = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.RandomSampler(seed=42))
    rnd.optimize(objective, n_trials=n_trials)

    print(f"Bayesian (TPE)  best CV PR-AUC: {tpe.best_value:.4f}")
    print(f"Random search   best CV PR-AUC: {rnd.best_value:.4f}")
    print(f"TPE best params: {tpe.best_params}")
    return {"tpe_best": tpe.best_value, "random_best": rnd.best_value,
            "best_params": tpe.best_params}


if __name__ == "__main__":
    run()
