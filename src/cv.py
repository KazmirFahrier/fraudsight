"""Cross-validation done correctly for imbalanced, time-ordered data.

- Temporal validation: training must always precede validation in time.
- Resampling (SMOTE etc.) belongs INSIDE the training fold, via imblearn's
  Pipeline, never before cross_val_score.
- Nested CV: inner loop tunes, outer loop estimates honest performance.
"""
from __future__ import annotations
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score, GridSearchCV


def temporal_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Walk-forward splits: each validation fold comes strictly after its train fold."""
    return TimeSeriesSplit(n_splits=n_splits)


def leakage_safe_pipeline(estimator, sampler=None):
    """Wrap sampler + estimator so resampling is refit per fold (no cross-fold leak)."""
    if sampler is None:
        return estimator
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([("sampler", sampler), ("clf", estimator)])


def nested_cv_score(estimator, param_grid, X, y, scoring="average_precision",
                    outer_splits: int = 5, inner_splits: int = 3, seed: int = 42):
    """Return the UNBIASED outer-loop scores. Report these, never the inner best."""
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed)
    inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, param_grid, cv=inner, scoring=scoring)
    return cross_val_score(search, X, y, cv=outer, scoring=scoring)
