# Red-team audit log (Phase 10)

For each AI-generated solution reviewed, record a structured critique.

## Template

- **Submission:** <what/where>
- **Verdict:** correct / flawed / subtly flawed
- **Failure mode:** leakage / overfitting / imbalance / invalid statistics / engineering
- **Evidence:** the exact line(s) and why they fail
- **Fix:** corrected code
- **Reasoning feedback:** how the model's thought process went wrong

## Example entry

- **Submission:** `SMOTE().fit_resample(X, y)` before `cross_val_score`
- **Verdict:** flawed
- **Failure mode:** leakage
- **Evidence:** synthetic points derived from validation-fold neighbors leak across folds; resampling must be refit inside each training fold.
- **Fix:** wrap in `imblearn.pipeline.Pipeline([("smote", SMOTE()), ("clf", clf)])` and pass to `cross_val_score` with `TimeSeriesSplit`.
- **Reasoning feedback:** the model treated resampling as a one-time preprocessing step rather than part of the model that must be refit per fold.
