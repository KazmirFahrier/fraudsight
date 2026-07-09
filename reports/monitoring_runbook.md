# Monitoring & retraining runbook (Phase 9)

## What we watch

| Signal | Metric | Cadence | Action threshold |
|---|---|---|---|
| Input / data drift | PSI per feature (`src/monitor.py`) | daily | PSI > 0.25 on any core feature → investigate + consider retrain |
| Prediction drift | mean predicted fraud prob, score histogram | daily | sustained >30% shift vs 30-day baseline |
| Concept drift | rolling PR-AUC on labeled feedback | weekly (label delay) | PR-AUC drop > 20% vs release baseline → retrain |
| Calibration | Brier score / reliability on feedback | weekly | Brier worsens materially → recalibrate (isotonic/Platt) |
| Volume / health | request rate, latency, error rate | realtime | standard SRE alerts |

## Why two kinds of drift

- **Data drift**: the inputs move (new merchant mix, holiday spend). PSI catches it early, before labels arrive.
- **Concept drift**: fraudsters change tactics, so the same inputs now mean something different. Only labeled feedback reveals it — hence the rolling PR-AUC on confirmed outcomes.

## Retraining trigger & procedure

1. Trigger fires (PSI major shift, or PR-AUC drop, or scheduled monthly refresh).
2. Rebuild features with the **point-in-time** pipeline (no leakage) on the extended window.
3. Retrain via `python -m src.train`; log to MLflow (`python -m src.track`).
4. Gate: new model must beat the incumbent on a **paired bootstrap** of PR-AUC on a common temporal test (CI on the difference must exclude 0).
5. Recompute the cost-based operating threshold; update `THRESHOLD` in `src/serve.py`.
6. Shadow-deploy, compare live, then promote.

## Label delay note

Fraud labels (chargebacks) can lag weeks. Never compute "current" PR-AUC on
transactions whose outcomes are not yet mature — it will look falsely good.
