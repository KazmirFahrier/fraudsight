-- FraudSight — point-in-time-correct feature engineering (Phase 1)
-- Runs in DuckDB or SQLite. The whole discipline is the `p.ts < t.ts` predicate:
-- every aggregate for transaction t uses ONLY transactions strictly before t.
-- Using <= would leak the current row; dropping the time bound leaks the future.

-- 1) Rolling 24h spend & count PER CARD, excluding the current transaction.
WITH velocity AS (
  SELECT
    t.txn_id,
    t.card_id,
    t.ts,
    t.amount,
    COALESCE(SUM(p.amount), 0)   AS spend_prior_24h,
    COALESCE(COUNT(p.txn_id), 0) AS count_prior_24h
  FROM txns t
  LEFT JOIN txns p
    ON p.card_id = t.card_id
   AND p.ts <  t.ts                              -- strictly before: no leakage
   AND p.ts >= t.ts - INTERVAL 24 HOUR
  GROUP BY t.txn_id, t.card_id, t.ts, t.amount
),

-- 2) Lifetime prior-transaction count per card (also strictly-prior).
lifetime AS (
  SELECT
    txn_id,
    COUNT(*) OVER (
      PARTITION BY card_id ORDER BY ts
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING   -- excludes current row
    ) AS txns_prior_lifetime
  FROM txns
)

SELECT
  v.txn_id,
  v.card_id,
  v.amount,
  LN(1 + v.amount)              AS log_amount,
  EXTRACT(hour FROM v.ts)       AS hour,
  v.spend_prior_24h,
  v.count_prior_24h,
  COALESCE(l.txns_prior_lifetime, 0) AS txns_prior_lifetime
FROM velocity v
JOIN lifetime l USING (txn_id);

-- NOTE: any target-derived aggregate (e.g. a per-card historical fraud rate)
-- must ALSO use the strictly-prior window AND be computed on training rows only,
-- or it becomes target leakage. Never compute it over the full table.
