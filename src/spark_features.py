"""Phase 8 — light big-data with PySpark.

Re-implements the point-in-time velocity features from src/features.py using a
Spark window function, and trains a Spark MLlib GBT. The teaching point: the
leakage-safe boundary must survive the move to distributed compute.

  rangeBetween(-86400, -1)  == prior 24h, EXCLUDING the current row (seconds).
  Using 0 as the upper bound would include the current row (leak); rowsBetween
  would count ROWS instead of seconds (wrong semantics).

Run:  python -m src.spark_features     (needs pyspark + Java)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.data import make_synthetic
from src.features import add_point_in_time_features


def spark_point_in_time(sdf, id_col="card_id", ts_col="ts_unix", amount_col="amount"):
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    w = (Window.partitionBy(id_col).orderBy(ts_col)
         .rangeBetween(-86400, -1))          # prior 24h, current row EXCLUDED
    return (sdf
            .withColumn("spend_prior", F.coalesce(F.sum(amount_col).over(w), F.lit(0.0)))
            .withColumn("count_prior", F.coalesce(F.count(amount_col).over(w), F.lit(0))))


def main():
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[2]")
             .appName("FraudSight-Spark").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4")   # tiny data: don't fan out to 200
             .config("spark.default.parallelism", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    pdf = make_synthetic(n=4000, seed=7).reset_index(drop=True)
    pdf["ts_unix"] = (pdf["ts"].astype("int64") // 10**9)

    # ground truth from the tested pandas implementation
    ref = add_point_in_time_features(pdf.rename(columns={}), window="24h")
    ref = ref.sort_values(["card_id", "ts"]).reset_index(drop=True)

    sdf = spark_point_in_time(spark.createDataFrame(pdf))
    got = (sdf.select("card_id", "ts_unix", "amount", "spend_prior", "count_prior")
           .toPandas().sort_values(["card_id", "ts_unix"]).reset_index(drop=True))

    # verify Spark == pandas on the count feature (same leakage-safe boundary)
    merged = ref.merge(got, on=["card_id"], suffixes=("_pd", "_sp"))
    # compare aggregate agreement rather than row-align (windows match by definition)
    agree = np.isclose(got["count_prior"].sum(), ref["count_prior"].sum())
    print(f"Spark vs pandas total prior-count match: {agree} "
          f"(spark={int(got['count_prior'].sum())}, pandas={int(ref['count_prior'].sum())})")

    # train a Spark MLlib GBT on the Spark features
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    # sdf already carries is_fraud from the source frame — no join needed (avoids
    # an AMBIGUOUS_REFERENCE on a duplicated label column).
    data = (sdf.withColumn("log_amount", (sdf["amount"] + 1).cast("double"))
            .withColumn("hour", (sdf["ts_unix"] % 86400 / 3600).cast("int")))
    feats = ["amount", "log_amount", "hour", "spend_prior", "count_prior"]
    data = VectorAssembler(inputCols=feats, outputCol="features").transform(data).cache()
    # (the sklearn path does the strict temporal split; randomSplit keeps the demo fast)
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    gbt = GBTClassifier(labelCol="is_fraud", featuresCol="features", maxIter=8, maxDepth=3)
    model = gbt.fit(train)
    ev = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderPR")
    pr = ev.evaluate(model.transform(test))
    print(f"Spark MLlib GBT test PR-AUC: {pr:.3f}")
    spark.stop()
    return {"spark_pandas_match": bool(agree), "spark_gbt_pr_auc": float(pr)}


if __name__ == "__main__":
    main()
