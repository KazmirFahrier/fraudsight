"""Phase 6b — TensorFlow/Keras tabular fraud classifier (same architecture as torch).

Correctness points to audit for: PR-AUC (not accuracy) as the metric, a temporal
validation split, EarlyStopping on val PR-AUC, and class_weight for imbalance.
"""
from __future__ import annotations


def build_model(d_in: int, p_drop: float = 0.3):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input((d_in,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(p_drop),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(p_drop),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def early_stopping():
    import tensorflow as tf
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True
    )
