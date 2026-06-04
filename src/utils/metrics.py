"""Shared metrics and label-conversion utilities for all PIAD_Ext baselines."""

import numpy as np
from sklearn.metrics import roc_auc_score


def multihot_to_single(semi_y: np.ndarray) -> np.ndarray:
    """Convert (n, n_ac) semi-supervised multi-hot labels to (n,) single-class.

    Convention (matches PIAD_Ext preprocessing.py):
      all -1  -> -1  (unlabeled)
      all  0  ->  0  (labeled normal)
      has  1s -> col_index + 1  (anomaly class, 1-based)
    """
    single = np.full(len(semi_y), -1, dtype=np.int64)
    for i in range(len(semi_y)):
        row = semi_y[i]
        if np.all(row == 0):
            single[i] = 0
        elif np.any(row > 0):
            single[i] = int(np.where(row > 0)[0][0]) + 1
        # else: all -1 → remains -1
    return single


def truth_multihot_to_single(y: np.ndarray) -> np.ndarray:
    """Convert ground-truth multi-hot labels (no -1 entries) to single-class.

    all-0 -> 0 (normal); has 1s -> first-1-column + 1 (anomaly type).
    """
    single = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        cols = np.where(y[i] > 0)[0]
        if len(cols) > 0:
            single[i] = int(cols[0]) + 1
    return single


def compute_anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_score: np.ndarray) -> dict:
    """Compute anomaly detection metrics.

    Args:
        y_true:  (n,) integer class labels — 0 = normal, >0 = anomaly type
        y_pred:  (n,) predicted class labels
        y_score: (n,) anomaly score per sample (higher = more anomalous)

    Returns dict: auc, f1_macro, f1_weighted, accuracy, anomaly_recall
    """
    from sklearn.metrics import f1_score

    y_true_bin = (y_true > 0).astype(int)

    try:
        auc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        auc = float("nan")

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = float(np.mean(y_true == y_pred))

    anomaly_mask = y_true > 0
    if anomaly_mask.any():
        anomaly_recall = float(
            np.sum((y_pred[anomaly_mask] > 0) & (y_pred[anomaly_mask] == y_true[anomaly_mask]))
            / anomaly_mask.sum()
        )
    else:
        anomaly_recall = float("nan")

    return {
        "auc": auc,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "accuracy": acc,
        "anomaly_recall": anomaly_recall,
    }
