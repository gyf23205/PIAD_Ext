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


def compute_affiliation_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """Compute affiliation precision/recall/F1 (Huet et al. NeurIPS 2022).

    Used by the SimAD paper as UAff/NAff.  Rewards predictions that are
    temporally close to — not just overlapping with — actual anomaly events.

    Args:
        y_true_bin: (n,) binary int array — 1 = anomaly, 0 = normal
        y_pred_bin: (n,) binary int array — predicted labels

    Returns dict: p_aff, r_aff, f_aff
    """
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events

    Trange = (0, len(y_true_bin))
    events_gt   = convert_vector_to_events(y_true_bin.tolist())
    events_pred = convert_vector_to_events(y_pred_bin.tolist())

    if not events_gt:
        return {"p_aff": float("nan"), "r_aff": float("nan"), "f_aff": float("nan")}

    result = pr_from_events(events_pred, events_gt, Trange)
    p = float(result["precision"])
    r = float(result["recall"])
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p_aff": p, "r_aff": r, "f_aff": f}


def single_to_multihot(y_pred_single: np.ndarray, n_ac: int) -> np.ndarray:
    """Convert (n,) single-class int predictions to (n, n_ac) binary indicator.

    Convention: 0 = normal (all-zero row), k>=1 = col k-1 set to 1.
    """
    out = np.zeros((len(y_pred_single), n_ac), dtype=int)
    for i, cls in enumerate(y_pred_single):
        if cls >= 1:
            col = int(cls) - 1
            if col < n_ac:
                out[i, col] = 1
    return out


def compute_multihot_metrics(y_true_mh: np.ndarray, y_pred_mh: np.ndarray) -> dict:
    """Compute multi-label metrics using sklearn indicator (multi-hot) format.

    Returns: f1_macro, f1_weighted, mh_acc (1-hamming_loss), mh_recall (macro avg).
    """
    from sklearn.metrics import f1_score, hamming_loss, recall_score
    return {
        "f1_macro":    float(f1_score(y_true_mh, y_pred_mh, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(y_true_mh, y_pred_mh, average="weighted", zero_division=0)),
        "mh_acc":      float(1.0 - hamming_loss(y_true_mh, y_pred_mh)),
        "mh_recall":   float(recall_score(y_true_mh, y_pred_mh, average="macro", zero_division=0)),
    }
