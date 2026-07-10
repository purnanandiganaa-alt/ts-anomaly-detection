"""
Shared threshold selection and metric computation for all three pipelines.

Consolidates the threshold-search logic that was copy-pasted (with small
inconsistencies) across notebooks 03, 04, and 05.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, scores):
    """Sweep the precision-recall curve and return the threshold maximizing F1.

    Fix (Phase 2): `precision_recall_curve` returns `thresholds` one element
    shorter than `precision`/`recall` - index `i` of `thresholds` pairs with
    index `i` of `precision`/`recall`, not `i - 1`. The notebooks' original
    `thresholds[best_idx - 1]` was a workaround for the length mismatch that
    shifted every selection down by one, so the threshold actually applied
    never matched the F1 that was reported as "best". This affects every
    pipeline's threshold choice and reported metric consistency.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.nanargmax(f1)
    # best_idx == len(thresholds) is the trivial "no threshold" edge point
    # (precision=1, recall=0) that precision_recall_curve appends; only
    # reachable in practice if no real threshold does better.
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    return best_threshold, f1[best_idx]


def compute_metrics(y_true, scores, threshold):
    """Precision/recall/F1 at a given threshold, plus ROC-AUC over the raw scores."""
    y_pred = (scores > threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "best_threshold": float(threshold),
        "anomaly_rate": float(y_pred.mean()),
    }
