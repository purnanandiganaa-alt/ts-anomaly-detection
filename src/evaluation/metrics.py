"""
Shared threshold selection and metric computation for all three pipelines.

Consolidates the threshold-search logic that was copy-pasted (with small
inconsistencies) across notebooks 03, 04, and 05.
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, scores, beta=1.0):
    """Sweep the precision-recall curve and return the threshold maximizing F-beta.

    Fix (Phase 2): `precision_recall_curve` returns `thresholds` one element
    shorter than `precision`/`recall` - index `i` of `thresholds` pairs with
    index `i` of `precision`/`recall`, not `i - 1`. The notebooks' original
    `thresholds[best_idx - 1]` was a workaround for the length mismatch that
    shifted every selection down by one, so the threshold actually applied
    never matched the F1 that was reported as "best". This affects every
    pipeline's threshold choice and reported metric consistency.

    Round 4 fix: `beta` < 1 weights precision over recall. Given how
    overlapped the normal/anomalous score distributions are in this project,
    the plain-F1 (beta=1) optimum has consistently landed on an aggressive,
    high-recall/low-precision operating point (over-predicting anomalies).
    beta=0.5 (the default used by the pipelines via config.THRESHOLD_BETA)
    trades some recall for a much less extreme false-positive rate.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-8)

    best_idx = np.nanargmax(f_beta)
    # best_idx == len(thresholds) is the trivial "no threshold" edge point
    # (precision=1, recall=0) that precision_recall_curve appends; only
    # reachable in practice if no real threshold does better.
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    return best_threshold, f_beta[best_idx]


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


def fit_mahalanobis(normal_errors):
    """Fit a Gaussian error model on confirmed-normal per-sensor error vectors.

    Round 4: `normal_errors` is (n_windows, n_sensors) - per-sensor CNN
    reconstruction error, one row per window known to be normal. Uses a
    Ledoit-Wolf shrinkage estimator rather than a raw sample covariance,
    since it stays well-conditioned (and safely invertible) even with a
    modest number of normal windows relative to the number of sensors.
    Returns (mu, precision) where precision is the inverse covariance,
    ready to plug into mahalanobis_scores.
    """
    mu = normal_errors.mean(axis=0)
    precision = LedoitWolf().fit(normal_errors).precision_
    return mu, precision


def mahalanobis_scores(errors, mu, precision):
    """Mahalanobis distance of each error vector from the fitted normal model.

    Round 4 fix: replaces flat mean-squared-error (which averages away
    exactly the information about *which* sensor is off) with a score that
    weighs each sensor's deviation by how much it naturally varies on normal
    data, and accounts for how sensors' errors normally co-vary. `errors` is
    (n_windows, n_sensors); returns one score per window.
    """
    diff = errors - mu
    return np.einsum("ij,jk,ik->i", diff, precision, diff)
