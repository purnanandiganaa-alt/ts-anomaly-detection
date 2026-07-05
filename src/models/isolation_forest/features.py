"""
Per-window statistical features for the Isolation Forest branch of the
hybrid pipeline (notebooks/05_hybrid_model.ipynb).
"""

import numpy as np


def extract_window_features(data, window_size=200, step=10):
    """Build a per-sensor feature matrix (mean/std/min/max/median/p25/p75) per window.

    Fix (Phase 2): the original implementation computed each statistic over
    the whole window array with no `axis` argument, flattening all sensor
    channels into a single scalar per statistic - Isolation Forest never saw
    which sensor was actually off, only one blended number per window. Adding
    `axis=0` keeps stats per-sensor, producing `7 * n_sensors` features
    instead of 7. This is the most likely explanation for Isolation Forest's
    weak standalone F1 (~0.40) compared to the CNN branch (~0.60): expected
    to give it the discriminative signal it was previously missing entirely.
    """
    X = []

    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]  # shape: (window_size, n_sensors)

        features = np.concatenate([
            np.mean(window, axis=0),
            np.std(window, axis=0),
            np.min(window, axis=0),
            np.max(window, axis=0),
            np.median(window, axis=0),
            np.percentile(window, 25, axis=0),
            np.percentile(window, 75, axis=0),
        ])
        X.append(features)

    return np.array(X, dtype=np.float32)
