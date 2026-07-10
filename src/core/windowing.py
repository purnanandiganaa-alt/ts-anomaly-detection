"""
Shared windowing utilities for the CNN autoencoder and window-based
Isolation Forest branches.

This consolidates the ~4 near-identical copies of load/window logic that
were duplicated across notebooks 03, 04, and 05.
"""

import glob
import os

import numpy as np
import pandas as pd


def list_split_files(folder):
    files = sorted(glob.glob(os.path.join(str(folder), "*.csv")))
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder}")
    return files


def load_split(folder) -> pd.DataFrame:
    """Concatenate every CSV run in a split folder into one DataFrame.

    Only safe for row-independent uses (e.g. fitting a StandardScaler).
    Do not window the result directly - use load_split_per_file +
    create_windows_per_file instead, so windows never cross run boundaries.
    """
    files = list_split_files(folder)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_split_per_file(folder):
    """Load each CSV run in a split folder as a separate DataFrame."""
    files = list_split_files(folder)
    return [pd.read_csv(f) for f in files]


def load_labels_per_file(folder, column="label"):
    """Load each run's label CSV as a separate array, aligned by sorted filename
    with load_split_per_file(val_folder) (both splits use matching series_*.csv names).
    """
    files = list_split_files(folder)
    return [pd.read_csv(f)[column].values for f in files]


def create_windows(data, labels=None, window_size=200, step=1):
    """
    Slide a fixed-size window over `data` (shape: [n_timesteps, n_features])
    from a single run.

    A window is labeled anomalous if any of its timesteps are anomalous
    (`np.mean(labels[i:i+window_size]) > 0`), matching the rule used by the
    pipeline cells that actually produced the project's reported results.
    """
    X, y = [], []

    for i in range(0, len(data) - window_size + 1, step):
        X.append(data[i:i + window_size])

        if labels is not None:
            y.append(1 if np.mean(labels[i:i + window_size]) > 0 else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32) if labels is not None else None

    return X, y


def create_windows_per_file(arrays, labels_per_file=None, window_size=200, step=1):
    """Window each run separately and concatenate the results.

    Fix (Phase 2): the original pipelines concatenated every file in a split
    into one long array before windowing, so a window could splice the tail
    of one run onto the head of an unrelated run. Windowing each run
    separately (this function) and only concatenating the resulting windows
    keeps every window's timesteps physically contiguous within one run,
    removing that corrupted training/eval signal.
    """
    X_all, y_all = [], []

    for i, arr in enumerate(arrays):
        labels = labels_per_file[i] if labels_per_file is not None else None
        X, y = create_windows(arr, labels, window_size=window_size, step=step)
        if len(X) == 0:
            continue
        X_all.append(X)
        if y is not None:
            y_all.append(y)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0) if y_all else None

    return X_all, y_all


def expand_window_scores_to_timesteps(window_scores, n_timesteps, window_size, step):
    """Map one score per window back to one score per raw timestep.

    Fix (Phase 2): the submission format is per-timestep, but scoring used a
    single window-level label blindly broadcast across every raw timestep in
    that block (10, with the old STEP) - any anomaly shorter than a block, or
    offset from its boundary, was mislabeled either way. Scoring with a dense
    stride (e.g. EVAL_STEP=1) and averaging every overlapping window's score
    per timestep gives a real per-timestep signal instead of a coarse,
    blindly-broadcast one.
    """
    sums = np.zeros(n_timesteps, dtype=np.float64)
    counts = np.zeros(n_timesteps, dtype=np.float64)

    for idx, i in enumerate(range(0, n_timesteps - window_size + 1, step)):
        sums[i:i + window_size] += window_scores[idx]
        counts[i:i + window_size] += 1

    # Timesteps with no covering window (only possible right at a run's tail
    # when step doesn't evenly divide it) fall back to the nearest score.
    uncovered = counts == 0
    if uncovered.any():
        last_covered = np.where(~uncovered)[0].max()
        sums[uncovered] = sums[last_covered] / counts[last_covered]
        counts[uncovered] = 1

    return sums / counts
