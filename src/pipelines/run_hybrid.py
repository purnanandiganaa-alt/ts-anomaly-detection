"""
Hybrid pipeline: CNN autoencoder + window-based Isolation Forest, fused.

Deduplicated from notebooks/05_hybrid_model.ipynb, with fixes applied across
several rounds: windows respect run boundaries, both branches are scored
densely and expanded to per-timestep scores before fusion, the fusion
weight and Isolation Forest contamination are swept on validation instead
of hardcoded, and (Round 4) the CNN score is a Mahalanobis distance over
per-sensor reconstruction error rather than a flat mean, with an F-beta
(precision-weighted) threshold used throughout.

Run from the repository root: python -m src.pipelines.run_hybrid
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import config
from src.core.windowing import (
    create_windows_per_file,
    expand_window_scores_to_timesteps,
    load_labels_per_file,
    load_split_per_file,
)
from src.evaluation.metrics import (
    compute_metrics,
    fit_mahalanobis,
    find_best_threshold,
    mahalanobis_scores,
)
from src.models.cnn.predict import get_reconstruction_errors
from src.models.cnn.train import train_autoencoder
from src.models.isolation_forest.features import extract_window_features
from src.models.isolation_forest.predict import score as score_isolation_forest
from src.models.isolation_forest.train import train_isolation_forest

OUTPUT_DIR = config.OUTPUT_DIR / "hybrid"
SUBMISSION_PATH = config.OUTPUT_DIR / "hybrid_predictions.csv"

# Fusion weight sweep grid: CNN_WEIGHT candidates (ISO_WEIGHT = 1 - CNN_WEIGHT).
# Fix (Phase 2): 0.85/0.15 was picked once by hand; this reuses scores already
# computed for both branches, so sweeping costs nothing extra to compute.
# Round 2: widened to 0.1 steps, still free since it's reusing computed scores.
CNN_WEIGHT_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Round 2: contamination was hand-set (0.2) and never tuned, same issue the
# fusion weight had. Isolation Forest is cheap to retrain (no epochs), so
# sweeping it costs little.
WINDOW_IF_CONTAMINATION_GRID = [0.1, 0.15, 0.2, 0.25, 0.3]


def fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device):
    """Fit a Gaussian error model on validation windows confirmed normal.

    Round 4: windows validation at train-time stride (config.STEP), gets
    per-sensor reconstruction errors, and fits mu/precision only on windows
    with no anomalous timestep at all (window label == 0).
    """
    errors_list, labels_list = [], []
    for df, labels in zip(val_files, val_labels):
        scaled = scaler.transform(df.values)
        windows, window_labels = create_windows_per_file(
            [scaled], [labels], window_size=config.WINDOW_SIZE, step=config.STEP
        )
        errors_list.append(
            get_reconstruction_errors(model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True)
        )
        labels_list.append(window_labels)

    errors = np.concatenate(errors_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return fit_mahalanobis(errors[labels == 0])


def score_cnn_densely(model, files, scaler, device, mu, precision):
    per_timestep = []
    for df in files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file([scaled], window_size=config.WINDOW_SIZE, step=config.EVAL_STEP)
        window_errors = get_reconstruction_errors(
            model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True
        )
        window_scores = mahalanobis_scores(window_errors, mu, precision)
        per_timestep.append(
            expand_window_scores_to_timesteps(window_scores, len(df), config.WINDOW_SIZE, config.EVAL_STEP)
        )
    return np.concatenate(per_timestep)


def score_iso_densely(model, files, scaler):
    per_timestep = []
    for df in files:
        scaled = scaler.transform(df.values)
        window_features = extract_window_features(scaled, window_size=config.WINDOW_SIZE, step=config.EVAL_STEP)
        window_scores = score_isolation_forest(model, window_features)
        per_timestep.append(
            expand_window_scores_to_timesteps(window_scores, len(df), config.WINDOW_SIZE, config.EVAL_STEP)
        )
    return np.concatenate(per_timestep)


def normalize(val_scores, test_scores):
    """Min-max normalize using validation-set stats applied to both (no test leakage)."""
    lo, hi = val_scores.min(), val_scores.max()
    return (val_scores - lo) / (hi - lo + 1e-8), (test_scores - lo) / (hi - lo + 1e-8)


def sweep_fusion_weight(y_val, cnn_val_n, iso_val_n):
    """Pick the CNN/ISO weight split that maximizes validation F-beta."""
    best = None
    for cnn_weight in CNN_WEIGHT_GRID:
        fused = cnn_weight * cnn_val_n + (1 - cnn_weight) * iso_val_n
        threshold, f_beta = find_best_threshold(y_val, fused, beta=config.THRESHOLD_BETA)
        if best is None or f_beta > best["f_beta"]:
            best = {"cnn_weight": cnn_weight, "threshold": threshold, "f_beta": f_beta}
    return best["cnn_weight"], best["threshold"], best["f_beta"]


def sweep_iso_contamination(X_train_iso, val_files, scaler, y_val):
    """Train one Isolation Forest per contamination candidate and keep
    whichever scores best on validation - mirrors sweep_fusion_weight above."""
    best = None
    for contamination in WINDOW_IF_CONTAMINATION_GRID:
        model = train_isolation_forest(
            X_train_iso,
            n_estimators=config.WINDOW_IF_N_ESTIMATORS,
            contamination=contamination,
            random_state=config.RANDOM_SEED,
        )
        iso_val = score_iso_densely(model, val_files, scaler)
        _, f_beta = find_best_threshold(y_val, iso_val, beta=config.THRESHOLD_BETA)
        print(f"  contamination={contamination}: val F{config.THRESHOLD_BETA}={f_beta:.4f}")
        if best is None or f_beta > best["f_beta"]:
            best = {"contamination": contamination, "model": model, "f_beta": f_beta}
    return best["model"], best["contamination"]


def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Loading data...")
    train_files = load_split_per_file(config.TRAIN_PATH)
    val_files = load_split_per_file(config.VAL_PATH)
    test_files = load_split_per_file(config.TEST_PATH)
    val_labels = load_labels_per_file(config.VAL_LABELS_PATH)
    y_val = np.concatenate(val_labels)

    scaler = StandardScaler()
    scaler.fit(np.concatenate([df.values for df in train_files], axis=0))
    train_scaled = [scaler.transform(df.values) for df in train_files]

    print("Windowing train (per-run, no cross-run splicing)...")
    X_train, _ = create_windows_per_file(train_scaled, window_size=config.WINDOW_SIZE, step=config.STEP)
    X_train_iso = np.concatenate(
        [extract_window_features(arr, window_size=config.WINDOW_SIZE, step=config.STEP) for arr in train_scaled],
        axis=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_fn(model):
        """Fit the error model and score validation at per-timestep
        resolution during training, so the best checkpoint (not necessarily
        the last epoch) gets kept."""
        mu, precision = fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device)
        val_scores = score_cnn_densely(model, val_files, scaler, device, mu, precision)
        _, f_beta = find_best_threshold(y_val, val_scores, beta=config.THRESHOLD_BETA)
        return f_beta

    print(f"Training CNN for {config.EPOCHS} epochs (checking validation every {config.CNN_EVAL_EVERY})...")
    cnn = train_autoencoder(
        X_train, n_features=X_train.shape[2], device=device,
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        eval_every=config.CNN_EVAL_EVERY, eval_fn=eval_fn,
    )

    print("Sweeping Isolation Forest contamination on validation...")
    iso, best_contamination = sweep_iso_contamination(X_train_iso, val_files, scaler, y_val)
    print(f"Selected WINDOW_IF_CONTAMINATION={best_contamination}")

    print("Fitting Mahalanobis error model on confirmed-normal validation windows...")
    mu, precision = fit_mahalanobis_from_validation(cnn, val_files, val_labels, scaler, device)

    print("Scoring validation and test at per-timestep resolution...")
    cnn_val = score_cnn_densely(cnn, val_files, scaler, device, mu, precision)
    cnn_test = score_cnn_densely(cnn, test_files, scaler, device, mu, precision)
    iso_val = score_iso_densely(iso, val_files, scaler)
    iso_test = score_iso_densely(iso, test_files, scaler)

    cnn_val_n, cnn_test_n = normalize(cnn_val, cnn_test)
    iso_val_n, iso_test_n = normalize(iso_val, iso_test)

    print("Sweeping fusion weight on validation...")
    cnn_weight, best_threshold, _ = sweep_fusion_weight(y_val, cnn_val_n, iso_val_n)
    iso_weight = 1 - cnn_weight
    print(f"Selected CNN_WEIGHT={cnn_weight}, ISO_WEIGHT={iso_weight}")

    val_score = cnn_weight * cnn_val_n + iso_weight * iso_val_n
    test_score = cnn_weight * cnn_test_n + iso_weight * iso_test_n

    hybrid_metrics = compute_metrics(y_val, val_score, best_threshold)
    hybrid_metrics["cnn_weight"] = cnn_weight
    hybrid_metrics["iso_weight"] = iso_weight
    hybrid_metrics["iso_contamination"] = best_contamination
    hybrid_metrics["threshold_beta"] = config.THRESHOLD_BETA

    cnn_threshold, _ = find_best_threshold(y_val, cnn_val_n, beta=config.THRESHOLD_BETA)
    iso_threshold, _ = find_best_threshold(y_val, iso_val_n, beta=config.THRESHOLD_BETA)
    all_metrics = {
        "cnn_only": compute_metrics(y_val, cnn_val_n, cnn_threshold),
        "isolation_forest_only": compute_metrics(y_val, iso_val_n, iso_threshold),
        "hybrid": hybrid_metrics,
    }
    print(json.dumps(all_metrics, indent=2))

    preds = (test_score > best_threshold).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Building submission...")
    test_file_paths = sorted((config.TEST_PATH).glob("*.csv"))
    rows = []
    idx = 0
    for run_id, path in enumerate(test_file_paths, start=1):
        n_timesteps = len(pd.read_csv(path))
        for t in range(n_timesteps):
            rows.append((run_id, t, int(preds[idx])))
            idx += 1

    submission = pd.DataFrame(rows, columns=["run_id", "timestep", "prediction"])
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Saved:", SUBMISSION_PATH, submission.shape)

    return all_metrics


if __name__ == "__main__":
    main()
