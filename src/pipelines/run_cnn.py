"""
Standalone CNN autoencoder pipeline.

Deduplicated from notebooks/04_cnn_tuning.ipynb, with fixes applied across
several rounds: windows respect run boundaries, val/test are scored densely
and expanded to per-timestep scores, training runs for the epoch count the
project's own tuning notebook validated with checkpoint selection on top
(see src/config.py), and (Round 4) the anomaly score is a Mahalanobis
distance over per-sensor reconstruction error rather than a flat mean,
with an F-beta (precision-weighted) threshold.

Run from the repository root: python -m src.pipelines.run_cnn
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

OUTPUT_DIR = config.OUTPUT_DIR / "cnn"


def fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device):
    """Fit a Gaussian error model on validation windows confirmed normal.

    Round 4: windows validation at train-time stride (config.STEP, not the
    dense EVAL_STEP - we just need a representative sample of normal
    windows, not every overlapping one), gets per-sensor reconstruction
    errors, and fits mu/precision only on windows with no anomalous
    timestep at all (window label == 0).
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


def score_split_densely(model, files, scaler, device, mu, precision):
    """Score each run at config.EVAL_STEP via Mahalanobis distance on
    per-sensor errors, and expand to one score per raw timestep."""
    per_timestep_scores = []
    for df in files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file(
            [scaled], window_size=config.WINDOW_SIZE, step=config.EVAL_STEP
        )
        window_errors = get_reconstruction_errors(
            model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True
        )
        window_scores = mahalanobis_scores(window_errors, mu, precision)
        timestep_scores = expand_window_scores_to_timesteps(
            window_scores, len(df), config.WINDOW_SIZE, config.EVAL_STEP
        )
        per_timestep_scores.append(timestep_scores)
    return np.concatenate(per_timestep_scores)


def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Loading data...")
    train_files = load_split_per_file(config.TRAIN_PATH)
    val_files = load_split_per_file(config.VAL_PATH)
    test_files = load_split_per_file(config.TEST_PATH)
    val_labels = load_labels_per_file(config.VAL_LABELS_PATH)

    print("Fitting scaler on train...")
    scaler = StandardScaler()
    scaler.fit(np.concatenate([df.values for df in train_files], axis=0))

    print("Windowing train (per-run, no cross-run splicing)...")
    train_scaled = [scaler.transform(df.values) for df in train_files]
    X_train, _ = create_windows_per_file(train_scaled, window_size=config.WINDOW_SIZE, step=config.STEP)
    print("Train windows:", X_train.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    y_val = np.concatenate(val_labels)

    def eval_fn(model):
        """Fit the error model and score validation at per-timestep
        resolution; used to keep the best checkpoint instead of blindly
        training to the last epoch."""
        mu, precision = fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device)
        val_scores = score_split_densely(model, val_files, scaler, device, mu, precision)
        _, f_beta = find_best_threshold(y_val, val_scores, beta=config.THRESHOLD_BETA)
        return f_beta

    print(f"Training for {config.EPOCHS} epochs (checking validation every {config.CNN_EVAL_EVERY})...")
    model = train_autoencoder(
        X_train,
        n_features=X_train.shape[2],
        device=device,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        eval_every=config.CNN_EVAL_EVERY,
        eval_fn=eval_fn,
    )

    print("Fitting Mahalanobis error model on confirmed-normal validation windows...")
    mu, precision = fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device)

    print("Scoring validation and test at per-timestep resolution...")
    val_scores = score_split_densely(model, val_files, scaler, device, mu, precision)
    test_scores = score_split_densely(model, test_files, scaler, device, mu, precision)

    best_threshold, _ = find_best_threshold(y_val, val_scores, beta=config.THRESHOLD_BETA)
    metrics = compute_metrics(y_val, val_scores, best_threshold)
    metrics["threshold_beta"] = config.THRESHOLD_BETA
    print(json.dumps(metrics, indent=2))

    preds = (test_scores > best_threshold).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    output = pd.DataFrame({
        "timestep_index": np.arange(len(preds)),
        "mahalanobis_score": test_scores,
        "is_anomaly": preds,
    })
    output.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    return metrics


if __name__ == "__main__":
    main()
