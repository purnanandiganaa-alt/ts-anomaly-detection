"""
Standalone CNN autoencoder pipeline.

Deduplicated from notebooks/04_cnn_tuning.ipynb, with the Phase 2 fixes
applied: windows respect run boundaries, val/test are scored densely and
expanded to per-timestep scores, and training runs for the epoch count the
project's own tuning notebook already validated (see src/config.py).

Run from the repository root: python -m src.pipelines.run_cnn
"""

import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src import config
from src.core.windowing import (
    create_windows_per_file,
    expand_window_scores_to_timesteps,
    load_labels_per_file,
    load_split_per_file,
)
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.models.cnn.predict import get_reconstruction_errors
from src.models.cnn.train import train_autoencoder

OUTPUT_DIR = config.OUTPUT_DIR / "cnn"


def score_split_densely(model, files, scaler, device):
    """Score each run at config.EVAL_STEP and expand to one score per raw timestep."""
    per_timestep_scores = []
    for df in files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file(
            [scaled], window_size=config.WINDOW_SIZE, step=config.EVAL_STEP
        )
        window_scores = get_reconstruction_errors(model, windows, device, batch_size=config.BATCH_SIZE)
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
        """Score validation at per-timestep resolution; used to keep the
        best checkpoint instead of blindly training to the last epoch."""
        val_scores = score_split_densely(model, val_files, scaler, device)
        _, f1 = find_best_threshold(y_val, val_scores)
        return f1

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

    print("Scoring validation and test at per-timestep resolution...")
    val_scores = score_split_densely(model, val_files, scaler, device)
    test_scores = score_split_densely(model, test_files, scaler, device)

    best_threshold, _ = find_best_threshold(y_val, val_scores)
    metrics = compute_metrics(y_val, val_scores, best_threshold)
    print(json.dumps(metrics, indent=2))

    preds = (test_scores > best_threshold).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    import pandas as pd

    output = pd.DataFrame({
        "timestep_index": np.arange(len(preds)),
        "reconstruction_error": test_scores,
        "is_anomaly": preds,
    })
    output.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    return metrics


if __name__ == "__main__":
    main()
