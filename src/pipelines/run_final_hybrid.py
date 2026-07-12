"""
Final hybrid: Round-4 Mahalanobis CNN + LOF (Local Outlier Factor), fused.

Round 5. Adapted from a teammate's CNN+LOF fusion (originally built on the
pre-Round-4 flat-MSE CNN) so it runs against the current Mahalanobis CNN and
the project's F-beta=0.5 threshold. Isolation Forest is dropped from the fused
model - it has taken ~0 weight in every prior sweep. LOF is a density-based
detector (different inductive bias from IF's tree-based isolation), fed the
same tabular features as IF.

Two design choices, both to avoid overfitting the single fusion parameter to
the small (10-run) validation set:
  1. Only ONE free parameter: cnn_weight (lof_weight = 1 - cnn_weight).
  2. The deployed weight is the MEAN of the weight chosen independently in each
     run-grouped CV fold, not a fresh re-optimization on all of validation.
     Out-of-fold (OOF) metrics are reported as the honest hybrid score.

Run from the repository root: python -m src.pipelines.run_final_hybrid
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src import config
from src.core.feature_pipeline import create_features
from src.core.windowing import (
    create_windows_per_file,
    load_labels_per_file,
    load_split_per_file,
)
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.models.cnn.train import train_autoencoder
# Reuse the Round-4 Mahalanobis CNN scoring path and score normalization.
from src.pipelines.run_hybrid import (
    fit_mahalanobis_from_validation,
    normalize,
    score_cnn_densely,
)

OUTPUT_DIR = config.OUTPUT_DIR / "final_hybrid"
SUBMISSION_PATH = config.OUTPUT_DIR / "final_hybrid_predictions.csv"
BETA = config.THRESHOLD_BETA


def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Loading data...")
    train_files = load_split_per_file(config.TRAIN_PATH)
    val_files = load_split_per_file(config.VAL_PATH)
    test_files = load_split_per_file(config.TEST_PATH)
    val_labels = load_labels_per_file(config.VAL_LABELS_PATH)
    y_val = np.concatenate(val_labels)
    # Per-timestep run id, so CV folds never split a run across train/test.
    val_run_ids = np.concatenate([np.full(len(df), i) for i, df in enumerate(val_files)])

    print("Fitting scaler on train...")
    scaler = StandardScaler()
    scaler.fit(np.concatenate([df.values for df in train_files], axis=0))
    train_scaled = [scaler.transform(df.values) for df in train_files]

    print("Windowing train (per-run, no cross-run splicing)...")
    X_train, _ = create_windows_per_file(train_scaled, window_size=config.WINDOW_SIZE, step=config.STEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    def eval_fn(model):
        """Best-checkpoint selection: fit the Mahalanobis error model and score
        validation at per-timestep resolution, return F-beta."""
        mu, precision = fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device)
        val_scores = score_cnn_densely(model, val_files, scaler, device, mu, precision)
        _, f_beta = find_best_threshold(y_val, val_scores, beta=BETA)
        return f_beta

    print(f"Training CNN for {config.EPOCHS} epochs (checking validation every {config.CNN_EVAL_EVERY})...")
    cnn = train_autoencoder(
        X_train, n_features=X_train.shape[2], device=device,
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        eval_every=config.CNN_EVAL_EVERY, eval_fn=eval_fn,
    )

    print("Fitting Mahalanobis error model on confirmed-normal validation windows...")
    mu, precision = fit_mahalanobis_from_validation(cnn, val_files, val_labels, scaler, device)

    print("Scoring CNN at per-timestep resolution...")
    cnn_val = score_cnn_densely(cnn, val_files, scaler, device, mu, precision)
    cnn_test = score_cnn_densely(cnn, test_files, scaler, device, mu, precision)

    print("Building tabular features + training LOF (CPU; may be slow on wide features)...")
    X_train_tab = create_features(str(config.TRAIN_PATH), str(config.SELECTED_SENSORS_PATH))
    X_val_tab = create_features(str(config.VAL_PATH), str(config.SELECTED_SENSORS_PATH))
    X_test_tab = create_features(str(config.TEST_PATH), str(config.SELECTED_SENSORS_PATH))

    # Standardize tabular features on TRAIN stats only (no leakage). LOF is a
    # distance-based detector: on the raw features the large-magnitude pairwise
    # product columns dominate the distance metric and swamp everything else,
    # which crippled LOF's quality. (StandardScaler leaves zero-variance
    # columns untouched, so no div-by-zero.) Also emits plain arrays, which
    # silences sklearn's "X has no valid feature names" warning.
    tab_scaler = StandardScaler()
    X_train_tab = tab_scaler.fit_transform(X_train_tab)
    X_val_tab = tab_scaler.transform(X_val_tab)
    X_test_tab = tab_scaler.transform(X_test_tab)

    lof = LocalOutlierFactor(n_neighbors=config.LOF_N_NEIGHBORS, novelty=True, n_jobs=-1)
    lof.fit(X_train_tab)
    # LOF score_samples: higher = more normal, so negate for an anomaly score.
    lof_val = -lof.score_samples(X_val_tab)
    lof_test = -lof.score_samples(X_test_tab)

    if len(lof_val) != len(cnn_val) or len(cnn_val) != len(y_val):
        raise ValueError(
            f"Length mismatch: CNN val={len(cnn_val)}, LOF val={len(lof_val)}, labels={len(y_val)}. "
            "The two branches must produce one score per validation timestep."
        )

    print("Normalizing score arrays (val-stats only, no test leakage)...")
    cnn_val_n, cnn_test_n = normalize(cnn_val, cnn_test)
    lof_val_n, lof_test_n = normalize(lof_val, lof_test)

    def blend(w_cnn, cnn_s, lof_s):
        return w_cnn * cnn_s + (1 - w_cnn) * lof_s

    print(f"Run-grouped {config.FUSION_CV_FOLDS}-fold CV: best cnn_weight per fold (beta={BETA})...")
    gkf = GroupKFold(n_splits=config.FUSION_CV_FOLDS)
    fold_best_weights = []
    oof_scores = np.zeros(len(y_val))

    for tr_idx, te_idx in gkf.split(cnn_val_n.reshape(-1, 1), y_val, groups=val_run_ids):
        best_f_beta, best_w = -1.0, None
        for w_cnn in config.FUSION_WEIGHT_GRID:
            b = blend(w_cnn, cnn_val_n[tr_idx], lof_val_n[tr_idx])
            _, f_beta = find_best_threshold(y_val[tr_idx], b, beta=BETA)
            if f_beta > best_f_beta:
                best_f_beta, best_w = f_beta, w_cnn
        fold_best_weights.append(best_w)
        oof_scores[te_idx] = blend(best_w, cnn_val_n[te_idx], lof_val_n[te_idx])

    print(f"Per-fold chosen cnn_weight: {[round(float(w), 2) for w in fold_best_weights]}")
    print(f"  mean={np.mean(fold_best_weights):.3f}  std={np.std(fold_best_weights):.3f}")

    oof_threshold, _ = find_best_threshold(y_val, oof_scores, beta=BETA)
    oof_metrics = compute_metrics(y_val, oof_scores, oof_threshold)

    final_w_cnn = float(np.mean(fold_best_weights))
    print(f"Final deployed weight (mean across folds): cnn={final_w_cnn:.3f}, lof={1 - final_w_cnn:.3f}")

    final_val_scores = blend(final_w_cnn, cnn_val_n, lof_val_n)
    final_threshold, _ = find_best_threshold(y_val, final_val_scores, beta=BETA)
    final_metrics = compute_metrics(y_val, final_val_scores, final_threshold)

    cnn_threshold, _ = find_best_threshold(y_val, cnn_val_n, beta=BETA)
    lof_threshold, _ = find_best_threshold(y_val, lof_val_n, beta=BETA)
    all_metrics = {
        "cnn_only": compute_metrics(y_val, cnn_val_n, cnn_threshold),
        "lof_only": compute_metrics(y_val, lof_val_n, lof_threshold),
        "hybrid_honest_oof": oof_metrics,
        "hybrid_averaged_weight_on_full_val": final_metrics,
        "final_weight_cnn": final_w_cnn,
        "final_weight_lof": 1 - final_w_cnn,
        "per_fold_weights": [round(float(w), 3) for w in fold_best_weights],
        "lof_n_neighbors": config.LOF_N_NEIGHBORS,
        "threshold_beta": BETA,
    }
    print(json.dumps(all_metrics, indent=2))

    # Deploy with the averaged weight; threshold chosen on full validation.
    final_test_scores = blend(final_w_cnn, cnn_test_n, lof_test_n)
    test_pred = (final_test_scores > final_threshold).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Building submission...")
    test_file_paths = sorted(config.TEST_PATH.glob("*.csv"))
    rows = []
    idx = 0
    for run_id, path in enumerate(test_file_paths, start=1):
        n_timesteps = len(pd.read_csv(path))
        for t in range(n_timesteps):
            rows.append((run_id, t, int(test_pred[idx])))
            idx += 1

    submission = pd.DataFrame(rows, columns=["run_id", "timestep", "prediction"])
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Saved:", SUBMISSION_PATH, submission.shape)

    return all_metrics


if __name__ == "__main__":
    main()
