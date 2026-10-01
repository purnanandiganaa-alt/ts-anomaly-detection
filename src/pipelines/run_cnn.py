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
SUBMISSION_PATH = config.OUTPUT_DIR / "cnn_predictions.csv"


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


def score_split_per_run(model, files, scaler, device, mu, precision):
    """Like score_split_densely, but returns the per-run list (keeps run
    boundaries so scores can be normalized per run before thresholding)."""
    per_run = []
    for df in files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file([scaled], window_size=config.WINDOW_SIZE, step=config.EVAL_STEP)
        we = get_reconstruction_errors(model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True)
        ws = mahalanobis_scores(we, mu, precision)
        per_run.append(expand_window_scores_to_timesteps(ws, len(df), config.WINDOW_SIZE, config.EVAL_STEP))
    return per_run


def robust_normalize_per_run(per_run_scores):
    """Robust z-score each run's scores by its own median and MAD.

    Diagnostic finding: raw Mahalanobis magnitudes vary ~100x across runs, so a
    single global threshold misses whole runs (two val runs scored F1=0 despite
    the model peaking correctly on their anomalies). Normalizing each run to its
    own baseline makes one global threshold comparable across runs. Unsupervised,
    so it applies identically to the unlabeled test runs.

    Caveat: assumes each run contains some anomalies; an entirely-normal run has
    its ordinary fluctuations amplified, which can create false positives.
    """
    out = []
    for s in per_run_scores:
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        out.append((s - med) / (1.4826 * mad + 1e-9))
    return np.concatenate(out)


def fit_mahalanobis_from_train(model, train_files, scaler, device):
    """Round 6 (B): fit the normal error model on TRAIN windows instead of
    confirmed-normal validation windows.

    The validation fit sees only the normal windows of 10 labeled runs; train
    has ~18k windows. More rows give a better-conditioned 18x18 covariance,
    and it removes the last place validation labels enter the *score* itself
    (they still choose the threshold). Caveat: train is assumed normal but is
    not guaranteed anomaly-free - the known contamination issue.
    """
    errors = []
    for df in train_files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file([scaled], window_size=config.WINDOW_SIZE, step=config.STEP)
        errors.append(
            get_reconstruction_errors(model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True)
        )
    return fit_mahalanobis(np.concatenate(errors, axis=0))


def compute_errors_per_run(model, files, scaler, device):
    """Dense per-sensor reconstruction errors per run, computed ONCE so several
    Mahalanobis models can be compared without re-running inference."""
    out = []
    for df in files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file([scaled], window_size=config.WINDOW_SIZE, step=config.EVAL_STEP)
        we = get_reconstruction_errors(model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True)
        out.append((we, len(df)))
    return out


def scores_from_errors(errors_per_run, mu, precision):
    """Apply one Mahalanobis model to cached per-run errors -> per-run timestep scores."""
    return [
        expand_window_scores_to_timesteps(
            mahalanobis_scores(we, mu, precision), n, config.WINDOW_SIZE, config.EVAL_STEP
        )
        for we, n in errors_per_run
    ]


def build_submission(preds, test_file_paths, path):
    """Write a per-run run_id/timestep/prediction portal submission."""
    rows, idx = [], 0
    for run_id, p in enumerate(test_file_paths, start=1):
        n = len(pd.read_csv(p))
        for t in range(n):
            rows.append((run_id, t, int(preds[idx]))); idx += 1
    pd.DataFrame(rows, columns=["run_id", "timestep", "prediction"]).to_csv(path, index=False)
    return idx


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

    # Apple-Silicon GPU (MPS) as a middle tier between CUDA and CPU; the
    # cluster path (cuda) is unchanged.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

    MODEL_PATH = OUTPUT_DIR / "model.pt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved checkpoint:", MODEL_PATH)

    # Two Mahalanobis normal models, same trained network:
    #   A = confirmed-normal validation windows (previous behaviour)
    #   B = train windows (Round 6 experiment)
    print("Fitting Mahalanobis error models (A: val-normal, B: train)...")
    mu_a, prec_a = fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device)
    mu_b, prec_b = fit_mahalanobis_from_train(model, train_files, scaler, device)

    print("Computing dense reconstruction errors once for val and test...")
    val_errors = compute_errors_per_run(model, val_files, scaler, device)
    test_errors = compute_errors_per_run(model, test_files, scaler, device)

    variants, test_preds = {}, {}
    for mname, (mu, prec) in [("mahalA_val", (mu_a, prec_a)), ("mahalB_train", (mu_b, prec_b))]:
        val_per_run = scores_from_errors(val_errors, mu, prec)
        test_per_run = scores_from_errors(test_errors, mu, prec)
        pairs = [
            ("raw", np.concatenate(val_per_run), np.concatenate(test_per_run)),
            ("per_run_norm", robust_normalize_per_run(val_per_run), robust_normalize_per_run(test_per_run)),
        ]
        for sname, vs, ts in pairs:
            for beta in (0.5, 1.0):
                thr, _ = find_best_threshold(y_val, vs, beta=beta)
                m = compute_metrics(y_val, vs, thr)
                m["threshold_beta"] = beta
                preds = (ts > thr).astype(int)
                # The val/test score-shift diagnostic: the same absolute cutoff
                # flags a very different fraction of test than of validation.
                m["test_flagged_rate"] = float(preds.mean())
                key = f"{mname}_{sname}_beta{beta}"
                variants[key] = m
                test_preds[key] = preds

                # Rate-matched alternative: keep the validation operating POINT
                # (fraction flagged) rather than the absolute cutoff. Neutral on
                # validation by construction; only the portal can judge it.
                q_thr = float(np.quantile(ts, 1.0 - m["anomaly_rate"]))
                test_preds[key + "_ratematched"] = (ts > q_thr).astype(int)

    best_key = max(variants, key=lambda k: variants[k]["f1"])
    all_metrics = {
        "variants": variants,
        "best_variant_by_val_f1": best_key,
        "notes": {
            "mahalA_val": "normal error model fit on confirmed-normal validation windows (previous behaviour)",
            "mahalB_train": "normal error model fit on all train windows (Round 6 experiment B)",
            "checkpoint_selection": "unchanged (F-beta on the val-fit model), so A and B share one network",
            "test_flagged_rate": "fraction of test timesteps flagged; validation anomaly rate is ~0.24",
            "ratematched": "test threshold set by quantile to reproduce the validation flagged rate; no local evidence, portal-only hypothesis",
        },
    }
    print(json.dumps(all_metrics, indent=2))
    print("Best validation F1 variant:", best_key, "F1 =", round(variants[best_key]["f1"], 4))

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    # Candidate submissions go to outputs/candidates/ - the active portal file
    # outputs/cnn_predictions.csv is NOT overwritten here.
    print("Building candidate submissions...")
    cand_dir = config.OUTPUT_DIR / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    test_file_paths = sorted(config.TEST_PATH.glob("*.csv"))
    for key in [best_key, best_key + "_ratematched", "mahalB_train_raw_beta1.0",
                "mahalB_train_raw_beta1.0_ratematched", "mahalB_train_raw_beta0.5"]:
        if key in test_preds:
            build_submission(test_preds[key], test_file_paths, cand_dir / f"cnn_{key}.csv")
    print("Wrote candidates to", cand_dir)

    return all_metrics


if __name__ == "__main__":
    main()
