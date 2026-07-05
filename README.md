# Time-series anomaly detection

## Overview

This project detects anomalies in multivariate sensor time-series data using two complementary approaches, combined into a hybrid ensemble:

- **CNN autoencoder** — an unsupervised 1D convolutional autoencoder that learns to reconstruct normal sensor windows. High reconstruction error signals a temporal anomaly (e.g. slow drift).
- **Isolation Forest** — a statistical outlier detector, run both on tabular lag/rolling features (per-row baseline) and on per-window summary statistics (used in the hybrid).
- **Hybrid fusion** — both models' scores are normalized and combined into a single anomaly score, with the fusion weight chosen on the validation set rather than fixed by hand.

Each raw file is one independent time-series run (e.g. an industrial process cycle); the task is to flag anomalous timesteps within each run.

---

## Repository structure

```text
ts-anomaly-detection/
├── data/
│   ├── raw/                  # train/val/val_labels/test CSV runs (not tracked in git)
│   ├── processed/            # sensor feature matrix + selected_sensors.json
│   ├── artifacts/            # exploratory analysis outputs (KDE groupings, etc.)
│   └── README.md
│
├── notebooks/
│   ├── 01_sensor_selection.ipynb
│   ├── 02_anomaly_detection_isolation_forest.ipynb   # tabular Isolation Forest baseline
│   ├── 03_cnn_baseline.ipynb                          # first CNN autoencoder pass
│   ├── 04_cnn_tuning.ipynb                            # window/stride/epoch experiments
│   ├── 05_hybrid_model.ipynb                          # thin runner for src/pipelines/run_hybrid.py
│   └── archive/               # earlier EDA/feature-engineering drafts, kept for reference
│
├── src/
│   ├── config.py               # all hyperparameters and paths in one place
│   ├── core/
│   │   ├── data_loader.py       # CSV loading/cleaning helpers (sensor-selection path)
│   │   ├── feature_pipeline.py  # lag/rolling/EWMA/interaction features for Isolation Forest
│   │   ├── feature_selection.py # sensor scoring + correlation filtering
│   │   └── windowing.py         # shared sliding-window utilities (both model branches)
│   ├── models/
│   │   ├── cnn/                 # ConvAutoencoder1D, training loop, scoring
│   │   └── isolation_forest/    # per-window features, training, scoring
│   ├── evaluation/
│   │   └── metrics.py           # threshold selection (best F1) + precision/recall/F1/ROC-AUC
│   └── pipelines/
│       ├── run_isolation_forest.py   # standalone tabular Isolation Forest baseline
│       ├── run_cnn.py                # standalone CNN autoencoder
│       └── run_hybrid.py             # CNN + window Isolation Forest, fused
│
├── outputs/
│   ├── isolation_forest/{metrics,predictions}/
│   ├── cnn/metrics.json, predictions.csv
│   ├── hybrid/metrics.json
│   └── hybrid_predictions.csv        # final run_id, timestep, prediction submission
│
├── plots/                     # figures referenced by the notebooks
├── HYBRID_MODEL_README.md     # architecture deep-dive for the hybrid model
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv ts-anomaly-env
source ts-anomaly-env/bin/activate   # Windows: ts-anomaly-env\Scripts\activate
pip install -r requirements.txt
```

Training runs on CPU (no CUDA required); a GPU will be used automatically if available.

---

## Dataset requirements

Place raw CSV runs under `data/raw/`:

```text
data/raw/
├── train/       (28 runs, unlabeled)
├── val/         (10 runs)
├── val_labels/  (10 files, per-timestep 0/1 labels)
└── test/        (53 runs, unlabeled - scored externally, e.g. Codabench)
```

Raw and processed data are not tracked in git due to size. See `data/README.md` for the full preprocessing/regeneration workflow.

---

## Training pipeline

Run from the repository root (each script reads paths/hyperparameters from `src/config.py`):

```bash
python -m src.pipelines.run_isolation_forest   # tabular baseline -> outputs/isolation_forest/
python -m src.pipelines.run_cnn                # CNN autoencoder  -> outputs/cnn/
python -m src.pipelines.run_hybrid             # CNN + window IF, fused -> outputs/hybrid/, outputs/hybrid_predictions.csv
```

All three are self-contained: they load raw data, fit a `StandardScaler` on train only, build sliding windows **per source run** (windows never cross run boundaries), train, score, select a threshold, and write metrics + predictions.

Notebooks 01-04 document the exploratory path (sensor selection, feature engineering, CNN architecture/window-size tuning); notebook 05 simply calls `run_hybrid.py` and displays the result.

---

## Evaluation pipeline

`src/evaluation/metrics.py::find_best_threshold` sweeps `sklearn.metrics.precision_recall_curve` on the validation set and picks the threshold maximizing F1; `compute_metrics` then reports precision, recall, F1, ROC-AUC, and predicted anomaly rate at that threshold.

For the CNN and hybrid pipelines, validation and test are scored at a dense stride (`config.EVAL_STEP`) and overlapping window scores are averaged per raw timestep before thresholding — so predictions map one-to-one onto the actual per-timestep submission format, rather than broadcasting one label across an entire window.

**`data/raw/test/` has no local labels** (it's scored externally), so every metric below is a **validation-set** metric, not a test metric.

---

## Results

Baseline (as originally reported, at window-level granularity) vs. current (validation, per-timestep granularity — a stricter metric, see caveat above):

| Model | Baseline F1 | Current F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Isolation Forest (tabular) | 0.40 | 0.437 | 0.30 | 0.83 | 0.585 |
| Isolation Forest (window-based, used in hybrid) | — | 0.491 | 0.33 | 0.92 | 0.649 |
| CNN autoencoder | 0.60 | 0.408 | 0.30 | 0.64 | 0.574 |
| Hybrid (fused) | 0.62 | 0.494 | 0.34 | 0.92 | 0.636 |

The Isolation Forest improvement is attributable to fixing its per-window features to be computed per sensor (`axis=0`) instead of flattening all 18 sensors into one scalar per statistic. The CNN's drop reflects evaluating at a genuinely harder granularity (per-timestep instead of per-window) rather than a regression in the model itself — see "Future improvements" below.

---

## Reproducing the experiments

```bash
source ts-anomaly-env/bin/activate
python -m src.pipelines.run_isolation_forest
python -m src.pipelines.run_cnn
python -m src.pipelines.run_hybrid
```

Random seeds are fixed (`config.RANDOM_SEED = 42`) for the scaler, Isolation Forest, and CNN initialization/shuffling. CNN training takes roughly 30 minutes on CPU at the configured 50 epochs.

---

## Future improvements

- **Re-tune the CNN against the per-timestep metric.** 50 epochs was validated against the old window-level F1 in `notebooks/04_cnn_tuning.ipynb`; it hasn't been re-swept against the stricter per-timestep evaluation now in place, and the current result suggests it may be overtraining (smoothly falling reconstruction loss with no plateau across all 50 epochs).
- **Contamination-robust training** — train isn't guaranteed anomaly-free; consider iteratively down-weighting the highest-error windows during training.
- **Deeper temporal receptive field** — dilated Conv1D layers to capture longer-range drift without enlarging the window or memory footprint.
- **Learned fusion** — replace the small fixed-grid weight sweep with a logistic regression over `[cnn_score, iso_score]`.
- **Held-out tuning split** — window size, stride, and epoch count have all been selected by watching the same 10-run validation set; a proper k-fold or separate tuning split would give a more honest read on generalization before submission.
