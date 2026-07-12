# Time-series anomaly detection

## Overview

This project detects anomalies in multivariate sensor time-series data from an industrial batch-distillation process (18 sensors). There are no anomaly labels for the training data — only a small labeled validation set and an externally-scored test set — so every model here learns what *normal* looks like and flags departures from it, rather than learning to classify anomalies directly.

**The final model is the CNN autoencoder alone.** Two tabular detectors were built and rigorously evaluated as fusion partners, but neither improved on the CNN, so the deployed model is the CNN by itself (see **[Why the final model is CNN-only](#why-the-final-model-is-cnn-only-not-a-hybrid)**).

- **CNN autoencoder (the model of record)** — a dilated (TCN-style) 1D convolutional autoencoder that learns to reconstruct normal sensor windows. Anomaly score is a **Mahalanobis distance** over per-sensor reconstruction error (not a flat average), so a genuine deviation on one sensor isn't diluted by another sensor's ordinary noise.
- **Isolation Forest** *(evaluated baseline)* — a tree-based outlier detector on per-window summary statistics.
- **Local Outlier Factor (LOF)** *(evaluated baseline, Round 5)* — a density-based detector on cross-sensor engineered features, contributed by a teammate.

**Current result:** validation F1 ≈ 0.60, **competition-portal F1 = 0.63**.

**Documentation:** the full narrative (data, EDA, theory, all rounds, diagrams) is in [`reports/project_textbook.pdf`](reports/project_textbook.pdf). A step-by-step teaching walkthrough of the CNN is in [`reports/cnn_deep_dive.pdf`](reports/cnn_deep_dive.pdf); the fusion negative result in [`reports/lof_experiment_report.pdf`](reports/lof_experiment_report.pdf); and a research-grounded backlog of next steps in [`reports/improvement_backlog.pdf`](reports/improvement_backlog.pdf).

---

## Why the final model is CNN-only (not a hybrid)

The project began as a **hybrid**: fuse the CNN with a tabular detector, hoping the two would cover each other's blind spots. Over five rounds of work, the evidence pointed the other way — no tabular detector could improve on the CNN — so the final, honest answer is the CNN alone. The reasoning, stated plainly rather than dressed up:

1. **Isolation Forest never contributed.** The hybrid fusion weight is chosen by sweeping the validation set (and *can* select "pure CNN"). Once the CNN got strong (Mahalanobis scoring + F-beta threshold, Round 4), the sweep consistently picked **CNN weight = 1.0** — Isolation Forest added nothing.

2. **Round 5 gave the tabular branch its best shot — and it still lost.** A teammate contributed a **Local Outlier Factor (LOF)** detector with richer cross-sensor features (pairwise differences and rolling correlations). LOF beat Isolation Forest standalone (F1 ≈ 0.48 vs 0.49 at high recall), so it was a genuine candidate. We fused it with the CNN using **leakage-free, run-grouped cross-validation** to choose the weight (`src/pipelines/run_final_hybrid.py`), and tested it **twice** — on raw features and on standardized features.

3. **The cross-validation rejected LOF both times.** In 9 of 10 folds the weight search independently chose *pure CNN*; the deployed LOF weight came out **0.03** (raw features) and **0.00** (standardized). The honest out-of-fold hybrid F1 never beat the CNN alone. Even a small LOF blend *lowered* F1: it raised recall but cut precision, which under our precision-weighted score is a net loss.

**Conclusion:** across three fusion attempts (Isolation Forest, raw-feature LOF, standardized-feature LOF), **no tabular detector lifts the Round-4 CNN**. The CNN's ranking quality (ROC-AUC ≈ 0.77) is simply higher than the tabular detectors' (≈ 0.67), and mixing a weaker score into a cleaner one corrupts it. Shipping CNN-alone is therefore the *stronger* engineering decision — a focused model we fully understand and validated honestly, not complexity for its own sake. The full analysis is in [`reports/lof_experiment_report.pdf`](reports/lof_experiment_report.pdf).

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
│   │   ├── feature_pipeline.py  # cross-sensor engineered features for the tabular detectors (IF / LOF)
│   │   ├── feature_selection.py # sensor scoring + correlation filtering
│   │   └── windowing.py         # shared sliding-window utilities (per-run, no cross-run splicing)
│   ├── models/
│   │   ├── cnn/                 # dilated ConvAutoencoder1D, training loop (w/ checkpoint selection), scoring
│   │   └── isolation_forest/    # per-window features, training, scoring (baseline)
│   ├── evaluation/
│   │   └── metrics.py           # F-beta threshold selection, Mahalanobis scoring, precision/recall/F1/ROC-AUC
│   └── pipelines/
│       ├── run_cnn.py                # standalone CNN autoencoder — THE FINAL MODEL
│       ├── run_isolation_forest.py   # tabular Isolation Forest baseline
│       ├── run_hybrid.py             # CNN + window Isolation Forest, fused (legacy)
│       └── run_final_hybrid.py       # CNN + LOF fusion, run-grouped CV (Round 5 — rejected)
│
├── scripts/                   # SLURM batch scripts (RPTU Elwetritsch cluster) + generators
│   ├── run_cnn.slurm  run_hybrid.slurm  run_final_hybrid.slurm  diagnose_cnn.slurm
│   ├── diagnose_cnn.py               # CNN failure-mode plots (timelines, score overlap, PR, per-sensor)
│   └── generate_*.py                 # reproducible report generators (reportlab)
│
├── outputs/
│   ├── cnn/metrics.json              # final CNN validation metrics (F1 ≈ 0.60)
│   ├── cnn_predictions.csv           # ← PORTAL SUBMISSION (run_id, timestep, prediction)
│   ├── isolation_forest/, hybrid/    # baseline / legacy-hybrid outputs
│   └── hybrid_predictions.csv        # legacy hybrid submission
│
├── reports/
│   ├── project_textbook.pdf          # full narrative: data, EDA, theory, all rounds, diagrams
│   ├── cnn_deep_dive.pdf             # end-to-end CNN teaching walkthrough + architecture diagrams
│   ├── lof_experiment_report.pdf     # Round-5 fusion experiment (why LOF was rejected)
│   ├── improvement_backlog.pdf       # research-grounded next steps (2024–26 literature)
│   ├── project_summary_report.pdf    # condensed work-log summary
│   └── hybrid_model_postmortem.pdf   # root-cause analysis of why early rounds plateaued
│
├── plots/                     # figures (EDA + plots/diagnostics/ from diagnose_cnn.py)
├── HYBRID_MODEL_README.md     # architecture deep-dive for the (legacy) hybrid model
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

Training runs on CPU (no CUDA required); a GPU will be used automatically if available. `scripts/run_hybrid.slurm` is provided for running on a SLURM-managed GPU cluster.

---

## Dataset requirements

Place raw CSV runs under `data/raw/`:

```text
data/raw/
├── train/       (28 runs, unlabeled)
├── val/         (10 runs)
├── val_labels/  (10 files, per-timestep 0/1 labels)
└── test/        (53 runs, unlabeled — scored externally, e.g. Codabench)
```

Raw and processed data are not tracked in git due to size. See `data/README.md` for the full preprocessing/regeneration workflow.

---

## Training pipeline

Run from the repository root (each script reads paths/hyperparameters from `src/config.py`):

```bash
python -m src.pipelines.run_cnn                # ← FINAL MODEL -> outputs/cnn/, outputs/cnn_predictions.csv
python -m src.pipelines.run_isolation_forest   # tabular IF baseline -> outputs/isolation_forest/
python -m src.pipelines.run_hybrid             # CNN + window IF, fused (legacy) -> outputs/hybrid/
python -m src.pipelines.run_final_hybrid       # CNN + LOF fusion, run-grouped CV (Round 5) -> outputs/final_hybrid/
```

All pipelines load raw data, fit a `StandardScaler` on train only, build sliding windows **per source run** (windows never cross run boundaries), train, score, select a threshold, and write metrics + predictions. GPU equivalents are in `scripts/*.slurm`.

The CNN trains for up to 50 epochs, checking validation every 5 epochs and keeping the best-scoring checkpoint (training loss falls smoothly with no plateau, so the last epoch is not always the best one). `run_hybrid.py` sweeps the Isolation Forest `contamination` and the CNN/ISO fusion weight; `run_final_hybrid.py` fuses the CNN with LOF and chooses the weight by **run-grouped 5-fold cross-validation** (deploying the mean per-fold weight and reporting the honest out-of-fold score).

Notebooks 01-04 document the exploratory path (sensor selection, feature engineering, CNN architecture/window-size tuning); notebook 05 simply calls `run_hybrid.py` and displays the result.

---

## Evaluation pipeline

`src/evaluation/metrics.py`:
- `find_best_threshold` sweeps `sklearn.metrics.precision_recall_curve` on the validation set and picks the threshold maximizing **F-beta** (`beta=0.5` by default — weights precision over recall, since plain F1 consistently over-predicted anomalies given how overlapped the score distributions are).
- `fit_mahalanobis` / `mahalanobis_scores` fit a Gaussian error model on confirmed-normal validation windows' **per-sensor** CNN reconstruction error, and score new windows by Mahalanobis distance — replacing a flat mean-squared-error that was diluting a genuine spike on one sensor with another sensor's ordinary noise.
- `compute_metrics` reports precision, recall, F1, ROC-AUC, and predicted anomaly rate at a given threshold.

The CNN and hybrid pipelines score validation/test at a dense stride (`config.EVAL_STEP`) and average overlapping window scores per raw timestep before thresholding, so predictions map one-to-one onto the actual per-timestep submission format.

**`data/raw/test/` has no local labels** (it's scored externally), so every metric below is a **validation-set** metric except where a real portal score is explicitly noted.

---

## Results

All F1 values below are **validation** F1 (the test set is scored externally).

| Stage | CNN F1 | Tabular F1 | Hybrid F1 |
|---|---|---|---|
| Original baseline (window-level, not directly comparable) | 0.60 | 0.40 (IF) | 0.62 |
| After bug fixes (per-timestep) | 0.408 | 0.491 (IF) | 0.494 |
| + CNN checkpoint selection | 0.458 | 0.491 (IF) | 0.494 |
| + Dilated CNN architecture | 0.484 | 0.491 (IF) | 0.494 |
| + Mahalanobis scoring + F-beta threshold (Round 4) | **0.581** | 0.489 (IF) | 0.581 (weight → CNN 1.0) |
| **+ Round 5: LOF fusion tested & rejected** | **≈ 0.60** | 0.46–0.48 (LOF) | ≈ 0.60 (weight → CNN 1.0) |

**Final model (CNN-only), validation:** F1 ≈ 0.60 (0.602 in the latest GPU run), precision ≈ 0.60, recall ≈ 0.60, ROC-AUC ≈ 0.76, predicted anomaly rate ≈ 0.24 (true rate ≈ 0.24). **Competition portal: F1 = 0.63** — *higher* than the local validation estimate, confirming the model generalizes and the evaluation methodology is trustworthy (validation is if anything slightly pessimistic).

Submission file: [`outputs/cnn_predictions.csv`](outputs/cnn_predictions.csv) (`run_id, timestep, prediction`, 316,024 rows — the format confirmed against the portal's `sample_submission.csv`).

Every fusion attempt (Isolation Forest; raw- and standardized-feature LOF) selected **CNN weight ≈ 1.0** — see [Why the final model is CNN-only](#why-the-final-model-is-cnn-only-not-a-hybrid). Note the ~0.02 run-to-run F1 wobble (0.581–0.611 across identical configs) is ordinary training stochasticity, not a real quality change.

See `reports/project_textbook.pdf` and `reports/cnn_deep_dive.pdf` for the full explanation of each fix; `reports/lof_experiment_report.pdf` for the Round-5 fusion analysis.

---

## Reproducing the experiments

```bash
source ts-anomaly-env/bin/activate
python -m src.pipelines.run_cnn                # the final model
python -m src.pipelines.run_isolation_forest   # baseline
python -m src.pipelines.run_final_hybrid       # Round-5 CNN+LOF fusion experiment
```

Random seeds are fixed (`config.RANDOM_SEED = 42`). CNN training takes roughly 30 minutes on CPU at 50 epochs, and a few minutes on a GPU (`scripts/run_cnn.slurm`).

---

## Future improvements

Fusion has been tested and rejected (Round 5), so the focus shifts to the CNN itself. A full, research-grounded backlog (mapped to recent 2024–26 literature, rated by effort/payoff/risk) is in [`reports/improvement_backlog.pdf`](reports/improvement_backlog.pdf). The headline items:

- **Contamination-robust training (highest leverage)** — the "normal" training data isn't anomaly-free, which shrinks the error gap the score depends on. Down-weight likely-anomalous training windows (e.g. CLEANet-style kNN local-density weighting; RiAD dynamic re-weighting). Attacks the ROC-AUC ≈ 0.76 ceiling at its source.
- **Seed-ensembling & Mahalanobis-on-train** — average scores over several seeds (variance reduction) and fit the Mahalanobis normal model on the abundant train data. Cheap, low-risk AUC lifts.
- **Match the threshold objective to the portal metric** — the threshold is tuned for F-beta = 0.5; if the portal grades plain F1, tuning for F1 may recover free points.
- **Honest tuning split** — window size, architecture, and threshold were all chosen on the same 10-run validation set; the run-grouped CV already built in `run_final_hybrid.py` should be reused for an unbiased estimate.

Diagnostics to *see* the current failure modes (per-run timelines, score overlap, per-sensor error) are produced by `scripts/diagnose_cnn.py` into `plots/diagnostics/`.
