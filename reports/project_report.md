# Time-Series Anomaly Detection

**Unsupervised detection of anomalous timesteps in multivariate batch-distillation sensor data with a dilated convolutional autoencoder and per-sensor Mahalanobis scoring**

Group ML_DL_5: Bhuvana Shikaripura Dasharatha, Purna Chandra Nandigana
Master-Project: Machine Learning 26, RPTU Kaiserslautern-Landau
Report compiled 30 Sep 2026 from the project repository (`main` branch at commit `129f112`, plus uncommitted working-tree changes dated 15 Sep 2026)

---

### How to read this report

Every factual statement in this report was checked against a project artifact. Sources are cited in square brackets, e.g. `[outputs/cnn/metrics.json @ HEAD]`. Where a statement rests on a secondary source (a PDF report or README written during the project) rather than on a primary artifact (code, a metrics file, a run log, a data file), this is stated. Evidence levels are marked where it matters:

| Tag | Meaning |
|---|---|
| **[verified]** | Recomputed or read directly from code, data, a metrics JSON, a run log, or git history during preparation of this report |
| **[documented]** | Stated in a project report/README/EXPERIMENTS.md written at the time; the underlying run output is not stored in the repository |
| **[interpretation]** | A reasoned explanation of a measured result; not isolated by a controlled experiment |
| **[unknown]** | Not available / not verified from the project artifacts |

**Validation F1** (computed locally on the 10 labelled validation runs) and **portal F1** (computed by the external Codabench competition server on the 53 unlabelled test runs) are different quantities and are never used interchangeably in this report.

---

## Abstract

We detect anomalous timesteps in 18-sensor recordings of a laboratory batch-distillation process, sampled at 1 Hz. The 28 training runs carry no labels (the organizers describe them as normal runs), 10 validation runs carry per-timestep labels, and 53 test runs are scored externally on a Codabench portal. We therefore treat the task as novelty detection. The final system is a dilated residual 1D convolutional autoencoder (59,858 parameters) trained on 200-timestep windows of standardized training data. For each window, its per-sensor reconstruction error forms an 18-dimensional vector. This vector is scored by squared Mahalanobis distance under a Ledoit–Wolf Gaussian model fitted on label-confirmed normal validation windows. Window scores are averaged to per-timestep scores, and a single global threshold that maximizes F<sub>0.5</sub> on validation turns them into binary predictions. The submission produced by this pipeline scored **F1 = 0.63 on the competition portal**, the best externally evaluated result of the project. The same run's validation metrics were F1 = 0.6022 and ROC-AUC = 0.7629.

Across five rounds of development, the change that moved the threshold-free ranking quality most was replacing flat reconstruction error with per-sensor Mahalanobis scoring: validation ROC-AUC went from 0.605 to 0.768 [documented]. Checkpoint selection and a dilated architecture each gave smaller standalone gains. Fusion with Isolation Forest and Local Outlier Factor detectors was tested three times and rejected. After the presentation checkpoint, three further changes were tested on one shared network and none was adopted: an F1-optimal threshold, per-run median/MAD score normalization, and a train-fitted Mahalanobis model. A leave-one-run-out analysis showed that the standard validation protocol is optimistic, by 0.070 ROC-AUC and 0.058 F1.

---

## 1. Introduction

Industrial processes are monitored by many sensors. Faults often show up as a departure from the joint behaviour of several sensors rather than as one out-of-range reading. This project addresses per-timestep anomaly detection in such data for a university competition. The competition is hosted on Codabench, and the organizers supplied a public data package with a manifest and a starter README [verified: `batch_distillation_public_data/manifest.json` and `readme.pdf` in git commit `42901df`].

The defining constraint is the label situation. The training split has no labels, and only 10 validation runs are labelled. A supervised classifier cannot be trained on anomalies. Every detector in this project instead models normal behaviour and scores departures from it.

The project went through five main development rounds before the presentation checkpoint on 13 Jul 2026, plus a set of controlled follow-up experiments on 15 Sep 2026. The main work was:

1. building and debugging a CNN-autoencoder + Isolation Forest hybrid;
2. fixing six correctness bugs in the original notebook-based implementation;
3. improving the CNN through checkpoint selection, a dilated architecture and Mahalanobis scoring;
4. testing and rejecting score fusion with tabular detectors;
5. testing and rejecting three further scoring and thresholding changes, and quantifying how optimistic the validation protocol is.

---

## 2. Problem Definition

**Input.** A set of independent runs. Each run is a multivariate time series $X \in \mathbb{R}^{L \times 18}$ with one row per timestep (1 Hz) and one column per sensor. $L$ varies by run.

**Output.** For every timestep of every test run, a binary label $\hat{y}_t \in \{0, 1\}$ (0 = normal, 1 = anomalous). The required upload format is one CSV with columns `run_id, timestep, prediction`, where `timestep` is the zero-based row index within the run and every timestep of every test run appears exactly once [verified: organizer `readme.pdf`, commit `42901df`].

**Evaluation metric.** The organizer README states that "Codabench evaluates submissions with binary F1-score" [verified: organizer `readme.pdf`, commit `42901df`, page 1]. The README does not say whether F1 is computed point-wise over all timesteps, per run, event-wise or point-adjusted. It also says the hidden test set contains a public and a private leaderboard split, both scored from the same uploaded CSV. Which runs belong to which split is not revealed [verified: same source]. Section 9.4 discusses what this implies for interpreting the portal score.

**Anomaly definition.** Anomalies are defined solely by the organizers' per-timestep validation labels (`val_labels/series_*.csv`, columns `timestep, label`). The README says the label is derived from "Label (common/all)". The physical meaning of an anomaly (fault type, root cause) is not documented [unknown].

---

## 3. Dataset

### 3.1 Source and structure

| Property | Value | Source |
|---|---|---|
| Dataset name | `lab_batch_distillation_task_02`, "organizer-selected ternary batch-distillation subset" | manifest, commit `42901df` [verified] |
| Sensors | 18: LS701, LS702, T701, T702, T703, T704, T706, T708, T709, T711, T712, T705, FT703, FT704, PDI701, PDI702, PY23, FYI702 | manifest + CSV headers [verified] |
| Sensor types | 2 levels, 9 temperatures, 2 mass flows (product, reflux), 1 differential pressure, 2 pressures, 1 cooling-medium flow | manifest `feature_descriptions` [verified] |
| Sampling rate | 1.0 Hz | manifest [verified] |
| Missing values | 0 in train, val and test | recomputed [verified] |

### 3.2 Splits

All counts below were recomputed from `data/raw/` during preparation of this report [verified].

| Split | Runs | Timesteps | Run length (min–max) | Labels | Use in this project |
|---|---|---|---|---|---|
| `train/` | 28 | 189,444 | 742 – 20,489 | none ("28 normal runs" per organizer README) | fit the scaler; train the autoencoder; fit IF/LOF |
| `val/` + `val_labels/` | 10 | 79,688 | 2,070 – 12,683 | per-timestep 0/1 | checkpoint selection, Mahalanobis fit, threshold selection, all reported local metrics |
| `test/` | 53 | 316,024 | 1,213 – 19,825 | none locally | portal submission only |

**Validation labels.** 19,086 of 79,688 timesteps are anomalous, a prevalence of 0.2395. This matches the manifest's `num_positive_labels` per run exactly [verified]. Prevalence varies strongly across runs:

| Val run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Timesteps | 8,414 | 11,195 | 2,070 | 7,353 | 7,156 | 8,126 | 8,896 | 6,117 | 7,678 | 12,683 |
| Anomaly rate | 0.376 | 0.031 | 0.411 | 0.008 | 0.024 | 0.215 | 0.141 | 0.814 | 0.633 | 0.131 |

The validation labels contain 28 contiguous anomalous segments in total [verified: counted 0→1 transitions]. The organizer README calls the validation runs "10 labeled anomalous runs"; every one contains at least one anomalous segment.

**Class imbalance.** At timestep level the positive class is 24%. At window level (window 200, stride 10, window labelled anomalous if any timestep in it is anomalous) the anomalous share rises to 29.7%, i.e. 2,344 of 7,774 validation windows [verified: `notebooks/03_cnn_baseline.ipynb` output; `reports/EXPERIMENTS.md`, Run 2].

**Test-set prevalence** is not documented anywhere in the repository [unknown]. The organizer's `sample_submission.csv` flags 10.01% of timesteps, but it is a random format template, not ground truth [documented: EXPERIMENTS.md; organizer README says it is "a valid random submission"].

### 3.3 Corrections to the known setup

| Claimed setup item | Finding |
|---|---|
| 18 sensors | Correct [verified]. |
| 216 features | This is **not** the CNN input dimension. 216 is the column count of the tabular feature matrix built for the Isolation Forest baseline: 16 selected sensors, plus 5 temporal features per sensor (lag, delta, rolling mean, rolling std, EWMA), plus 120 pairwise products (16 + 80 + 120 = 216) [verified: `notebooks/02_…ipynb` output `(189444, 216)`]. The current `src/core/feature_pipeline.py` adds pairwise differences and rolling correlations, giving 456 columns (16 + 80 + 3×120) [verified by arithmetic from code; "~456" in docstring and LOF report]. The CNN uses the 18 raw sensors. |
| 28 training runs | Correct [verified]. |
| 10 validation runs | Correct [verified]. |
| 10 labelled evaluation runs | **There is no separate labelled evaluation split.** The 10 labelled runs *are* the validation runs (`val/` + `val_labels/`). They serve for tuning and for every local metric. |
| 53 test runs | Correct [verified]; unlabelled locally, scored by the portal. |

### 3.4 Exploratory analysis and sensor selection

Before modelling, the team ran exploratory analyses (April–June 2026) [verified: `notebooks/archive/`, `notebooks/01_sensor_selection.ipynb`, `plots/*.png`, `data/artifacts/`]:

- KDE-based grouping of sensor value distributions into "highly dynamic", "normal", "spike sensitive" and "stable" categories;
- a composite statistical sensor score, computed in `src/core/feature_selection.py`;
- correlation filtering at 0.85, which kept 16 of 18 sensors (`data/processed/selected_sensors.json`; LS701 and LS702 are dropped) [verified].

**Important:** the 16-sensor selection is used only by the tabular detectors (Isolation Forest tabular baseline, LOF). The CNN autoencoder, which is the final model, uses all 18 sensors [verified: `run_cnn.py` uses `df.values` of the raw CSVs; the model input is 18 channels].

---

## 4. Data Preprocessing

This section describes the preprocessing of the final CNN pipeline (`src/pipelines/run_cnn.py`, `src/core/windowing.py`) [verified from code].

1. **Loading per run.** Each CSV is loaded as its own DataFrame (`load_split_per_file`), so run boundaries are preserved. Validation labels are aligned to validation runs by sorted filename.
2. **Standardization.** One `sklearn.preprocessing.StandardScaler` is fitted on the concatenation of all 28 **training** runs, then applied unchanged to train, validation and test. It is a global, per-sensor z-score. It is not fitted per run, and no validation or test statistics enter it.
3. **Missing values.** No handling is needed, because the raw data contain no NaNs [verified]. (`src/core/data_loader.py` has a `handle_missing_values` helper, but the CNN pipeline does not use it.)
4. **Windowing.** Each run is cut separately into sliding windows of $T = 200$ timesteps × 18 sensors (`create_windows_per_file`), so no window spans two runs.
   - Training stride `STEP = 10` gives **18,401 training windows** [verified: run logs].
   - Validation/test stride `EVAL_STEP = 1` gives one window per start position.
   - Window label (used only to select normal windows for the Mahalanobis fit): 1 if any timestep in the window is anomalous.
5. **Tensor layout.** Windows are permuted to `(batch, channels=18, time=200)` for `Conv1d`.
6. **Window-to-timestep mapping.** Each window's score is added to every timestep it covers, and each timestep's score is the mean over all covering windows (`expand_window_scores_to_timesteps`). Any trailing timestep not covered by a window would take the last covered timestep's score. With stride 1 every timestep is covered, so the fallback never triggers. This produces exactly one continuous score per raw timestep, which is the granularity the submission requires.

**Preprocessing used by the other detectors** [verified from code]:

- *Isolation Forest (tabular baseline):* the 16 selected sensors plus lag-1, first difference, rolling mean/std (window 5) and EWMA (span 5) per sensor, plus pairwise products, differences and rolling correlations (window 20), with NaN/inf filled by bfill/ffill/0 within each run (`src/core/feature_pipeline.py`). The committed baseline metrics come from the earlier 216-feature version.
- *Isolation Forest (window branch of the hybrid):* seven statistics per sensor per window (mean, std, min, max, median, 25th/75th percentile), giving 126 features (`src/models/isolation_forest/features.py`).
- *LOF (Round 5):* the same tabular features, additionally standardized with a scaler fitted on train only in the final version (`run_final_hybrid.py`).

---

## 5. Methodology

### 5.1 Baseline

Two baselines exist in the project's history.

**(a) Isolation Forest, tabular features** (`notebooks/02_anomaly_detection_isolation_forest.ipynb`, ported to `src/pipelines/run_isolation_forest.py`) [verified]:

- `IsolationForest(n_estimators=300, contamination=0.24, random_state=42)` fitted on the 216-column training feature matrix.
- Anomaly score = −`decision_function(X)`.
- Threshold = the F1-maximizing point of the validation precision–recall curve.
- Validation result: **F1 = 0.4366, precision 0.296, recall 0.833, ROC-AUC 0.5851**. It flagged 67.5% of validation timesteps (true rate 24%) and 84.6% of test timesteps [verified: `outputs/isolation_forest/metrics/metrics.json`].
- With sklearn's default `predict()` cutoff, the anomalous-class F1 was only 0.30 [verified: notebook cell output].

**(b) Original hybrid (CNN autoencoder + window Isolation Forest), notebook implementation.** A shallow Conv1d autoencoder (one conv block, kernel 3, receptive field about 3 timesteps) was fused with a window Isolation Forest at a fixed 0.85/0.15 weight after min-max normalization. The threshold was the F1-optimal cutoff. The project documentation reports F1 ≈ 0.60 (CNN), 0.40 (IF) and 0.62 (hybrid) for this version, measured with window-level labels [documented: `HYBRID_MODEL_README.md`, `reports/project_summary_report.pdf`]. These figures are **not reproducible from any saved output**: the saved output of `notebooks/05_hybrid_model.ipynb` shows a hybrid F1 of 0.509 [verified]. They also used a coarser, window-level evaluation that the project later judged non-comparable. The report therefore treats them as historical claims only.

### 5.2 CNN-Based Approach (final architecture)

**Architecture** (`src/models/cnn/model.py`, class `ConvAutoencoder1D`) [verified from code; shapes and parameter count recomputed]:

| Stage | Layer | Output shape (channels × time) |
|---|---|---|
| Input | standardized window | 18 × 200 |
| Encoder | DilatedResidualBlock(18→32, d=1) | 32 × 200 |
| | DilatedResidualBlock(32→32, d=2) | 32 × 200 |
| | DilatedResidualBlock(32→32, d=4) | 32 × 200 |
| | DilatedResidualBlock(32→64, d=8) | 64 × 200 |
| Bottleneck | MaxPool1d(2) | 64 × 100 |
| Decoder | ConvTranspose1d(64→32, k=3, stride 2) + BatchNorm + ReLU | 32 × 200 |
| | DilatedResidualBlock(32→32, d=4) | 32 × 200 |
| | DilatedResidualBlock(32→32, d=2) | 32 × 200 |
| Output | Conv1d(32→18, k=3, pad 1) | 18 × 200 |

Each `DilatedResidualBlock` is Conv1d(k=3, dilation d, "same" padding) → BatchNorm → ReLU → Conv1d(k=3, dilation d) → BatchNorm, then a residual addition and a final ReLU. The shortcut is a 1×1 convolution when the channel count changes. The padding is symmetric, not causal, because this is a reconstruction task.

- **Total trainable parameters: 59,858** [verified].
- **Encoder receptive field: 61 timesteps** before the pool. Each kernel-3 convolution with dilation *d* adds 2*d*; there are two per block, so 1 + 4·(1+2+4+8) = 61. The code docstring, presentation and defense guide state "~63"; the LNCS draft states 61, which is correct by this arithmetic.
- **Design rationale** [documented]: a window-size sweep in `notebooks/03_cnn_baseline.ipynb` (5 epochs, shallow CNN, window-level labels) gave F1 0.4086 / 0.4300 / 0.4494 / 0.4695 for windows of 50 / 100 / 150 / 200 [verified: notebook output]. The team read this as evidence that anomalies are slow drifts that need wide temporal context. That reading is an [interpretation]: the sweep used a single short training run per setting.

**Training** (`src/models/cnn/train.py`) [verified from code and run logs]:

- Objective: mean-squared reconstruction error `nn.MSELoss()`; input = target (autoencoder).
- Optimizer: Adam, learning rate 1e-3, batch size 64, shuffled mini-batches, 50 epochs, no weight decay and no learning-rate schedule.
- Data: the 18,401 training windows only. No labels are used for the gradient updates.
- **Checkpoint selection** (`CNN_EVAL_EVERY = 5`): every 5 epochs the current model is evaluated on validation (fit Mahalanobis → dense scoring → best F<sub>0.5</sub> over thresholds). The state with the highest validation F<sub>0.5</sub> is restored at the end.
- In the 15 Sep MPS runs, training loss fell from 0.187 (epoch 1) to 0.0071 (epoch 50). The checkpoint scores were non-monotonic: 0.531, 0.522, 0.528, 0.564, 0.524, 0.508, 0.468, 0.537, 0.579, 0.518 at epochs 5–50, and epoch 45 was selected [verified: `outputs/experiments/2026-09-15_run1_threshold_ab/run.log`, `…run2…/run.log`]. The checkpoint epoch of the 12 Jul model-of-record run is **not recoverable** from the repository [unknown]. The CNN deep-dive report states "epoch 45 (0.601)" for "your last GPU run" [documented].

### 5.3 Anomaly Scoring

**Per-sensor reconstruction error** (`get_reconstruction_errors(per_sensor=True)`). For window $w$ with input $x_w$ and reconstruction $\hat{x}_w$:

$$
e_{w,s} = \frac{1}{T}\sum_{t=1}^{T}\left(x_{w,s,t} - \hat{x}_{w,s,t}\right)^2, \qquad e_w \in \mathbb{R}^{18}.
$$

**Gaussian normal-error model** (`fit_mahalanobis`). Mean $\mu$ and precision matrix $\Sigma^{-1}$ are estimated with `sklearn.covariance.LedoitWolf` shrinkage on the error vectors of **validation windows whose window label is 0**. These windows are taken at stride 10 across all 10 validation runs, 5,430 in the 15 Sep run [verified: code; count from EXPERIMENTS.md].

**Window score** (`mahalanobis_scores`): the **squared** Mahalanobis distance (no square root is taken):

$$
s(w) = (e_w - \mu)^\top \Sigma^{-1} (e_w - \mu).
$$

**Timestep score.** $S_t$ is the mean of $s(w)$ over all stride-1 windows containing timestep $t$ (Section 4, step 6).

**Why this replaced flat MSE** [documented rationale; interpretation]. The earlier score averaged the error over all sensors *and* timesteps into one scalar. A large deviation on a normally precise sensor could then be diluted by ordinary error on noisy sensors. The Mahalanobis form weights each sensor by its own normal error variance and accounts for cross-sensor error covariance.

### 5.4 Threshold Selection

**Procedure** (`src/evaluation/metrics.py::find_best_threshold`) [verified from code]:

1. Compute `precision_recall_curve(y_val, S_val)` over all 79,688 validation timestep scores.
2. At every candidate threshold compute $F_\beta = (1+\beta^2)\,PR / (\beta^2 P + R)$.
3. Select the threshold with maximal $F_\beta$.
4. Predict $\hat{y}_t = \mathbb{1}[S_t > \tau]$, one global threshold for all runs, applied unchanged to test.

**Configuration of record:** $\beta = 0.5$ (`config.THRESHOLD_BETA`), which weights precision more than recall. The model-of-record threshold was **τ = 43.298** (squared-Mahalanobis units) [verified: `outputs/cnn/metrics.json @ HEAD`].

**Why F<sub>0.5</sub> rather than F1** [documented]. Before Round 4, with weakly separating scores, the F1-optimal threshold flagged 65–88% of timesteps against a true rate of 24%, and 39–41 of 53 test runs were predicted entirely anomalous [documented: summary report and post-mortem]. β = 0.5 was adopted in Round 4 together with Mahalanobis scoring.

**Label use.** Validation labels are used (i) to choose the threshold, (ii) to choose the checkpoint, and (iii) to select the normal windows for the Mahalanobis fit. **Test labels are never available and never used**; the organizers do not distribute them. Section 9 discusses the consequences.

---

## 6. Experimental Setup

| Item | Setting | Source |
|---|---|---|
| Window / train stride / eval stride | 200 / 10 / 1 | `src/config.py` [verified] |
| Epochs / batch / LR / optimizer | 50 / 64 / 1e-3 / Adam | `config.py`, `train.py` [verified] |
| Checkpoint evaluation interval | every 5 epochs, criterion validation F<sub>0.5</sub> | `config.py`, `run_cnn.py` [verified] |
| Threshold objective | F<sub>0.5</sub> on validation, global threshold | `config.py` [verified] |
| Covariance estimator | Ledoit–Wolf | `metrics.py` [verified] |
| Random seed | `torch.manual_seed(42)`, `np.random.seed(42)` | `run_cnn.py` [verified] |
| Determinism flags | none (no `torch.use_deterministic_algorithms`, no cuDNN flags) | code [verified] |
| Metrics | precision, recall, F1 (sklearn, positive class 1), ROC-AUC on continuous scores, predicted anomaly rate | `compute_metrics` [verified] |
| Hardware, 12 Jul model of record | RPTU "Elwetritsch" cluster, V100 GPU via SLURM (`scripts/run_cnn.slurm`) | [documented]; cluster logs were never committed, so the device is not provable |
| Hardware, 15 Sep experiments | Apple M1 GPU (MPS) | run logs "Device: mps" [verified] |
| Software | torch 2.12.1, scikit-learn 1.9.0, numpy 2.4.6, pandas 3.0.3 | `requirements.txt` [verified] |

**Metric granularity.** All local metrics from Round 1 onward are point-wise over the 79,688 validation timesteps, pooled across the 10 runs. Pre-Round-1 notebook metrics used window-level labels and are not comparable.

**Run-to-run variance.** Identical configurations of the Round-4 method gave validation F1 values of 0.568, 0.581, 0.592, 0.602 and 0.611 across separate training runs [verified: 0.581 `outputs/hybrid/metrics.json`, 0.611 commit `610e874`, 0.602 HEAD; documented: 0.568/0.592 LOF report]. The project therefore treats differences below about ±0.02 F1 as noise. The fixed seed did not make runs reproducible across machines; see Section 9.5.

---

## 7. Experiments and Results

The chronology below groups related code changes into scientific stages. Dates come from git commit history [verified]. Unless stated otherwise, all numbers are **validation** metrics.

### 7.1 Baseline

| Stage (date) | Method | Val F1 | Val ROC-AUC | Notes |
|---|---|---|---|---|
| EDA & sensor selection (Apr–Jun 2026) | KDE / statistical scoring / correlation filter → 16 sensors | — | — | [verified: notebooks, plots, `selected_sensors.json`] |
| IF tabular baseline (Jun 2026) | IsolationForest on 216 features, F1-optimal threshold | 0.4366 | 0.5851 | 67.5% val / 84.6% test flagged [verified] |
| CNN notebooks (Jun–Jul 2026) | shallow Conv1d AE, flat MSE, window-level labels | 0.4086–0.4695 (window sweep, 5 ep.) | — | not comparable to later per-timestep metrics [verified: notebook output] |
| Original hybrid (≤ 5 Jul 2026) | shallow CNN + window IF, fixed 0.85/0.15 | "~0.62" (window-level) | — | [documented only; the saved notebook output shows 0.509] |

`notebooks/04_cnn_tuning.ipynb` also holds exploratory outputs: a stride sweep (F1 0.4796–0.5264 for strides 2–12) and an early dilated variant (F1 0.535). These are 5–10-epoch runs with window-level labels. The pipeline did not adopt them directly (the train stride stayed at 10) [verified: notebook output].

### 7.2 Normalization Experiments

Two kinds of normalization appear in the project. They are easily confused.

**(a) Feature standardization (retained).** A global StandardScaler fitted on training data is used in every pipeline (Section 4). It was never the subject of a controlled experiment [verified: no such ablation exists].

**(b) Per-run median/MAD score normalization (tested, rejected; 15 Sep 2026).** The motivation came from the diagnostic run (Section 7.4): raw Mahalanobis magnitudes differ by about 100× across runs, and two validation runs (5 and 7) scored F1 = 0 because their peaks sat below the global threshold [verified: `plots/diagnostics/diagnostic_summary.json`; figure `val_score_timelines.png`]. The proposed fix rescales each run's timestep scores:

$$
\tilde{S}_t = \frac{S_t - \operatorname{median}(S_{\text{run}})}{1.4826\cdot \operatorname{MAD}(S_{\text{run}}) + 10^{-9}}
$$

It is unsupervised and applied identically to test runs (`robust_normalize_per_run`, uncommitted change to `run_cnn.py`) [verified from code]. All variants below share **one trained network** (MPS run, epoch-45 checkpoint), so differences reflect the variant and not training noise [verified: `outputs/experiments/2026-09-15_run1_threshold_ab/metrics.json`]:

| Variant | F1 | Precision | Recall | ROC-AUC | Val flagged | Test flagged |
|---|---|---|---|---|---|---|
| raw scores + F<sub>0.5</sub> threshold (reference) | 0.5829 | 0.577 | 0.589 | 0.7689 | 0.244 | 0.617 |
| per-run norm + F<sub>0.5</sub> | 0.4050 | 0.690 | 0.287 | 0.6769 | 0.099 | 0.102 |
| per-run norm + F1 | 0.4585 | 0.380 | 0.577 | 0.6769 | 0.363 | 0.367 |

**Result:** ROC-AUC fell from 0.769 to 0.677. The rescaling damaged the *ranking* itself, so no threshold choice recovers it. Per-run normalization was **rejected** [verified]. It has never been evaluated under leave-one-run-out calibration [documented: EXPERIMENTS.md].

### 7.3 Mahalanobis Experiments

**Round 4, introduction of Mahalanobis scoring (10 Jul 2026; commits `55b3c2c`, `e0e4f95`).** Mahalanobis scoring (fitted on label-confirmed normal validation windows) replaced flat MSE on the Round-3 dilated CNN. **In the same round** the threshold objective changed from F1 to F<sub>0.5</sub>, and pure-CNN weight 1.0 was added to the fusion grid:

| Metric (CNN) | Round 3 (flat MSE, F1 threshold) | Round 4 (Mahalanobis, F<sub>0.5</sub> threshold) |
|---|---|---|
| F1 | 0.484 | 0.581 |
| Precision / recall | 0.347 / 0.796 | 0.601 / 0.562 |
| ROC-AUC | 0.605 | 0.768 |
| Predicted anomaly rate (true 0.24) | 0.894 | 0.224 |

Round 3 values are [documented: textbook and summary report]; Round 4 values are [verified: `outputs/hybrid/metrics.json` → `cnn_only`, committed in `cb490f9`].

*Attribution.* ROC-AUC depends only on the scores, not on the threshold, so the AUC rise from 0.605 to 0.768 is attributable to the scoring change. The attribution is not perfectly clean, because the checkpoint-selection criterion also changed from F1 to F<sub>0.5</sub> and may have picked a different epoch. The F1 rise from 0.484 to 0.581 combines the scoring change and the threshold change, and **their separate contributions were never isolated** [interpretation].

**15 Sep 2026, source of the normal model (A vs B vs C, plus leave-one-run-out).** Reconstruction errors were computed once from one network, and several normal models were applied to the same cached errors [verified: `outputs/cnn/metrics.json` (working tree), `outputs/experiments/2026-09-15_run2_mahalanobis_source/`]:

| Normal model | Fitted on | F1 (F<sub>0.5</sub> thr.) | F1 (F1 thr.) | ROC-AUC | Test flagged (F<sub>0.5</sub>) |
|---|---|---|---|---|---|
| **A** (pipeline default) | label-confirmed normal val windows (5,430) | 0.5829 | 0.6067 | 0.7689 | 0.617 |
| **B** | all 18,401 train windows | 0.4829 | 0.4829 | 0.6109 | 0.833 |
| **C** | all val windows, unfiltered (~30% anomalous) | 0.5261 | — | 0.6929 | — |
| **A under leave-one-run-out** | normal windows of the other 9 val runs | 0.5251 | — | 0.6985 | — |

A and B values are [verified: metrics JSON]. C and LORO values are [documented: EXPERIMENTS.md]. The verification scripts were read-only and are not saved in the repository.

Supporting measurements [documented: EXPERIMENTS.md, from the saved `outputs/cnn/model.pt`]:

- Per-window mean error: train 0.00668, val-normal 0.02139, val-anomalous 0.04609.
- Mean per-sensor error variance: train 0.000144, val-normal 0.001727.
- Median per-sensor error ratio val-normal/train = 2.693, with 17 of 18 sensors larger.
- Median Mahalanobis score: 83.39 under B vs 7.30 under A.
- LORO lowers the AUC of every one of the 10 validation runs (per-run values in Appendix A.3).

**Findings:**

1. Fitting on training windows (B) is clearly worse than fitting on validation-normal windows (A): ROC-AUC 0.611 vs 0.769 [verified]. B was **rejected**.
2. A vs B is **not a one-variable comparison**. The two fits differ in three ways at once: in-sample vs held-out errors, contaminated vs label-filtered windows, and different-run vs same-run calibration. Model C (held-out but unfiltered) still beats B, so contamination alone does not explain B's deficit [documented]. That B's failure is driven mainly by training-set errors being much tighter than held-out errors is a plausible [interpretation], not an isolated result.
3. **The pipeline's default validation estimate is optimistic.** Model A is calibrated on normal windows from the same runs it then scores, and test runs can never supply such windows. Leave-one-run-out calibration lowers ROC-AUC by 0.070 (0.7689 → 0.6985) and F1 by 0.058 (0.5829 → 0.5251) [documented]. That same-run calibration is the specific *cause* is an interpretation; the drop itself was measured.

### 7.4 Threshold Experiments

**(a) Diagnostic run (12–13 Jul 2026; `scripts/diagnose_cnn.py`, commit `c9b23c7`).** A separate training run of the Round-4 pipeline was scored and analysed [verified: `plots/diagnostics/diagnostic_summary.json`]:

| Threshold objective | F1 | Precision | Recall |
|---|---|---|---|
| F<sub>0.5</sub>-optimal | **0.5731** | 0.5975 | 0.5506 |
| F1-optimal | **0.6181** | — | — |

The diagnostic run also recorded 8,578 false negatives and 7,079 false positives at the F<sub>0.5</sub> threshold. Per-run F1 ranged from 0.000 (runs 5 and 7) to 0.789 (run 8). The top signal sensors by normal-vs-anomalous error were FT703, T703, FT704, PDI702 and FYI702 [verified]. The diagnostic run is a **different trained network** from the model of record (0.6022); its numbers are not the model-of-record metrics.

**(b) Controlled F1-vs-F<sub>0.5</sub> comparison (15 Sep 2026, Run 1, one shared network)** [verified: `…run1_threshold_ab/metrics.json`]:

| Threshold objective | F1 | Precision | Recall | ROC-AUC | τ | Val flagged | Test flagged |
|---|---|---|---|---|---|---|---|
| F<sub>0.5</sub> (method of record) | 0.5829 | 0.577 | 0.589 | 0.7689 | 42.77 | 0.244 | 0.617 |
| F1 | **0.6067** | 0.505 | 0.759 | 0.7689 | 15.11 | 0.360 | **0.842** |

Findings:

- Within one network, the F1 threshold gains +0.024 validation F1, not the +0.045 the diagnostic run implied (0.5731 → 0.6181). The gain is inside the documented ±0.02 run-to-run band [verified arithmetic].
- The same absolute threshold flags a very different fraction of test than of validation (84.2% vs 36.0%) [verified]. **This validation–test score shift is a new finding.** The F1-threshold submission was therefore **not promoted**.
- A "rate-matched" candidate, whose test threshold is chosen by quantile to reproduce the validation flagging rate, was written to `outputs/candidates/`. By construction it cannot be evaluated locally, and it was not uploaded [documented: EXPERIMENTS.md].

**(c) Isolation Forest contamination sweep (Round 2).** Contamination values 0.1–0.3 all gave identical F1 (0.4907). The pipeline thresholds the continuous `decision_function` itself, so contamination (which only moves sklearn's internal cutoff) has no effect [documented: summary report; consistent with the code].

### 7.5 CNN Improvement

The CNN branch improved across Rounds 1–4. Rounds 1–3 ran inside the hybrid pipeline (`run_hybrid.py`); Round 4 onward also ran standalone.

| Round (date, commit) | Change | CNN val F1 | CNN val ROC-AUC | Hybrid val F1 | Evidence |
|---|---|---|---|---|---|
| Round 1 (5 Jul, `2e2c8f8`) | Refactor notebooks → `src/`; fix 6 bugs: IF feature flattening, threshold off-by-one, cross-run windows, epochs 10→50, window-level → dense per-timestep scoring, hard-coded fusion weight → validation sweep | 0.408 | 0.574 | 0.494 | [documented: summary report, post-mortem] |
| Round 2 (5 Jul, `2e2c8f8`) | Validation checkpoint selection every 5 epochs; IF contamination sweep; finer fusion grid | 0.458 | 0.614 | 0.494 (weight CNN 0.4) | [verified: `outputs/hybrid/metrics.json @ 2e2c8f8`] |
| Round 3 (10 Jul, `55b3c2c`) | Dilated residual (TCN-style) architecture | 0.484 | 0.605 | 0.494 | [documented: summary report, textbook] |
| Round 4 (10 Jul, `55b3c2c`/`e0e4f95`) | Per-sensor Mahalanobis + F<sub>0.5</sub> threshold; grid includes pure CNN | 0.581 | 0.768 | 0.581 (weight CNN 1.0) | [verified: `outputs/hybrid/metrics.json @ cb490f9`] |
| Round 4 re-run, standalone (12 Jul, `610e874`) | Same method, `run_cnn.py` | 0.611 | 0.758 | — | [verified: `outputs/cnn/metrics.json @ 610e874`] |
| **Model of record** (12 Jul, `b456f6d`) | Same method, `run_cnn.py`, portal submission written | **0.6022** | **0.7629** | — | [verified: `outputs/cnn/metrics.json @ HEAD`] |

**Round 1** also produced a committed standalone `run_cnn.py` result of F1 0.4406 / ROC-AUC 0.5958, flagging 71.3% of validation [verified: `outputs/cnn/metrics.json @ 2e2c8f8`]. It differs from the 0.408 in-hybrid figure because it comes from a different run and pipeline.

**Round 2 observations** [documented]: validation F1 peaked at epoch 5 and declined thereafter, while training loss kept falling. The team interpreted this as the autoencoder learning to reconstruct anomalies present in the unlabelled training data [interpretation; see Section 9.6].

**Round 3 observations** [documented]: CNN F1 rose, but CNN ROC-AUC did not (0.614 → 0.605). The predicted anomaly rate worsened (0.656 → 0.894), and test runs predicted fully anomalous rose from 39 to 41 of 53. Round 3 improved the F1-optimal operating point, not the ranking.

**Round 5 fusion with LOF (11–12 Jul; commits `56ca1ea` on branch `hybrid-lof-model`, then `ac2042a`, `a5326db`).** A teammate's density-based LOF detector (`n_neighbors=200`, `novelty=True`) was fused with the Round-4 CNN as $w \cdot \text{CNN} + (1-w)\cdot\text{LOF}$ after min-max normalization with validation statistics. The weight $w$ was chosen by **run-grouped 5-fold CV** (`GroupKFold` by validation run), and the deployed weight was the mean of the per-fold choices. Results from the cluster runs [documented: `reports/lof_experiment_report.pdf`]:

| Experiment | CNN only F1 / AUC | LOF only F1 / AUC | Hybrid out-of-fold F1 / AUC | Per-fold CNN weight | Deployed LOF weight |
|---|---|---|---|---|---|
| A: LOF on raw features | 0.592 / 0.783 | 0.461 / 0.687 | 0.549 / 0.716 | 1.00, 1.00, 1.00, 0.85, 1.00 | 0.03 |
| B: LOF on standardized features | 0.568 / 0.765 | 0.481 / 0.667 | 0.568 / 0.765 | 1.00 ×5 | 0.00 |

Nine of ten folds chose pure CNN, and the out-of-fold hybrid never exceeded CNN alone. Fusion was **rejected**, and the CNN alone became the model of record (commit `b456f6d`).

> **Discrepancy.** The untracked local file `outputs/final_hybrid/metrics.json` (12 Jul 14:36) records a *different* Round-5 run: CNN-only F1 0.479 (AUC 0.691), LOF-only 0.461, OOF hybrid 0.403, per-fold weights [1.0, 1.0, 1.0, 0.7, 1.0], deployed LOF weight 0.06. It matches neither cluster run in the LOF report. The LOF report explicitly calls its cluster runs authoritative. The local file appears to be an earlier or local run and is treated here as non-authoritative. Its conclusion (fusion does not help) is the same.

### 7.6 Final Result

**Portal submissions with a recorded score.** Scores are [documented] in the textbook, summary report and README; flag rates were [verified] recomputed from the prediction file committed alongside each reported score:

| Date | Submission | Val F1 of that run | Test timesteps flagged | Portal F1 |
|---|---|---|---|---|
| ~5 Jul 2026 | Round-1 hybrid (`outputs/hybrid_predictions.csv`) | 0.494 | 0.885 (file at `2e2c8f8`) | public 0.399 / private 0.471 |
| ~10 Jul 2026 | Round-4 hybrid, CNN weight 1.0 (`outputs/hybrid_predictions.csv`) | 0.581 | 0.560 (file at `cb490f9`) | 0.59 |
| ~12 Jul 2026 | **CNN only, model of record (`outputs/cnn_predictions.csv`)** | **0.6022** | **0.622** | **0.63** |

Caveat on the flag rates: the repository does not record which exact file was uploaded for each score. The committed file is the one the documentation associates with the score. For the Round-1 file, the committed version at `2e2c8f8` was generated after the Round-2 changes (its companion metrics show CNN 0.458), so it may not be byte-identical to the file that scored 0.399/0.471 [unknown].

**Current best / final verified result: portal F1 = 0.63**, produced by `outputs/cnn_predictions.csv`. The file is unchanged since commit `b456f6d` [verified: `git diff HEAD` is empty for this file]. It has 316,024 rows across 53 runs and flags 62.2% of test timesteps [verified]. Whether 0.63 is the public-split score, the private-split score or a combined score is not recorded [unknown].

**The "0.58 → 0.63" change.** The Round-4 method first reached validation F1 0.581 (portal 0.59). A later standalone re-run of the **same method** reached validation 0.602 and portal 0.63. No method change separates the 0.59 and 0.63 submissions:

- the fused score at weight 1.0 is a monotone transform of the CNN score, so the Round-4 hybrid and standalone pipelines are equivalent in method;
- the difference is a different training run (different weights, probably a different device) and the resulting different threshold;
- the validation F1 difference (0.581 vs 0.602) is inside the documented ±0.02 run-to-run band.

**It should not be presented as a method improvement.** The LNCS draft reaches the same conclusion in a source comment [verified: `reports/paper_lncs/main.tex`, lines 38–41].

---

## 8. Failed Approaches

| Approach | Why it seemed reasonable | Result | Why rejected | Lesson |
|---|---|---|---|---|
| **Isolation Forest as a fusion partner** (Rounds 1–4) | Different inductive bias (global isolation, no temporal model); hoped to cover CNN blind spots | Hybrid F1 flat at 0.494 through Rounds 1–3. Once the CNN had Mahalanobis scoring, the sweep chose CNN weight 1.0 (F1 0.581) [verified] | Added nothing once the CNN ranking was stronger (IF AUC 0.658 vs CNN 0.768) | A weaker score blended into a stronger one did not help here; the hybrid's plateau was set by IF, not the CNN [interpretation] |
| **LOF fusion**, raw and standardized features (Round 5) | Density-based detector with cross-sensor features; beat IF standalone on F1 | OOF hybrid 0.549 (raw) and 0.568 (std.) vs CNN-only 0.592 / 0.568; 9 of 10 folds chose pure CNN [documented] | Leakage-free CV assigned LOF ≈ 0 weight | Run-grouped CV is a usable tool for honest fusion decisions |
| **IF contamination tuning** (Round 2) | Hand-set hyperparameter never tuned | Identical F1 0.4907 for all values [documented] | No effect by construction | Know which hyperparameters the scoring path actually uses |
| **Per-run median/MAD normalization** (15 Sep) | Two val runs had F1 = 0 because their peaks were below the global threshold | ROC-AUC 0.769 → 0.677; F1 0.583 → 0.405 [verified] | Damages ranking, not just calibration | Cross-run magnitude carries real signal; the diagnostic's "fix" did not survive a controlled test |
| **Mahalanobis fitted on train windows** (15 Sep) | 18k windows give a better-conditioned covariance, and it removes validation labels from the score | ROC-AUC 0.611, F1 0.483 [verified] | Far worse than the validation-normal fit | In-sample training errors are much tighter than held-out errors (variance 0.000144 vs 0.001727) [documented] |
| **F1-optimal threshold** (15 Sep) | The portal grades F1; diagnostic suggested +0.045 | +0.024 val F1 within one network, but 84.2% of test flagged [verified] | Gain within noise; extreme test flagging rate | Validation and test score distributions do not match under a fixed absolute threshold |
| **Round 3 dilated architecture** (partial success) | Tuning suggested slow drifts need wide context | CNN F1 0.458 → 0.484, but AUC 0.614 → 0.605 and over-flagging worsened [documented] | Kept (it is the final architecture), but on its own it did not improve ranking | F1 at an F1-optimal threshold can rise without better separation |

---

## 9. Evaluation and Reliability

### 9.1 Split separation and label use

| Data | Used for | Labels used? |
|---|---|---|
| Train (28 runs) | StandardScaler fit; autoencoder training; IF/LOF fit; LOF feature scaler | none exist |
| Validation (10 runs) | checkpoint selection; Mahalanobis μ, Σ fit (normal windows only); threshold selection; fusion-weight selection; min-max score normalization stats; **all reported local metrics** | **yes** |
| Test (53 runs) | inference only; portal submission | never available |

- **No test-label leakage is possible** from the pipeline, because the organizers never distributed test labels [verified: the manifest's test runs have no `label_path`; the README describes test as unlabelled].
- **No test-statistics leakage** in the final pipeline: the scaler is fitted on train, and Mahalanobis and threshold on validation [verified from code]. (The rejected per-run normalization used each test run's own score statistics, which is transductive but label-free.)
- **Validation metrics are in-sample.** The same 10 runs and labels select the checkpoint, fit μ and Σ, choose the threshold, and produce the reported F1/AUC. Window size, architecture and β were also chosen by looking at results on these runs. Reported validation F1 is therefore **not** an unbiased generalization estimate. Leave-one-run-out calibration alone costs 0.070 AUC and 0.058 F1 (Section 7.3). Checkpoint and threshold selection were not moved out of sample, so the full optimism is larger than or equal to that figure [interpretation].
- **The model depends on validation labels beyond the threshold.** The presentation (slide 3) and the LNCS draft say labels are used only for checkpoint and threshold. This is incorrect: they also select the "confirmed-normal" windows used to fit the Mahalanobis model [verified: `fit_mahalanobis_from_validation` uses `window_labels == 0`].

### 9.2 Local validation vs portal score

| | Local validation (model of record) | Portal |
|---|---|---|
| Data | 10 labelled val runs, 79,688 timesteps | hidden labels of 53 test runs (public/private split unknown) |
| Metric | point-wise F1 over pooled timesteps (sklearn) | "binary F1-score" (organizer README); exact aggregation not stated |
| Threshold | tuned on these same labels | applied blind |
| Value | F1 0.6022, ROC-AUC 0.7629 | F1 0.63 |

The two numbers **cannot be compared directly**. Earlier project documents (README, defense guide, presentation slide 7) argue that "portal 0.63 > validation 0.60, so validation is if anything pessimistic". The evidence does not support that conclusion (Section 9.4).

### 9.3 Validation–test score shift

At the model-of-record threshold, 24.1% of validation timesteps but 62.2% of test timesteps are flagged [verified: metrics JSON and submission file]. The 15 Sep network shows the same pattern: 24.4% vs 61.7% at the F<sub>0.5</sub> threshold, and 36.0% vs 84.2% at the F1 threshold [verified]. Test scores are systematically higher than validation scores. The cause is **not established** [unknown]. Possible explanations include higher test prevalence, different operating conditions in test runs, and the validation-calibrated Mahalanobis model not transferring (consistent with the LORO result).

### 9.4 What the portal score means

The organizer README states that the portal uses **binary F1** [verified]. For a file flagging a fraction $q$ of test timesteps, on a test set with prevalence $p$, at recall $r$, **point-wise** F1 equals $2rp/(q+p)$. With $q = 0.622$:

| Assumed test prevalence p | Point-wise F1 at r = 1.0 | at r = 0.76 |
|---|---|---|
| 0.2395 (validation value) | 0.556 | 0.423 |
| 0.30 | 0.651 | 0.495 |
| 0.40 | 0.783 | 0.595 |

These values were recomputed [verified arithmetic; the table also appears in EXPERIMENTS.md]. If the portal computes point-wise F1 over all 53 runs, a score of 0.63 requires test prevalence ≥ 0.286 even at perfect recall. Three possibilities remain open:

1. test prevalence is substantially higher than validation's;
2. the portal aggregates F1 differently (per run, event-wise or point-adjusted);
3. 0.63 is a public- or private-split score on a subset of runs whose prevalence differs from the whole.

The repository contains nothing that decides between them [unknown].

**Correction to EXPERIMENTS.md and the LNCS draft.** Both state that the portal metric is "not documented anywhere in the repository". The organizer `readme.pdf` in commit `42901df` does state "binary F1-score". It is an image-only PDF, which may be why a text search missed it. What remains undocumented is the aggregation (point-wise vs other), the public/private composition, and the test prevalence.

### 9.5 Reproducibility

- **Seeds** are fixed (42) for torch and numpy, but no deterministic-algorithm flags are set. Training was run on different devices (V100/CUDA [documented] and M1/MPS [verified]).
- The 12 Jul (0.6022) and 15 Sep (0.5829 at matched β = 0.5) runs used identical code, but their test scores have Spearman rank correlation 0.947, and 90.9% of test timesteps shift rank by more than 1,000 positions [documented: EXPERIMENTS.md]. **The fixed seed did not reproduce the model across these runs.**
- **The model-of-record checkpoint (portal 0.63) is not saved.** `outputs/cnn/model.pt` was first written on 15 Sep and belongs to the MPS re-run. The 0.63 result can be re-submitted from the saved CSV but **cannot be regenerated bit-exactly** [verified: file dates, EXPERIMENTS.md].
- **Cluster logs** (`logs/*.out`) were never committed, so the device and checkpoint epoch of the 12 Jul run cannot be proven [verified: no such files in git].
- The **working tree contains uncommitted changes** (`run_cnn.py`, `train.py`, `outputs/cnn/metrics.json`, `outputs/cnn/predictions.csv`). The uncommitted `run_cnn.py` no longer writes `outputs/cnn_predictions.csv`; it writes candidate files to `outputs/candidates/` instead. Running the current working-tree pipeline does **not** reproduce the model-of-record submission path; the committed `run_cnn.py @ HEAD` does [verified: `git diff`].
- The file `outputs/experiments/2026-09-15_run1_threshold_ab/metrics.json` contains the key `"deployed_cnn_predictions": "raw_beta1.0"`. EXPERIMENTS.md states that this variant was **not** promoted and that the 0.63 file was restored. The restoration is confirmed: `outputs/cnn_predictions.csv` is identical to HEAD [verified]. The JSON key is stale.

### 9.6 Other methodological limitations

- **Pooled metric.** Validation F1 pools all timesteps. Runs 8 (81% anomalous) and 9 (63%) dominate the positives, while runs 4 and 5 (< 3%) barely affect it.
- **Training-set contamination** is a hypothesis used throughout the project reports to explain early checkpoint peaks and the train-fit result. The organizers describe the training runs as "normal runs" [verified: README]. Contamination has not been demonstrated [unknown].
- **Threshold inequality (minor).** `precision_recall_curve` counts scores ≥ τ as positive, but `compute_metrics` predicts with > τ. The applied threshold therefore excludes timesteps whose score equals τ exactly. With continuous scores this affects at most a handful of timesteps [verified from code; impact not measured].
- **Config comment mismatch (minor).** `src/config.py` cites a notebook-04 epoch experiment ("F1 0.4099 → 0.4699 going from 15 to 50 epochs"). The saved notebook outputs show those approximate numbers for a *window-size* sweep (0.4086 → 0.4695), not an epoch sweep [verified]. The epoch justification for 50 epochs is therefore not supported by a saved output.

---

## 10. Discussion

**Scoring mattered more than capacity** [demonstrated for AUC; interpretation for the general claim]. Across the CNN rounds, validation ROC-AUC barely moved with checkpoint selection (0.574 → 0.614) and not at all with the dilated architecture (0.614 → 0.605). It rose sharply, 0.605 → 0.768, when per-sensor Mahalanobis scoring replaced flat MSE. ROC-AUC is threshold-independent, so this is a real improvement in separating normal from anomalous timesteps. The diagnostic plots support the mechanism: the error signal concentrates in a few flow, pressure and temperature sensors (FT703, T703, FT704, PDI702, FYI702), which a flat average dilutes [verified: diagnostic summary; interpretation of mechanism].

**F1 can move without ranking moving, and vice versa** [demonstrated]. Round 3 raised F1 while AUC fell slightly. The 15 Sep threshold comparison changed F1 by 0.024 at identical AUC. Reports of F1 alone would have mis-ranked these changes, so AUC was a useful companion metric.

**Plausible fixes can fail controlled tests** [demonstrated]. Per-run normalization was motivated by a clear diagnostic observation: two runs at F1 = 0 with visibly correct peaks. Tested on a shared network, it lowered AUC by 0.09. The diagnostic was correct that a global threshold misses those runs. The proposed remedy removed cross-run information that the ranking relies on [interpretation].

**Fusion did not help** [demonstrated for IF, and for LOF under this protocol]. Every fusion attempt converged on CNN weight ≈ 1.0. The tabular detectors' ROC-AUC (0.66–0.69) was lower than the CNN's (≈ 0.76–0.78). Blending in a weaker, differently distributed score lowered precision more than it raised recall under F<sub>0.5</sub> [documented results; interpretation of mechanism].

**The validation protocol is optimistic, and the portal comparison is uninformative** [demonstrated for LORO; unknown for the portal]. The leave-one-run-out result is the most consequential finding of the late-stage work. The Mahalanobis model borrows normal windows from the runs it scores, and that advantage disappears on test. The portal score, in turn, cannot confirm or refute generalization until its aggregation and test prevalence are known.

**Validation and test score scales differ** [demonstrated]. A fixed absolute threshold flags about 2.5× more of test than of validation. This affects every threshold decision for the portal. It explains why the F1-optimal threshold, better on validation, was not submitted.

**Run-to-run variance is comparable to most effects studied** [demonstrated]. Validation F1 of the same method ranged 0.568–0.611. Apart from the Round-4 scoring change and the rejected variants (which lost 0.10–0.18 F1), most candidate improvements were smaller than this band.

---

## 11. Limitations

1. **Small labelled set.** Only 10 labelled runs, containing 28 anomalous segments, served for tuning and for every local metric. No held-out labelled data exists.
2. **In-sample validation estimate.** Checkpoint, Mahalanobis fit, threshold, β, window size and architecture were all chosen on the same 10 runs. LORO shows at least 0.058 F1 of optimism from the Mahalanobis fit alone.
3. **Unknown portal aggregation, test prevalence and public/private composition.** These prevent interpreting 0.63 as point-wise F1 and prevent principled threshold tuning for the portal.
4. **Single global threshold under a validation–test score shift** (24% vs 62% flagged).
5. **Poor reproducibility across hardware.** The model-of-record checkpoint and cluster logs are not saved.
6. **Run-to-run variance of about ±0.02 F1.** Most results rest on a single training run per configuration; no seed averaging was performed.
7. **Confounded changes.** Round 4 changed scoring, threshold objective and checkpoint criterion together; the A-vs-B normal-model comparison varies three factors at once.
8. **Limited architecture exploration.** LSTM and Transformer alternatives were considered but not implemented [documented: post-mortem].

---

## 12. Conclusion

The project detected per-timestep anomalies in 18-sensor batch-distillation runs without labelled training data. The final system is a dilated residual convolutional autoencoder with per-sensor squared-Mahalanobis scoring of its reconstruction error, dense per-timestep score averaging, and a validation-tuned F<sub>0.5</sub> threshold. Its submission reached **F1 = 0.63 on the competition portal**, the best verified external result of the project (validation F1 0.6022, ROC-AUC 0.7629 for the same run).

The development history shows that correcting the evaluation pipeline and changing *how the model's output is scored* mattered more than increasing model capacity. Six correctness bugs had to be fixed before any number was meaningful. Checkpoint selection and a dilated architecture gave modest standalone gains. Per-sensor Mahalanobis scoring produced the largest change in ranking quality (ROC-AUC 0.605 → 0.768). Fusion with Isolation Forest and LOF was rejected under a leakage-free, run-grouped protocol. Three later ideas (F1-optimal thresholding, per-run score normalization, train-fitted Mahalanobis) were tested on a shared network and rejected.

The main methodological lessons:

- validation metrics are optimistic when a calibration model shares runs with the evaluation (−0.070 AUC, −0.058 F1 under LORO);
- a fixed seed did not guarantee a reproducible model across hardware;
- claims based on a portal score need the portal's exact metric definition before they can support conclusions about generalization.

No claim of superiority over other methods is made: no external baselines beyond the project's own Isolation Forest and LOF were evaluated.

---

## 13. Future Work

Ordered by the strength of the evidence behind each item:

1. **Adopt run-grouped (leave-one-run-out) validation for every claim**, including checkpoint and threshold selection. The `GroupKFold` machinery exists in `run_final_hybrid.py`, and `train_autoencoder(collect_states=True)` (uncommitted) was added to support nested checkpoint selection. *Motivated by the measured LORO optimism.*
2. **Establish the portal's aggregation, the test prevalence, and whether 0.63 is a public or private score.** Ask the organizers or read the Codabench competition page. *Motivated by the arithmetic bound in Section 9.4; required before any further threshold tuning.*
3. **Address the validation–test score shift**, e.g. by calibrating the normal model on data that transfers to test runs. *Motivated by the measured 24% vs 62% flagging gap; no candidate fix has been tested.*
4. **Seed ensembling** (average scores over several training seeds) to reduce the ±0.02 run-to-run variance. *Proposed in `improvement_backlog.pdf`; not tested.*
5. **Contamination-robust training** (down-weighting high-error training windows). *Proposed repeatedly in project reports; rests on the unverified contamination hypothesis; not tested.*
6. **Save every submitted checkpoint, its run log and the exact uploaded CSV**, so each portal score can be traced to reproducible artifacts.

---

## 14. Presentation Talking Points

**Problem.** Flag anomalous timesteps in 18-sensor, 1 Hz batch-distillation runs. Training runs are unlabelled, so we learn normal behaviour and score deviation (novelty detection). The portal grades binary F1.

**Dataset.**
- 28 unlabelled train runs (189,444 timesteps).
- 10 labelled validation runs (79,688 timesteps, 24% anomalous, per-run range 0.8–81%).
- 53 unlabelled test runs (316,024 timesteps), scored on Codabench.
- No separate labelled evaluation split: the 10 labelled runs are the validation set.
- "216 features" refers only to the Isolation Forest tabular baseline, not the CNN.

**Initial approach.** A hybrid of a shallow CNN autoencoder and an Isolation Forest, fused 0.85/0.15. The notebook code had six correctness bugs (cross-run windows, threshold off-by-one, flattened IF features, window-level scoring, reverted epoch count, hand-set fusion weight).

**Baseline.**
- Isolation Forest (tabular): val F1 0.437, AUC 0.585.
- First per-timestep hybrid after bug fixes: val F1 0.494. Portal: public 0.399 / private 0.471.

**Important experiments.**
- Checkpoint selection: CNN F1 0.408 → 0.458.
- Dilated TCN architecture: CNN F1 0.458 → 0.484; AUC did not improve.
- **Per-sensor Mahalanobis scoring + F0.5 threshold: CNN F1 0.484 → 0.581, AUC 0.605 → 0.768.** This was the key improvement. The AUC gain belongs to the scoring change; the F1 gain mixes scoring and threshold.
- Leave-one-run-out: validation is optimistic by 0.070 AUC / 0.058 F1.

**Failed experiments (be open about them).**
- IF fusion: weight → 0.
- LOF fusion: 9 of 10 CV folds chose pure CNN.
- Per-run median/MAD normalization: AUC 0.769 → 0.677.
- Train-fitted Mahalanobis: AUC 0.611.
- F1-optimal threshold: +0.024 val F1 (within noise), but flags 84% of test.

**Final approach.** StandardScaler (train) → per-run 200×18 windows → dilated residual CNN autoencoder (59,858 params, 50 epochs, best-F0.5 checkpoint) → per-sensor reconstruction error → squared Mahalanobis (Ledoit–Wolf, fitted on normal validation windows) → per-timestep averaging → global F0.5 threshold (τ ≈ 43.3) → 0/1 per timestep.

**Final result.** **Portal F1 = 0.63** (validation F1 0.602, ROC-AUC 0.763, same run).

**How to describe 0.58/0.59 → 0.63.** Same method, different training run. Do not present it as a method improvement.

**Main takeaway.** Fixing evaluation and how the output is scored mattered more than model capacity. Rigorous tests rejected several plausible ideas.

**Limitations.**
- Only 10 labelled runs, used for all tuning.
- Validation estimate is in-sample and optimistic.
- Portal aggregation and test prevalence are unknown.
- Test scores run higher than validation scores.
- ±0.02 run-to-run noise.
- The 0.63 checkpoint was not saved.

**Future work.** Run-grouped validation for all selection; confirm the portal metric details; address the val/test score shift; seed ensembles; contamination-robust training.

### 14.1 Claims in `reports/presentation_final.tex` that conflict with the evidence

| Slide | Current claim | Problem | Suggested fix |
|---|---|---|---|
| 10 | "**Fix:** per-run normalization (cheap, unsupervised)" | **Still present.** Tested 15 Sep: AUC 0.769 → 0.677, F1 0.583 → 0.405 [verified] | Reframe as "tested and rejected: it damages ranking" |
| 12 | Roadmap item 2 "Per-run score normalization … rescues the missed runs" | Tested and rejected | Remove, or move to "tested, negative" |
| 12 | Roadmap item 3 "fit Mahalanobis on train" | Tested and rejected (AUC 0.611) | Remove; keep "seed-ensemble" |
| 11 | "portal appears to grade F1 (~0.045 recoverable)" | Organizer README confirms binary F1; the controlled test found +0.024 (within noise) with 84% of test flagged | "Portal grades binary F1 (aggregation unstated); F1-threshold gave +0.024 on val, within noise, and was not adopted" |
| 3 | "Labels touch only the checkpoint and threshold" | Labels also select the normal windows for the Mahalanobis fit [verified from code] | "…checkpoint, threshold, and the normal-error model" |
| 5 | "Changing the scoring … F1 0.48 → 0.60" | Round 4 changed scoring **and** threshold objective together and measured 0.484 → 0.581; 0.60 came from a later re-run | "F1 0.48 → 0.58 and AUC 0.61 → 0.77 (scoring + F0.5 threshold together)" |
| 7, 13 | Round 3 ROC-AUC "0.65"; "ROC-AUC ~0.65 → 0.76" | Project reports give Round-3 **CNN** AUC 0.605 (0.65 matches the IF/hybrid AUC) [documented] | Use 0.605 (≈ 0.61) |
| 7 | "Portal F1 = 0.63 higher than validation: not over-fit" | LORO shows validation is optimistic; portal aggregation, prevalence and public/private split are unknown (Section 9.4) | Drop the inference; state both numbers separately |
| 8 | "CNN only 0.59 / **0.77**" | LOF report gives CNN AUC 0.783 in Exp A (0.765 in Exp B) | Use 0.78, or cite both experiments |
| 4 | "~63-step receptive field" | Arithmetic gives 61 | "61 timesteps" |
| 11 | "Training-set contamination: … likely contains undetected anomalies" | Hypothesis only; organizers describe train as "normal runs" | Label it as an untested hypothesis |
| 11 | "Run-to-run variance 0.568–0.611" | Supported [verified/documented] | Keep |
| 2 | "28 train / 10 validation / 53 test" | Correct [verified] | Keep |

**Related documents with the same issues:**
- `reports/presentation_defense_guide.pdf` and `reports/cnn_deep_dive.pdf` repeat "validation is pessimistic", "~63", and "the portal appears to score F1".
- `README.md` repeats "validation is if anything slightly pessimistic" and recommends Mahalanobis-on-train and threshold matching as future work, both since tested.
- `reports/paper_lncs/main.tex` says the portal metric is "not documented" and that labels are used "for two decisions only".
- `reports/EXPERIMENTS.md` says the portal metric is undocumented; the organizer README states binary F1.

---

## Appendix A: Experiment Table

### A.1 Chronological experiment history

"Val" = validation (10 labelled runs, point-wise, in-sample). "Portal" = external Codabench score.

| # | Date | Experiment | Main change | Val F1 | Other metrics | Status | Reason | Evidence |
|---|---|---|---|---|---|---|---|---|
| B0 | Jun 2026 | IF tabular baseline | IsolationForest, 216 features | 0.4366 | AUC 0.5851; val flagged 0.675, test flagged 0.846 | Baseline | First per-timestep baseline | verified |
| B1 | ≤ 5 Jul | Notebook hybrid | shallow CNN + window IF, 0.85/0.15 | "~0.62" (window-level) | — | Superseded | Buggy, non-comparable metric | documented only |
| R1 | 5 Jul | Refactor + 6 bug fixes | per-run windows, dense scoring, fixed threshold index, IF per-sensor stats, 50 epochs, weight sweep | CNN 0.408 / IF 0.491 / hybrid 0.494 | AUC 0.574 / 0.649 / 0.636; **portal public 0.399 / private 0.471** | Kept | Correctness | documented |
| R2 | 5 Jul | Checkpoint selection | best val checkpoint every 5 epochs | CNN 0.458 / hybrid 0.494 | CNN AUC 0.614; hybrid weight CNN 0.4 | Kept | CNN gain | verified (`@2e2c8f8`) |
| R2b | 5 Jul | IF contamination sweep | 0.1–0.3 | 0.4907 (all) | — | No effect | Contamination unused by scoring path | documented |
| R3 | 10 Jul | Dilated TCN AE | dilations 1-2-4-8, residual | CNN 0.484 / hybrid 0.494 | CNN AUC 0.605; val flagged 0.894 | Kept (final arch.) | F1 gain; AUC flat | documented |
| R4 | 10 Jul | Mahalanobis + F<sub>0.5</sub> | per-sensor error, Ledoit–Wolf, β = 0.5; weight grid incl. 1.0 | CNN = hybrid 0.581 | AUC 0.768; P 0.601, R 0.562; IF 0.489; **portal 0.59** | Kept (key change) | Large AUC gain | verified (`@cb490f9`) + documented (portal) |
| R4r | 12 Jul | Standalone re-run | `run_cnn.py` | 0.611 | AUC 0.758 | Superseded | Re-run of same method | verified (`@610e874`) |
| R5a | 11–12 Jul | LOF fusion, raw features | run-grouped 5-fold CV | CNN 0.592; OOF hybrid 0.549 | LOF 0.461; LOF weight 0.03 | Rejected | CV chose pure CNN | documented (LOF report) |
| R5b | 12 Jul | LOF fusion, standardized | + train-fit scaler | CNN 0.568; OOF hybrid 0.568 | LOF 0.481 (AUC 0.667); weight 0.00 | Rejected | CV chose pure CNN | documented (LOF report) |
| **F** | **12 Jul** | **Model of record** | CNN only, Round-4 method | **0.6022** | **AUC 0.7629; P 0.5998, R 0.6047; τ 43.298; val flagged 0.2415; test flagged 0.622; PORTAL F1 = 0.63** | **Final** | Best external score | verified (`@HEAD`) + documented (portal) |
| D | 12–13 Jul | Diagnostic re-run | failure-mode analysis | 0.5731 (F<sub>0.5</sub> thr.) / 0.6181 (F1 thr.) | runs 5, 7 at F1 = 0 | Analysis only | Different network | verified |
| E1a | 15 Sep | Threshold A/B | F1 vs F<sub>0.5</sub>, shared network | 0.6067 vs 0.5829 | AUC 0.7689; test flagged 0.842 vs 0.617 | Rejected | Within noise; extreme test flagging | verified |
| E1b | 15 Sep | Per-run median/MAD norm. | score rescaling per run | 0.4050 (F<sub>0.5</sub>) / 0.4585 (F1) | AUC 0.6769 | Rejected | Damages ranking | verified |
| E2a | 15 Sep | Mahalanobis on train (B) | normal model from train windows | 0.4829 | AUC 0.6109; test flagged 0.833 | Rejected | Much worse ranking | verified |
| E2b | 15 Sep | Mahalanobis on all val (C) | unfiltered val windows | 0.5261 | AUC 0.6929 | Diagnostic | Contamination test | documented |
| E2c | 15 Sep | Leave-one-run-out calibration | A fitted on other 9 runs | 0.5251 | AUC 0.6985; P 0.462, R 0.609 | Diagnostic | Measures optimism | documented |

### A.2 Hints in the task brief checked against artifacts

| Hint | Artifact value | Verdict |
|---|---|---|
| CNN val F1 ≈ 0.6022 | 0.6022386 | Confirmed (`outputs/cnn/metrics.json @ HEAD`) |
| ROC-AUC ≈ 0.7629 | 0.7629170 | Confirmed (same file) |
| Threshold experiment F1 ≈ 0.5731 | 0.5731 | Confirmed; F1 at the F<sub>0.5</sub> threshold in the **diagnostic re-run**, not the model of record |
| F1-optimal threshold ≈ 0.6181 | 0.6181 | Confirmed; same diagnostic re-run. The controlled comparison on one network gave 0.6067 vs 0.5829 |
| Mahalanobis 0.4829 F1 / 0.6109 AUC | 0.4829 / 0.6109 | Confirmed; train-fitted normal model (B) |
| Mahalanobis 0.6067 F1 / 0.7689 AUC | 0.6067 / 0.7689 | Confirmed, but this is the **default** validation-normal model (A) with an **F1 threshold**, not a different Mahalanobis fit |
| Final portal F1 = 0.63 | 0.63 | Confirmed as documented (README @ `b456f6d`, EXPERIMENTS.md); the portal page itself is not in the repository |
| "Latest CNN improvement 0.58 → 0.63" | val 0.581 (portal 0.59) → val 0.602 (portal 0.63), **same method** | **Not supported as a method improvement**; a re-run of the same method (Section 7.6) |

### A.3 Leave-one-run-out per-run ROC-AUC (model A → LORO) [documented: EXPERIMENTS.md]

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.767 | 0.933 | 0.771 | 0.872 | 0.810 | 0.807 | 0.799 | 0.759 | 0.521 | 0.891 |
| LORO | 0.756 | 0.930 | 0.658 | 0.774 | 0.753 | 0.776 | 0.724 | 0.724 | 0.496 | 0.870 |

### A.4 Per-run F1 of the diagnostic run at the F<sub>0.5</sub> threshold [verified: `plots/diagnostics/diagnostic_summary.json`]

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Anomaly rate | 0.376 | 0.031 | 0.411 | 0.008 | 0.024 | 0.215 | 0.141 | 0.814 | 0.633 | 0.131 |
| F1 | 0.622 | 0.397 | 0.569 | 0.068 | 0.000 | 0.537 | 0.000 | 0.789 | 0.430 | 0.779 |
| Precision | 0.462 | 0.283 | 0.677 | 0.035 | 0.000 | 0.957 | 0.000 | 0.936 | 0.662 | 0.848 |
| Recall | 0.952 | 0.666 | 0.490 | 1.000 | 0.000 | 0.373 | 0.000 | 0.682 | 0.319 | 0.720 |

---

## Appendix B: Reproducibility Information

### B.1 Key artifacts

| Artifact | Path | Status |
|---|---|---|
| Portal submission (F1 = 0.63) | `outputs/cnn_predictions.csv` | committed `b456f6d`; unchanged in working tree |
| Model-of-record validation metrics | `outputs/cnn/metrics.json @ HEAD` (`git show HEAD:outputs/cnn/metrics.json`) | committed; **overwritten in the working tree** by the 15 Sep variant metrics |
| Same metrics, archived copy | `outputs/experiments/2026-09-15_run1_threshold_ab/metrics_previous_run_of_record.json` | untracked |
| Model-of-record checkpoint | — | **not saved** |
| 15 Sep checkpoint (MPS re-run) | `outputs/cnn/model.pt` | untracked |
| 15 Sep experiment archives | `outputs/experiments/2026-09-15_*/` (metrics, run logs, candidate CSVs) | untracked |
| Candidate submissions (not uploaded) | `outputs/candidates/*.csv` | untracked |
| Round-4 hybrid metrics (CNN weight 1.0) | `outputs/hybrid/metrics.json` | committed |
| LOF fusion results (authoritative) | `reports/lof_experiment_report.pdf` | committed; run outputs not stored |
| Diagnostics | `plots/diagnostics/` | untracked |
| Organizer README and manifest | `git show 42901df:readme.pdf`, `git show 42901df:batch_distillation_public_data/manifest.json` | in git history only (branch-merge commit) |

### B.2 Reproducing the model-of-record pipeline

The model-of-record pipeline is the committed version of `src/pipelines/run_cnn.py` at HEAD. The working-tree version is the 15 Sep experimental variant. To restore and run the committed version:

```bash
git stash
```

```bash
source ts-anomaly-env/bin/activate
```

```bash
python -m src.pipelines.run_cnn
```

Expect metrics near the model of record, but not identical, because of the non-determinism documented in Section 9.5 (±0.02 F1). The committed pipeline overwrites `outputs/cnn_predictions.csv`; back up the 0.63 submission before running it. `git stash pop` restores the working-tree changes afterwards.

### B.3 Configuration of record (`src/config.py`)

`WINDOW_SIZE=200`, `STEP=10`, `EVAL_STEP=1`, `BATCH_SIZE=64`, `EPOCHS=50`, `CNN_EVAL_EVERY=5`, `THRESHOLD_BETA=0.5`, `RANDOM_SEED=42`, Adam lr = 1e-3, Ledoit–Wolf covariance, scaler fitted on train.

### B.4 Items not available / not verified from the project artifacts

- The exact portal F1 aggregation (point-wise / per-run / event-wise / point-adjusted), whether 0.63 is a public, private or combined score, and the test anomaly prevalence.
- The model-of-record checkpoint, its selected epoch, and the cluster run log.
- Scripts and raw outputs for model C and the leave-one-run-out analysis (only the results in EXPERIMENTS.md).
- Raw outputs for Round 1 and Round 3 metrics and for the two LOF cluster runs (only the written reports).
- Whether the committed `hybrid_predictions.csv` versions are byte-identical to the files that received portal scores 0.399/0.471 and 0.59.
- The physical meaning of the anomalies and the definition of "Label (common/all)".
- Any saved output supporting the original "~0.60 / 0.40 / 0.62" window-level baseline figures.
