

---

#   HYBRID ANOMALY DETECTION (FINAL VERSION)

# 1. PROJECT SUMMARY

This project builds a hybrid anomaly detection system for multivariate time-series data.

It combines:
- CNN Autoencoder (deep learning reconstruction model)
- Isolation Forest (statistical anomaly detector)

Final prediction is obtained using a weighted ensemble of both models.

---

# 2. OBJECTIVE

Given raw sensor time-series data:
- Detect anomalies at each timestep
- Output binary classification:
  - 0 → Normal
  - 1 → Anomaly

Data contains multiple independent runs (files).

---

# 3. PIPELINE OVERVIEW

The system follows these steps:

1. Load raw CSV data
2. Normalize using StandardScaler
3. Create sliding windows
4. Train CNN Autoencoder
5. Train Isolation Forest
6. Generate anomaly scores
7. Normalize scores
8. Combine scores using weighted fusion
9. Tune threshold using F1-score
10. Generate Codabench submission

---

# 4. DATA PREPROCESSING

## 4.1 Loading Data

All CSV files are loaded and concatenated:

```

train/, val/, test/

```

Each file represents a separate time-series run.

---

## 4.2 Standard Scaling

We apply:

```

StandardScaler()

```

Purpose:
- Normalize feature distributions
- Improve CNN convergence
- Reduce scale bias

---
# 5 WINDOWING STRATEGY

## 5.1 Sliding Window Concept

We convert time-series into overlapping windows:

```

WINDOW_SIZE = 200
STEP = 10

```

Example:

```

[0–199]
[10–209]
[20–219]
...

```

---

## 5.2 Why Windowing?

- CNN requires fixed-size input
- Captures temporal dependencies
- Converts sequence → supervised-like format

---

# 6. CNN AUTOENCODER

## 6.1 Architecture

Encoder:
- Conv1D (input → 32 filters)
- BatchNorm + ReLU
- Conv1D (32 → 64 filters)
- MaxPooling

Decoder:
- ConvTranspose1D (upsampling)
- Conv1D (reconstruction)

---

## 6.2 Training Objective

Loss function:

```

MSE Loss (Mean Squared Error)

```

Goal:
Reconstruct input time-series as accurately as possible.

---

## 6.3 Anomaly Score

```

CNN Score = Reconstruction Error

```

Interpretation:
- Low error → normal pattern
- High error → anomaly

---

# 7. ISOLATION FOREST

## 7.1 Concept

Isolation Forest detects anomalies by:
- Random partitioning of data
- Measuring how easily a point is isolated

---

## 7.2 Features Used

For each window:

- mean
- standard deviation
- min
- max
- median
- 25th percentile
- 75th percentile

---

## 7.3 Output Score

```

ISO Score = -decision_function(X)

```

Higher score → more anomalous

---

# 8. SCORE NORMALIZATION

Before combining models:

```

score = (score - min) / (max - min)

```

Applied separately for:
- CNN scores
- Isolation Forest scores

Purpose:
- Align both models to same scale [0,1]
- Prevent dominance of one model

---

# 9. HYBRID ENSEMBLE MODEL

Final anomaly score:

```

Final Score =
CNN_WEIGHT × CNN Score +
ISO_WEIGHT × Isolation Forest Score

```

**Update:** `src/pipelines/run_hybrid.py` now sweeps `CNN_WEIGHT` over a small
grid on the validation set and keeps whichever maximizes F1, instead of
hardcoding 0.85/0.15. See section 14 and the main `README.md` for the current
selected weight and results.

---

# 10. THRESHOLD SELECTION

We use validation set to compute:

- Precision
- Recall
- F1 Score

Formula:

```

F1 = 2 * (Precision * Recall) / (Precision + Recall)

```

Best threshold is selected as:

```

argmax(F1)

```

---

# 11. PREDICTION GENERATION

Final decision:

```

prediction = 1 if score > threshold else 0

```

---

# 12. SUBMISSION FORMAT (IMPORTANT)

Codabench requires:

```

run_id,timestep,prediction

```

Where:

- run_id → file index
- timestep → index inside each file
- prediction → 0 or 1

---

# 13. KEY DESIGN CHOICES

## Why CNN + Isolation Forest?

| Model | Strength |
|------|----------|
| CNN Autoencoder | Learns temporal patterns |
| Isolation Forest | Detects statistical outliers |

Combination improves robustness across anomaly types.

---

## Why Weighted Fusion?

- CNN alone misses statistical anomalies
- Isolation Forest alone misses temporal patterns
- Fusion balances both

---

# 14. IMPORTANT PARAMETERS

All parameters now live in `src/config.py` (single source of truth):

```

WINDOW_SIZE = 200
STEP = 10          # train-time stride
EVAL_STEP = 1      # val/test stride - scored densely, averaged per timestep
BATCH_SIZE = 64
EPOCHS = 50        # restored from 10; see README "Future improvements"

CNN_WEIGHT, ISO_WEIGHT   # chosen per-run by the validation weight sweep,
                         # not fixed - see section 9

```

---

# 15. OUTPUT FILE

Generated file:

```

../outputs/hybrid_predictions.csv

```

Contains:
- run_id
- timestep
- prediction

---

# 16. PERFORMANCE

Original (window-level) vs. current (validation, per-timestep - see main
`README.md` "Results" for the full table and methodology caveat):

| Model | Original F1 | Current F1 |
|------|----------|----------|
| Isolation Forest (window-based) | ~0.40 | 0.491 |
| CNN Autoencoder | ~0.60 | 0.408 |
| Hybrid Model | ~0.62 | 0.494 |

---

# 17. FUTURE IMPROVEMENTS

- ~~Learn weights automatically instead of fixed values~~ — done via a
  validation weight sweep (section 9); a learned (e.g. logistic regression)
  combiner is still future work
- Re-tune CNN epoch count against the per-timestep metric (see main README)
- Use LSTM/Transformer or dilated convolutions instead of the current CNN
- Use adaptive threshold per run
- Add frequency-domain features
- Use smoothing filter to reduce noise

---

