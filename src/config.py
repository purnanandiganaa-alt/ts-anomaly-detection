"""
Central configuration for the anomaly detection pipelines.

Every hyperparameter and path here used to be a magic number copy-pasted
across notebooks 03/04/05 with small, undocumented drift between copies.
Scripts and notebooks should import from here instead of redefining values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw")
TRAIN_PATH = DATA_DIR / "train"
VAL_PATH = DATA_DIR / "val"
VAL_LABELS_PATH = DATA_DIR / "val_labels"
TEST_PATH = DATA_DIR / "test"

PROCESSED_DIR = Path("data/processed")
SENSOR_FEATURES_PATH = PROCESSED_DIR / "df_sensor_features.csv"
SELECTED_SENSORS_PATH = PROCESSED_DIR / "selected_sensors.json"

OUTPUT_DIR = Path("outputs")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Windowing (shared by the CNN autoencoder and the window-based Isolation
# Forest branch of the hybrid pipeline)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 200
STEP = 10  # train-time stride, used by all pipelines

# Validation/test scoring stride. The standalone CNN notebook (04) scores
# val/test with step=1 (one prediction per timestep); the hybrid notebook (05)
# reused STEP=10 for val/test too, which is coarser. run_cnn.py uses this;
# run_hybrid.py still uses STEP for val/test in Phase 1 to faithfully match
# notebook 05 - see Phase 2 notes in run_hybrid.py.
EVAL_STEP = 1

# ---------------------------------------------------------------------------
# CNN autoencoder
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
# Fix (Phase 2): notebook 04's own tuning experiment already measured that
# convergence needs ~50 epochs at this window size (F1 0.4099 -> 0.4699 going
# from 15 to 50 epochs), but the pipeline that actually produced submissions
# had reverted to 10. Restoring the epoch count the project's own tuning
# already validated.
EPOCHS = 50

# Round 2: 50 epochs was only validated against the old window-level metric.
# Training loss fell smoothly the whole way with no plateau, suggesting
# possible overtraining against the stricter per-timestep metric now in use
# - check validation every N epochs during training and keep the
# best-scoring checkpoint instead of blindly using the final epoch.
CNN_EVAL_EVERY = 5

# ---------------------------------------------------------------------------
# Isolation Forest — tabular lag/rolling features (standalone baseline,
# notebooks/02_anomaly_detection_isolation_forest.ipynb)
# ---------------------------------------------------------------------------
TABULAR_IF_N_ESTIMATORS = 300
TABULAR_IF_CONTAMINATION = 0.24

# ---------------------------------------------------------------------------
# Isolation Forest — per-window statistical features (used inside the
# hybrid pipeline, notebooks/05_hybrid_model.ipynb)
#
# No fixed contamination here (Round 2 fix): it was hand-set to 0.2 and never
# tuned, same issue the fusion weight had - now chosen per-run by
# src.pipelines.run_hybrid.sweep_iso_contamination against validation.
# ---------------------------------------------------------------------------
WINDOW_IF_N_ESTIMATORS = 200

# ---------------------------------------------------------------------------
# Hybrid score fusion
#
# No fixed CNN_WEIGHT/ISO_WEIGHT here (Phase 2 fix): the weight used to be
# hand-picked once; it's now chosen per-run by sweeping
# src.pipelines.run_hybrid.CNN_WEIGHT_GRID against the validation set.
# ---------------------------------------------------------------------------
