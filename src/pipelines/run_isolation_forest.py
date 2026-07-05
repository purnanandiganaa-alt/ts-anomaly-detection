"""
Standalone Isolation Forest baseline (tabular lag/rolling features).

Faithful, deduplicated port of notebooks/02_anomaly_detection_isolation_forest.ipynb.
Run from the repository root: python -m src.pipelines.run_isolation_forest
"""

import glob
import json

import pandas as pd

from src import config
from src.core.feature_pipeline import create_features
from src.models.isolation_forest.train import train_isolation_forest

OUTPUT_DIR = config.OUTPUT_DIR / "isolation_forest"


def load_val_labels():
    files = sorted(glob.glob(str(config.VAL_LABELS_PATH / "*.csv")))
    labels = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return labels["label"].values


def main():
    print("Building tabular features...")
    X_train = create_features(str(config.TRAIN_PATH), str(config.SELECTED_SENSORS_PATH))
    X_val = create_features(str(config.VAL_PATH), str(config.SELECTED_SENSORS_PATH))
    X_test = create_features(str(config.TEST_PATH), str(config.SELECTED_SENSORS_PATH))

    y_val = load_val_labels()

    print("Training Isolation Forest...")
    model = train_isolation_forest(
        X_train,
        n_estimators=config.TABULAR_IF_N_ESTIMATORS,
        contamination=config.TABULAR_IF_CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )

    val_scores = -model.decision_function(X_val)
    test_scores = -model.decision_function(X_test)

    # NOTE (ported as-is, unchanged in Phase 1): this notebook's own threshold
    # search never had the off-by-one bug present in the CNN/hybrid notebooks
    # (thresholds[argmax(f1)], no shift) - kept here to faithfully reproduce
    # the committed baseline metrics. Phase 2 unifies this with the shared,
    # corrected src.evaluation.metrics.find_best_threshold.
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_threshold = thresholds[np.argmax(f1)]

    y_pred = (val_scores >= best_threshold).astype(int)
    test_pred = (test_scores >= best_threshold).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        "roc_auc": roc_auc_score(y_val, val_scores),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "best_threshold": float(best_threshold),
        "validation_anomaly_rate": float(y_pred.mean()),
        "test_anomaly_rate": float(test_pred.mean()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "metrics" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    submission = pd.DataFrame({"id": range(len(test_pred)), "label": test_pred})
    submission.to_csv(OUTPUT_DIR / "predictions" / "submission_if_final.csv", index=False)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
