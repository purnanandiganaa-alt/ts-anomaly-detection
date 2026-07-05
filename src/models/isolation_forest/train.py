"""Training entry points for both Isolation Forest variants used in this project."""

from sklearn.ensemble import IsolationForest


def train_isolation_forest(X_train, n_estimators, contamination, random_state=42):
    """Fit an IsolationForest on a feature matrix (tabular or per-window)."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model
