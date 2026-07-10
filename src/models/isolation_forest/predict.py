"""Scoring for a trained Isolation Forest."""


def score(model, X):
    """Higher score = more anomalous (sklearn's decision_function is inverted)."""
    return -model.decision_function(X)
