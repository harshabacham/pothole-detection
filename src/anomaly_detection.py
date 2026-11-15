"""
src/anomaly_detection.py
========================
Core anomaly detection logic and score normalization using Isolation Forest.
Provides utilities to transform raw Isolation Forest outputs into intuitive,
monotonically increasing anomaly scores without claiming probabilistic certainty.
"""

import os
import sys
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def create_isolation_forest(
    n_estimators: int = None,
    contamination: float = None,
    max_samples=None,
    random_state: int = None,
) -> IsolationForest:
    """
    Factory function to instantiate an Isolation Forest model with project defaults.
    """
    if n_estimators is None:
        n_estimators = config.IFOREST_N_ESTIMATORS
    if contamination is None:
        contamination = config.IFOREST_CONTAMINATION
    if max_samples is None:
        max_samples = config.IFOREST_MAX_SAMPLES
    if random_state is None:
        random_state = config.IFOREST_RANDOM_STATE

    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )


def compute_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> tuple:
    """
    Compute anomaly predictions and calibrated anomaly scores.

    Notes on Isolation Forest Scoring:
    ---------------------------------
    In scikit-learn, decision_function(X) yields:
      - Positive values for standard inliers.
      - Negative values for isolated outliers/anomalies.
    To make the score intuitive (higher value = more anomalous):
      anomaly_score = -raw_decision_score
    Additionally, we compute a normalized score scaled between [0.0, 1.0]
    relative to the observed dataset distribution.

    DO NOT interpret this score as a probability or confidence value.
    It is a relative distance/isolation index.

    Parameters
    ----------
    model : IsolationForest
        Fitted IsolationForest model.
    X : np.ndarray
        Feature matrix (N samples, D features).

    Returns
    -------
    tuple
        (is_anomaly_bool, raw_decision_score, anomaly_score, normalized_anomaly_score)
    """
    raw_decision = model.decision_function(X)  # Higher is normal, lower is anomalous
    preds = model.predict(X)  # 1 for inlier, -1 for outlier

    is_anomaly = (preds == -1)

    # Invert so higher score = more anomalous
    anomaly_score = -raw_decision

    # Normalize across the batch to [0.0, 1.0] for consistent visual mapping
    s_min = np.min(anomaly_score)
    s_max = np.max(anomaly_score)
    if s_max > s_min:
        norm_score = (anomaly_score - s_min) / (s_max - s_min)
    else:
        norm_score = np.zeros_like(anomaly_score)

    return is_anomaly, raw_decision, anomaly_score, norm_score

