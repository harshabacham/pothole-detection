"""
src/model_prediction.py
=======================
Inference and scoring pipeline for road patches using trained Isolation Forest models.
Generates decision scores, normalized anomaly indices, and anomaly rankings.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.anomaly_detection import compute_anomaly_scores
from src.feature_scaling import META_COLUMNS


def load_isolation_forest(model_path: str = None) -> tuple:
    """
    Load a trained Isolation Forest artifact from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved joblib artifact.

    Returns
    -------
    tuple
        (model, artifact_dict)
    """
    if model_path is None:
        model_path = config.IFOREST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained Isolation Forest not found at: {model_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact
    return artifact, None


def predict_patches(
    model,
    X: np.ndarray,
    metadata_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generate complete anomaly predictions, scores, and rankings for road patches.

    Parameters
    ----------
    model : IsolationForest
        Trained model instance.
    X : np.ndarray
        Feature matrix aligned with metadata_df.
    metadata_df : pd.DataFrame, optional
        Patch metadata containing coordinates and filenames.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with prediction labels, scores, and rankings.
    """
    is_anomaly, raw_decision, anomaly_score, norm_score = compute_anomaly_scores(model, X)

    # Compute ranking: Rank 1 is the most anomalous patch
    # np.argsort of -anomaly_score gives descending order
    order = np.argsort(-anomaly_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    results = {
        "is_anomaly": is_anomaly,
        "prediction_label": np.where(is_anomaly, "anomalous", "normal"),
        "raw_decision_score": raw_decision,
        "anomaly_score": anomaly_score,
        "normalized_anomaly_score": norm_score,
        "anomaly_rank": ranks,
    }

    res_df = pd.DataFrame(results)

    if metadata_df is not None:
        # Prepend available metadata columns
        meta_cols = [c for c in META_COLUMNS if c in metadata_df.columns]
        if not meta_cols:
            meta_cols = metadata_df.columns.tolist()
        res_df = pd.concat([metadata_df[meta_cols].reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)

    return res_df


def save_predictions(predictions_df: pd.DataFrame, save_path: str = None) -> str:
    """
    Save predictions DataFrame to CSV format.
    """
    if save_path is None:
        save_path = config.ANOMALY_PREDICTIONS_CSV

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    predictions_df.to_csv(save_path, index=False)
    print(f"[INFO] model_prediction: Predictions saved to {save_path} ({len(predictions_df)} patches)")
    return save_path

