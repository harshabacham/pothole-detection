"""
src/model_training.py
=====================
Model training pipeline for Isolation Forest anomaly detection.
Executes hyperparameter experiments over contamination rates and handles model persistence.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.anomaly_detection import create_isolation_forest, compute_anomaly_scores


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = None,
    n_estimators: int = None,
    feature_names: list = None,
    save_path: str = None,
) -> tuple:
    """
    Train an Isolation Forest model without using any manual defect labels.

    Parameters
    ----------
    X : np.ndarray
        Training feature representation (PCA or scaled features).
    contamination : float, optional
        Expected outlier fraction.
    n_estimators : int, optional
        Number of decision trees.
    feature_names : list, optional
        Names of the features used in training.
    save_path : str, optional
        Destination path to save the model artifact via joblib.

    Returns
    -------
    tuple
        (fitted_model, artifact_dict)
    """
    if contamination is None:
        contamination = config.IFOREST_CONTAMINATION
    if n_estimators is None:
        n_estimators = config.IFOREST_N_ESTIMATORS

    model = create_isolation_forest(
        n_estimators=n_estimators,
        contamination=contamination,
    )

    print(f"[INFO] model_training: Fitting IsolationForest (n_estimators={n_estimators}, contamination={contamination})...")
    model.fit(X)

    artifact = {
        "model": model,
        "n_estimators": n_estimators,
        "contamination": contamination,
        "feature_names": feature_names,
        "n_samples_trained": len(X),
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(artifact, save_path)
        print(f"[INFO] model_training: Model artifact saved successfully to: {save_path}")

    return model, artifact


def experiment_contamination(
    X: np.ndarray,
    contamination_levels: list = None,
) -> pd.DataFrame:
    """
    Evaluate the sensitivity of anomaly isolation across different contamination parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    contamination_levels : list, optional
        List of contamination floats to test.

    Returns
    -------
    pd.DataFrame
        Summary table comparing anomaly counts and score dynamics.
    """
    if contamination_levels is None:
        contamination_levels = config.IFOREST_CONTAMINATION_EXPERIMENTS

    records = []
    n_samples = len(X)

    for c in contamination_levels:
        model = create_isolation_forest(contamination=c)
        model.fit(X)

        is_anom, raw_dec, anom_score, norm_score = compute_anomaly_scores(model, X)
        n_anom = int(np.sum(is_anom))

        records.append({
            "contamination": c,
            "detected_anomalies": n_anom,
            "anomaly_percentage": (n_anom / n_samples) * 100.0,
            "raw_decision_min": float(np.min(raw_dec)),
            "raw_decision_mean": float(np.mean(raw_dec)),
            "anomaly_score_mean": float(np.mean(anom_score)),
            "anomaly_score_std": float(np.std(anom_score)),
            "anomaly_score_max": float(np.max(anom_score)),
        })

        print(f"[INFO] model_training: Contamination={c:.2f} -> {n_anom}/{n_samples} anomalous patches ({(n_anom/n_samples)*100:.1f}%)")

    return pd.DataFrame(records)


def plot_contamination_experiments(
    exp_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Plot detected anomaly count and score distribution across contamination rates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 1. Anomaly Count Bar Chart
    axes[0].bar(
        [f"{c:.2f}" for c in exp_df["contamination"]],
        exp_df["detected_anomalies"],
        color="#c0392b",
        alpha=0.8,
        edgecolor="black",
    )
    axes[0].set_title("Detected Anomalous Patches vs Contamination", fontsize=11)
    axes[0].set_xlabel("Contamination Parameter", fontsize=10)
    axes[0].set_ylabel("Anomalous Patch Count", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Score Metrics Line Plot
    axes[1].plot(
        exp_df["contamination"],
        exp_df["anomaly_score_mean"],
        "b-o",
        label="Mean Anomaly Score",
        linewidth=2,
    )
    axes[1].plot(
        exp_df["contamination"],
        exp_df["anomaly_score_max"],
        "r--s",
        label="Max Anomaly Score",
        linewidth=2,
    )
    axes[1].set_title("Anomaly Score Statistics across Contamination Rates", fontsize=11)
    axes[1].set_xlabel("Contamination Parameter", fontsize=10)
    axes[1].set_ylabel("Anomaly Score (-decision_function)", fontsize=10)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Isolation Forest Contamination Experiments", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] model_training: Saved contamination experiment plot to: {save_path}")
    plt.close("all")

