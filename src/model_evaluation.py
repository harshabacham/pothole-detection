"""
src/model_evaluation.py
=======================
Evaluation metrics and diagnostic visualizations for unsupervised anomaly detection.
Computes precision, recall, F1-score, and confusion matrix using evaluation-only labels.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

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


def evaluate_anomaly_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None,
) -> dict:
    """
    Compute rigorous evaluation metrics comparing unsupervised detections against
    manually verified ground-truth evaluation labels.

    IMPORTANT: Ground-truth labels are strictly used for post-hoc validation
    and are NEVER provided to the model during training.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (1 for defect/damaged, 0 for normal).
    y_pred : np.ndarray
        Binary model predictions (1 for anomaly, 0 for normal).
    y_scores : np.ndarray, optional
        Continuous anomaly scores for ROC-AUC computation.

    Returns
    -------
    dict
        Dictionary of precision, recall, f1, specificity, and confusion matrix.
    """
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    if y_scores is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            roc_auc = None

    metrics = {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
    }

    print("\n============================================================")
    print("  MODEL EVALUATION (POST-HOC GROUND TRUTH AUDIT)")
    print("============================================================")
    print(f"  Precision       : {prec*100:.2f}%")
    print(f"  Recall          : {rec*100:.2f}%")
    print(f"  F1-Score        : {f1*100:.2f}%")
    print(f"  Specificity     : {specificity*100:.2f}%")
    if roc_auc is not None:
        print(f"  ROC-AUC Score   : {roc_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("============================================================\n")

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = ["Normal", "Anomalous"],
    save_path: str = None,
) -> None:
    """
    Plot and format a 2x2 confusion matrix heatmap.
    """
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Anomaly Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Surface Condition", fontsize=11, labelpad=8)
    ax.set_title("Confusion Matrix (Evaluation Set)", fontsize=12, pad=12)

    # Print numbers in cells
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] model_evaluation: Saved confusion matrix to: {save_path}")
    plt.close("all")


def plot_score_distribution_by_class(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    class_names: list = ["Normal Road", "Damaged Surface"],
    save_path: str = None,
) -> None:
    """
    Plot overlaid histograms comparing anomaly score distributions for normal vs damaged surfaces.
    """
    scores_normal = anomaly_scores[y_true == 0]
    scores_damaged = anomaly_scores[y_true == 1]

    plt.figure(figsize=(9, 5))
    plt.hist(scores_normal, bins=25, alpha=0.6, color="#2980b9", edgecolor="black", label=f"{class_names[0]} (N={len(scores_normal)})")
    plt.hist(scores_damaged, bins=25, alpha=0.6, color="#c0392b", edgecolor="black", label=f"{class_names[1]} (N={len(scores_damaged)})")

    plt.title("Anomaly Score Distribution: Normal vs Damaged Road Patches", fontsize=12, pad=10)
    plt.xlabel("Anomaly Score (Higher = More Isolated / Anomalous)", fontsize=10)
    plt.ylabel("Patch Frequency", fontsize=10)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] model_evaluation: Saved score distribution by class to: {save_path}")
    plt.close("all")

