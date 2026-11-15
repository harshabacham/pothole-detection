"""
src/kmeans_clustering.py
========================
K-Means clustering implementation and evaluation for road patches.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

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


def experiment_kmeans(X: np.ndarray, k_range: list = None) -> dict:
    """
    Run K-Means across a range of K values, computing inertia and validation metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (PCA projected or scaled features).
    k_range : list, optional
        List of cluster counts to evaluate.

    Returns
    -------
    dict
        Evaluation results with keys: 'k_values', 'inertias', 'silhouettes',
        'davies_bouldins', 'models'.
    """
    if k_range is None:
        k_range = config.KMEANS_K_RANGE

    inertias = []
    silhouettes = []
    davies_bouldins = []
    models = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(float(kmeans.inertia_))
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))

        silhouettes.append(sil)
        davies_bouldins.append(db)
        models[k] = kmeans

        print(f"[INFO] kmeans: K={k} | Inertia={kmeans.inertia_:.2f} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

    return {
        "k_values": k_range,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "davies_bouldins": davies_bouldins,
        "models": models,
    }


def plot_elbow_and_metrics(results: dict, save_path: str = None) -> None:
    """
    Plot Elbow curve (Inertia), Silhouette Score, and Davies-Bouldin Index.
    """
    k_vals = results["k_values"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Inertia
    axes[0].plot(k_vals, results["inertias"], "b-o", linewidth=2)
    axes[0].set_title("Elbow Curve (Inertia)", fontsize=11)
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[0].set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Silhouette
    axes[1].plot(k_vals, results["silhouettes"], "g-s", linewidth=2)
    axes[1].set_title("Silhouette Score (Higher is Better)", fontsize=11)
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[1].set_ylabel("Silhouette Score", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Davies-Bouldin
    axes[2].plot(k_vals, results["davies_bouldins"], "r-^", linewidth=2)
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=11)
    axes[2].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[2].set_ylabel("Davies-Bouldin Index", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] kmeans: Saved evaluation metrics plot to {save_path}")
    plt.close("all")


def fit_kmeans(X: np.ndarray, k: int = 3, model_save_path: str = None) -> tuple:
    """
    Fit a final K-Means model with chosen K.

    Returns
    -------
    tuple
        (fitted_model, labels)
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump({"kmeans": kmeans, "k": k}, model_save_path)
        print(f"[INFO] kmeans: Saved model to {model_save_path}")

    return kmeans, labels


def visualize_cluster_gallery(
    metadata_df: pd.DataFrame,
    labels: np.ndarray,
    n_samples_per_cluster: int = 4,
    save_path: str = None,
) -> None:
    """
    Display visual samples from each cluster to visually inspect cluster contents.
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(
        n_clusters,
        n_samples_per_cluster,
        figsize=(n_samples_per_cluster * 2.5, n_clusters * 2.5),
    )

    if n_clusters == 1:
        axes = np.expand_dims(axes, 0)
    if n_samples_per_cluster == 1:
        axes = np.expand_dims(axes, 1)

    for r, cl in enumerate(unique_clusters):
        cluster_indices = np.where(labels == cl)[0]
        chosen = np.random.RandomState(42).choice(
            cluster_indices,
            size=min(len(cluster_indices), n_samples_per_cluster),
            replace=False,
        )

        for c in range(n_samples_per_cluster):
            ax = axes[r, c]
            if c < len(chosen):
                idx = chosen[c]
                row = metadata_df.iloc[idx]
                p_path = row.get("color_path") or row.get("gray_path")
                if p_path and os.path.exists(p_path):
                    img = cv2.imread(p_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                else:
                    ax.text(0.5, 0.5, "Image missing", ha="center", va="center")
                ax.set_title(f"Cl {cl}: #{idx}", fontsize=8)
            ax.axis("off")

    plt.suptitle("K-Means Cluster Visual Sample Gallery", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] kmeans: Saved cluster sample gallery to {save_path}")
    plt.close("all")

