"""
src/dbscan_clustering.py
========================
DBSCAN and HDBSCAN density-based clustering and noise identification for road patches.

Unsupervised algorithms used:
  - DBSCAN  : Density-Based Spatial Clustering of Applications with Noise.
              Identifies dense clusters and marks low-density points as noise (-1),
              which correspond to anomalous road patches.
  - HDBSCAN : Hierarchical DBSCAN — eliminates the need for a fixed epsilon
              parameter, adapts cluster density automatically, and is more robust
              on road datasets where defect density varies across image regions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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


# ---------------------------------------------------------------------------
# HDBSCAN support (prefers sklearn.cluster.HDBSCAN, falls back to standalone hdbscan)
# ---------------------------------------------------------------------------
try:
    from sklearn.cluster import HDBSCAN as _HDBSCANCls
    HDBSCAN_AVAILABLE = True
except ImportError:
    try:
        import hdbscan as _hdbscan_lib
        _HDBSCANCls = _hdbscan_lib.HDBSCAN
        HDBSCAN_AVAILABLE = True
    except ImportError:
        _HDBSCANCls = None
        HDBSCAN_AVAILABLE = False


def experiment_dbscan(X: np.ndarray, eps_values: list = None, min_samples_values: list = None) -> pd.DataFrame:
    """
    Grid experiment over DBSCAN parameters (eps, min_samples).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    eps_values : list, optional
        List of eps values to evaluate.
    min_samples_values : list, optional
        List of min_samples values to evaluate.

    Returns
    -------
    pd.DataFrame
        Table summarizing clusters, noise count, and silhouette scores.
    """
    if eps_values is None:
        eps_values = config.DBSCAN_EPS_VALUES
    if min_samples_values is None:
        min_samples_values = config.DBSCAN_MIN_SAMPLES_VALUES

    records = []
    n_samples = len(X)

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)

            unique_labels = set(labels)
            n_noise = int(np.sum(labels == -1))
            n_clusters = len(unique_labels - {-1})

            # Silhouette score is only valid if >= 2 clusters exist (excluding noise or considering all)
            sil = np.nan
            non_noise_mask = labels != -1
            if n_clusters >= 2 and np.sum(non_noise_mask) > n_clusters:
                try:
                    sil = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))
                except Exception:
                    sil = np.nan

            records.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": n_noise / n_samples,
                "silhouette_core": sil,
            })

            print(f"[INFO] dbscan: eps={eps:.2f}, min_samples={ms:2d} -> {n_clusters} clusters, {n_noise} noise ({n_noise/n_samples*100:.1f}%), Sil={sil if not np.isnan(sil) else 0.0:.3f}")

    return pd.DataFrame(records)


def fit_dbscan(X: np.ndarray, eps: float = 1.5, min_samples: int = 5, model_save_path: str = None) -> tuple:
    """
    Fit a final DBSCAN model with specified parameters.

    Returns
    -------
    tuple
        (fitted_model, labels)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump({"dbscan": db, "eps": eps, "min_samples": min_samples}, model_save_path)
        print(f"[INFO] dbscan: Saved model to {model_save_path}")

    return db, labels


# ---------------------------------------------------------------------------
# HDBSCAN — Adaptive Hierarchical Clustering
# ---------------------------------------------------------------------------

def run_hdbscan_clustering(
    X: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    model_save_path: str = None,
) -> tuple:
    """
    Run HDBSCAN (Hierarchical DBSCAN) on a feature matrix.

    Why HDBSCAN over DBSCAN for road anomaly detection:
    ---------------------------------------------------
    Standard DBSCAN requires a fixed epsilon (neighbourhood radius) which is
    difficult to tune when road defect density varies across image patches.
    HDBSCAN builds a cluster hierarchy from multiple epsilon values and
    extracts the most stable clusters automatically — no global epsilon needed.

    Points assigned label -1 are classified as noise (candidate anomalies).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (N x D), typically PCA-reduced features.
    min_cluster_size : int
        Minimum number of points required to form a cluster.
        Smaller values → more sensitive to low-density defects.
    min_samples : int
        Controls how conservative the clustering is; higher = more noise.
    model_save_path : str, optional
        If provided, persist the fitted HDBSCAN model via joblib.

    Returns
    -------
    tuple : (clusterer_or_None, labels)
        clusterer is None when hdbscan package is unavailable (falls back to DBSCAN).
    """
    if not HDBSCAN_AVAILABLE or _HDBSCANCls is None:
        print("[WARNING] HDBSCAN not available in environment. Falling back to DBSCAN.")
        _, labels = fit_dbscan(
            X,
            eps=config.DBSCAN_DEFAULT_EPS,
            min_samples=config.DBSCAN_DEFAULT_MIN_SAMPLES,
        )
        return None, labels

    try:
        clusterer = _HDBSCANCls(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X)
    except Exception as exc:
        print(f"[WARNING] HDBSCAN failed ({exc}), falling back to standard DBSCAN.")
        _, labels = fit_dbscan(
            X,
            eps=config.DBSCAN_DEFAULT_EPS,
            min_samples=config.DBSCAN_DEFAULT_MIN_SAMPLES,
        )
        return None, labels

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    print(f"[INFO] hdbscan: {n_clusters} clusters, {n_noise} noise points "
          f"({n_noise / max(1, len(X)) * 100:.1f}%)")

    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump(
            {"hdbscan": clusterer, "min_cluster_size": min_cluster_size, "min_samples": min_samples},
            model_save_path,
        )
        print(f"[INFO] hdbscan: Saved model to {model_save_path}")

    return clusterer, labels


def visualize_dbscan_2d(
    pca_df: pd.DataFrame,
    labels: np.ndarray,
    title: str = "DBSCAN Clustering & Noise Detection (2D PCA)",
    save_path: str = None,
) -> None:
    """
    Visualize DBSCAN clusters and noise points on the first 2 principal components.
    Noise points are highlighted distinctly.
    """
    if "pca_0" not in pca_df.columns or "pca_1" not in pca_df.columns:
        print("[WARNING] dbscan: 'pca_0' or 'pca_1' missing from DataFrame.")
        return

    plt.figure(figsize=(9, 7))

    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    # Plot clusters
    if np.any(cluster_mask):
        scatter = plt.scatter(
            pca_df.loc[cluster_mask, "pca_0"],
            pca_df.loc[cluster_mask, "pca_1"],
            c=labels[cluster_mask],
            cmap="tab10",
            s=40,
            alpha=0.75,
            label="Dense Clusters",
        )
        plt.colorbar(scatter, label="Cluster ID")

    # Plot noise points
    if np.any(noise_mask):
        plt.scatter(
            pca_df.loc[noise_mask, "pca_0"],
            pca_df.loc[noise_mask, "pca_1"],
            color="black",
            marker="x",
            s=55,
            linewidths=1.5,
            label=f"Noise / Potential Anomalies (N={np.sum(noise_mask)})",
        )

    plt.title(title, fontsize=12, pad=10)
    plt.xlabel("Principal Component 1 (PCA 0)", fontsize=10)
    plt.ylabel("Principal Component 2 (PCA 1)", fontsize=10)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] dbscan: Saved 2D DBSCAN plot to {save_path}")
    plt.close("all")

