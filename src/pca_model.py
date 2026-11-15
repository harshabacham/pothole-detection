"""
src/pca_model.py
================
Principal Component Analysis (PCA) for dimensionality reduction of road-patch features.
Standardizes features and projects into optimal orthogonal component space.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
from src.feature_scaling import get_feature_columns, META_COLUMNS


def fit_pca(
    features_df: pd.DataFrame,
    n_components: int = None,
    variance_target: float = 0.90,
    model_save_path: str = None,
) -> tuple:
    """
    Standardize numeric features and fit PCA on them.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing numeric features.
    n_components : int, optional
        Explicit number of components to keep.
    variance_target : float
        Target cumulative explained variance ratio if n_components is None.
    model_save_path : str, optional
        Path to save fitted standardizer and PCA object.

    Returns
    -------
    tuple
        (fitted_pca_model, fitted_standardizer, n_components_selected, feature_cols)
    """
    feat_cols = get_feature_columns(features_df)
    if not feat_cols:
        raise ValueError("No numeric feature columns available for PCA.")

    X = features_df[feat_cols].to_numpy()

    # Standardize features so variance is comparable across heterogeneous scales
    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)

    if n_components is None:
        full_pca = PCA(random_state=42)
        full_pca.fit(X_std)
        cumsum = np.cumsum(full_pca.explained_variance_ratio_)
        idx = np.where(cumsum >= variance_target)[0]
        if len(idx) > 0:
            optimal_k = int(idx[0] + 1)
        else:
            optimal_k = len(feat_cols)
        optimal_k = max(2, min(optimal_k, len(feat_cols)))
        print(f"[INFO] pca_model: {optimal_k} components explain {cumsum[optimal_k-1]*100:.2f}% of variance (target: {variance_target*100:.1f}%)")
        n_components = optimal_k

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_std)

    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump(
            {
                "pca": pca,
                "standardizer": standardizer,
                "feature_columns": feat_cols,
                "n_components": n_components,
            },
            model_save_path,
        )
        print(f"[INFO] pca_model: Saved fitted PCA & standardizer to {model_save_path}")

    return pca, standardizer, n_components, feat_cols


def transform_pca(
    features_df: pd.DataFrame,
    pca_model,
    standardizer=None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Standardize and project feature columns onto the fitted principal components.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with original or scaled features.
    pca_model : PCA
        Fitted PCA model.
    standardizer : StandardScaler, optional
        Fitted standardizer. If None, expects X to already be standardized.
    feature_cols : list, optional
        Columns matching the fitted PCA model.

    Returns
    -------
    pd.DataFrame
        DataFrame retaining metadata columns with 'pca_0', 'pca_1', ...
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    X = features_df[feature_cols].to_numpy()
    if standardizer is not None:
        X = standardizer.transform(X)

    X_pca = pca_model.transform(X)

    pca_col_names = [f"pca_{i}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_col_names, index=features_df.index)

    # Retain metadata
    meta_present = [col for col in META_COLUMNS if col in features_df.columns]
    res_df = pd.concat([features_df[meta_present], pca_df], axis=1)

    return res_df


def plot_explained_variance(pca_model, variance_target: float = 0.90, save_path: str = None) -> None:
    """
    Plot scree plot and cumulative explained variance.
    """
    var_exp = pca_model.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)
    n_comp = len(var_exp)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(range(1, n_comp + 1), var_exp * 100, alpha=0.6, color="#1f77b4", label="Individual explained variance")
    ax1.set_xlabel("Principal Component Index", fontsize=11)
    ax1.set_ylabel("Individual Explained Variance (%)", color="#1f77b4", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(range(1, n_comp + 1), cum_var_exp * 100, "r-o", linewidth=2, label="Cumulative explained variance")
    ax2.axhline(y=variance_target * 100, color="gray", linestyle="--", label=f"Target ({variance_target*100:.0f}%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="r", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="r")

    plt.title(f"PCA Scree Plot and Cumulative Explained Variance ({n_comp} Components)", fontsize=13)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] pca_model: Saved explained variance plot to {save_path}")
    plt.close("all")


def plot_pca_2d(pca_df: pd.DataFrame, title: str = "2D PCA Projection of Road Patches", save_path: str = None, color_by=None) -> None:
    """
    Generate a 2D scatter plot of the first two principal components.
    """
    if "pca_0" not in pca_df.columns or "pca_1" not in pca_df.columns:
        print("[WARNING] pca_model: 'pca_0' or 'pca_1' not found in DataFrame.")
        return

    plt.figure(figsize=(9, 7))
    if color_by is not None:
        scatter = plt.scatter(pca_df["pca_0"], pca_df["pca_1"], c=color_by, cmap="tab10", alpha=0.7, edgecolors="none", s=35)
        plt.colorbar(scatter, label="Group / Cluster")
    else:
        plt.scatter(pca_df["pca_0"], pca_df["pca_1"], color="#2b5c8f", alpha=0.7, edgecolors="none", s=35)

    plt.title(title, fontsize=12, pad=10)
    plt.xlabel("Principal Component 1 (PCA 0)", fontsize=10)
    plt.ylabel("Principal Component 2 (PCA 1)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] pca_model: Saved 2D PCA plot to {save_path}")
    plt.close("all")


def load_pca_model(model_path: str):
    """
    Load a saved PCA model and standardizer artifact.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PCA model file not found at: {model_path}")

    data = joblib.load(model_path)
    if isinstance(data, dict) and "pca" in data:
        return data["pca"], data.get("standardizer"), data["feature_columns"], data["n_components"]
    return data, None, None, getattr(data, "n_components_", None)
