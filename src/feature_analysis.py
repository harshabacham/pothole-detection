"""
src/feature_analysis.py
========================
Feature analysis and visualization utilities.
Generates correlation heatmaps, distribution plots, and normalized comparisons.
"""

import os
import sys
import numpy as np
import pandas as pd

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
from src.feature_scaling import get_feature_columns


def plot_correlation_heatmap(features_df: pd.DataFrame, max_features: int = 30, save_path: str = None) -> None:
    """
    Compute and plot a correlation matrix heatmap for top numeric feature columns.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing numeric feature columns.
    max_features : int
        Maximum features to display (chosen by highest variance).
    save_path : str, optional
        Destination path to save the generated figure.
    """
    feat_cols = get_feature_columns(features_df)
    if not feat_cols:
        print("[WARNING] feature_analysis: No features available for correlation heatmap.")
        return

    # Select top features by variance to keep heatmap legible
    variances = features_df[feat_cols].var().sort_values(ascending=False)
    selected_cols = variances.head(max_features).index.tolist()

    corr = features_df[selected_cols].corr().fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(selected_cols)))
    ax.set_yticks(range(len(selected_cols)))
    ax.set_xticklabels(selected_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(selected_cols, fontsize=8)
    ax.set_title(f"Feature Correlation Heatmap (Top {len(selected_cols)} Features by Variance)", fontsize=13, pad=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] feature_analysis: Saved correlation heatmap to {save_path}")
    plt.close("all")


def plot_feature_distributions(features_df: pd.DataFrame, n_features: int = 12, save_path: str = None) -> None:
    """
    Plot histograms for the top n_features features by variance.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing numeric feature columns.
    n_features : int
        Number of top features to plot.
    save_path : str, optional
        Destination path to save the generated figure.
    """
    feat_cols = get_feature_columns(features_df)
    if not feat_cols:
        print("[WARNING] feature_analysis: No features available for distribution plots.")
        return

    variances = features_df[feat_cols].var().sort_values(ascending=False)
    selected_cols = variances.head(n_features).index.tolist()

    n_cols = 3
    n_rows = int(np.ceil(len(selected_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(selected_cols):
        ax = axes[i]
        vals = features_df[col].dropna()
        ax.hist(vals, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.75)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused subplots
    for j in range(len(selected_cols), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Feature Distributions (Top Features by Variance)", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] feature_analysis: Saved feature distributions to {save_path}")
    plt.close("all")


def plot_normalized_comparison(features_df: pd.DataFrame, n_features: int = 15, save_path: str = None) -> None:
    """
    Plot box plots of normalized/scaled values for comparison across features.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing scaled or raw numeric feature columns.
    n_features : int
        Number of top features to compare.
    save_path : str, optional
        Destination path to save the generated figure.
    """
    feat_cols = get_feature_columns(features_df)
    if not feat_cols:
        print("[WARNING] feature_analysis: No features available for normalized comparison.")
        return

    variances = features_df[feat_cols].var().sort_values(ascending=False)
    selected_cols = variances.head(n_features).index.tolist()

    data_to_plot = [features_df[col].dropna().values for col in selected_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    box = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=selected_cols, showfliers=False)

    for patch in box["boxes"]:
        patch.set_facecolor("#4575b4")
        patch.set_alpha(0.7)

    ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Feature Value / Scale", fontsize=10)
    ax.set_title("Normalized Feature Spread Comparison (Box Plot, Outliers Hidden for Clarity)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] feature_analysis: Saved normalized comparison to {save_path}")
    plt.close("all")


def summarize_features(features_df: pd.DataFrame) -> dict:
    """
    Summarize dataset dimensions, feature counts, and variance rankings.

    Parameters
    ----------
    features_df : pd.DataFrame
        Extracted features DataFrame.

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    feat_cols = get_feature_columns(features_df)
    n_patches = len(features_df)
    n_features = len(feat_cols)

    nan_count = int(features_df[feat_cols].isna().sum().sum())
    inf_count = int(np.isinf(features_df[feat_cols].to_numpy()).sum())

    variances = features_df[feat_cols].var().sort_values(ascending=False)
    top_var = variances.head(10).index.tolist()

    summary = {
        "n_patches": n_patches,
        "n_features": n_features,
        "feature_columns": feat_cols,
        "n_nan": nan_count,
        "n_inf": inf_count,
        "top_variance_features": top_var,
    }

    print("\n============================================================")
    print("  FEATURE EXTRACTION SUMMARY")
    print("============================================================")
    print(f"  Total Patches Processed   : {n_patches}")
    print(f"  Total Numerical Features  : {n_features}")
    print(f"  Missing / NaN Values      : {nan_count}")
    print(f"  Infinite Values           : {inf_count}")
    print("  Top 5 High-Variance Features:")
    for rank, f in enumerate(top_var[:5], 1):
        print(f"    {rank}. {f} (var={variances[f]:.4f})")
    print("============================================================\n")

    return summary

