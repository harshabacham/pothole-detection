"""
src/heatmap.py
================
Spatial anomaly density heatmap generator.
Accurately aggregates overlapping and adjacent patch anomaly scores into continuous,
smoothed 2D defect density maps.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def compute_spatial_anomaly_density(
    image_shape: tuple,
    image_predictions_df: pd.DataFrame,
    blur_sigma: int = None,
) -> np.ndarray:
    """
    Build a continuous per-pixel spatial anomaly density map.

    Properly handles arbitrary patch grids and overlaps via an accumulator
    and visitation counter matrix.

    Parameters
    ----------
    image_shape : tuple
        (height, width) of the original road image.
    image_predictions_df : pd.DataFrame
        Prediction rows containing coordinates (x1, y1, x2, y2) and anomaly scores.
    blur_sigma : int, optional
        Standard deviation for Gaussian smoothing kernel.

    Returns
    -------
    np.ndarray
        Float32 2D array of normalized pixel-level anomaly scores in [0.0, 1.0].
    """
    if blur_sigma is None:
        blur_sigma = config.HEATMAP_BLUR_SIGMA

    h, w = image_shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for _, row in image_predictions_df.iterrows():
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        score = float(row.get("normalized_anomaly_score", 0.0))

        # Clip bounds to valid image geometry
        x1_c, x2_c = max(0, x1), min(w, x2)
        y1_c, y2_c = max(0, y1), min(h, y2)

        if x2_c > x1_c and y2_c > y1_c:
            accum[y1_c:y2_c, x1_c:x2_c] += score
            counts[y1_c:y2_c, x1_c:x2_c] += 1.0

    # Calculate mean score per pixel
    density = np.zeros((h, w), dtype=np.float32)
    visited = counts > 0
    density[visited] = accum[visited] / counts[visited]

    # Smooth the density field to eliminate artificial block boundaries
    if blur_sigma > 0:
        density = cv2.GaussianBlur(density, (0, 0), blur_sigma)

    # Normalize smoothed values to [0.0, 1.0] while preserving unvisited areas at 0.0
    density_roi = density[visited]
    if len(density_roi) > 0 and np.max(density_roi) > np.min(density_roi):
        d_min, d_max = np.min(density_roi), np.max(density_roi)
        density[visited] = (density[visited] - d_min) / (d_max - d_min)
    density[~visited] = 0.0

    return density


def generate_anomaly_heatmap(
    image_bgr: np.ndarray,
    image_predictions_df: pd.DataFrame,
    alpha: float = None,
    blur_sigma: int = None,
) -> tuple:
    """
    Generate a full color heatmap blended onto the original image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Original BGR road image.
    image_predictions_df : pd.DataFrame
        Prediction coordinates and scores.
    alpha : float, optional
        Heatmap opacity factor.
    blur_sigma : int, optional
        Spatial smoothing sigma.

    Returns
    -------
    tuple
        (blended_image_bgr, raw_density_map_float32, standalone_heatmap_bgr)
    """
    if alpha is None:
        alpha = config.HEATMAP_ALPHA

    h, w = image_bgr.shape[:2]
    density = compute_spatial_anomaly_density((h, w), image_predictions_df, blur_sigma=blur_sigma)

    # Convert density to uint8 grayscale
    density_u8 = np.clip(density * 255.0, 0, 255).astype(np.uint8)

    # Apply Jet/Turbo colormap (Blue=Normal, Yellow=Moderate, Red=Severe)
    heatmap_color = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)

    # Identify active ROI mask (where patches exist)
    roi_mask = np.zeros((h, w), dtype=bool)
    for _, row in image_predictions_df.iterrows():
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        roi_mask[y1:y2, x1:x2] = True

    blended = image_bgr.copy()
    # Blend only within the evaluated road surface
    blended[roi_mask] = cv2.addWeighted(
        heatmap_color[roi_mask],
        alpha,
        image_bgr[roi_mask],
        1 - alpha,
        0,
    )

    return blended, density, heatmap_color

