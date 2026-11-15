"""
src/anomaly_visualization.py
============================
Visualizes patch-level anomaly predictions directly on original road surface imagery.
Produces configurable color-coded bounding overlays and anomaly score annotations.
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


def get_patch_color_bgr(norm_score: float, is_anomaly: bool) -> tuple:
    """
    Map an anomaly score to a configurable BGR color convention:
      - Red    : High anomaly
      - Yellow : Moderate anomaly
      - Green  : Normal road surface
    """
    if norm_score >= config.ANOMALY_SCORE_HIGH:
        return config.COLOR_HIGH_BGR
    elif norm_score >= config.ANOMALY_SCORE_MODERATE or is_anomaly:
        return config.COLOR_MODERATE_BGR
    else:
        return config.COLOR_NORMAL_BGR


def draw_patch_overlay(
    image_bgr: np.ndarray,
    image_predictions_df: pd.DataFrame,
    show_scores: bool = True,
    show_normal: bool = True,
    alpha: float = None,
) -> np.ndarray:
    """
    Render semi-transparent color-coded patch rectangles and anomaly scores on an image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Original road image in BGR format.
    image_predictions_df : pd.DataFrame
        Predictions and coordinate rows corresponding to this specific image.
    show_scores : bool
        Whether to print numeric anomaly score text inside anomalous patches.
    show_normal : bool
        Whether to highlight normal patches in green or leave them transparent.
    alpha : float, optional
        Transparency weighting for patch fill.

    Returns
    -------
    np.ndarray
        Annotated BGR image with overlays.
    """
    if alpha is None:
        alpha = config.OVERLAY_ALPHA

    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    output = image_bgr.copy()

    for _, row in image_predictions_df.iterrows():
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        score = float(row.get("normalized_anomaly_score", 0.0))
        is_anom = bool(row.get("is_anomaly", False))

        color = get_patch_color_bgr(score, is_anom)
        is_high_or_mod = (score >= config.ANOMALY_SCORE_MODERATE) or is_anom

        # Fill rectangle on overlay
        if is_high_or_mod or show_normal:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlay with original
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Draw solid borders and text labels on top of the blended image
    for _, row in image_predictions_df.iterrows():
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        score = float(row.get("normalized_anomaly_score", 0.0))
        is_anom = bool(row.get("is_anomaly", False))
        color = get_patch_color_bgr(score, is_anom)
        is_high_or_mod = (score >= config.ANOMALY_SCORE_MODERATE) or is_anom

        thickness = 2 if is_high_or_mod else 1
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        if show_scores and is_high_or_mod:
            label = f"{score:.2f}"
            font_scale = 0.4
            text_pos = (x1 + 4, y1 + 16)
            # Text background badge for clarity
            cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return output


def draw_patch_grid(
    image_bgr: np.ndarray,
    image_predictions_df: pd.DataFrame,
) -> np.ndarray:
    """
    Draw a clean structural grid of patch boundaries categorized by anomaly level.
    """
    output = image_bgr.copy()
    for _, row in image_predictions_df.iterrows():
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        score = float(row.get("normalized_anomaly_score", 0.0))
        is_anom = bool(row.get("is_anomaly", False))
        color = get_patch_color_bgr(score, is_anom)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
    return output

