"""
src/region_grouping.py
======================
Spatial grouping of anomalous road patches using mathematical morphology
and connected component analysis.
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


def create_binary_anomaly_mask(
    image_shape: tuple,
    image_predictions_df: pd.DataFrame,
    score_threshold: float = None,
) -> np.ndarray:
    """
    Construct a binary spatial mask where pixels inside anomalous patches are active (255).

    Parameters
    ----------
    image_shape : tuple
        (height, width) of the original road image.
    image_predictions_df : pd.DataFrame
        Prediction rows containing (x1, y1, x2, y2), normalized_anomaly_score, and is_anomaly.
    score_threshold : float, optional
        Normalized score threshold for anomaly inclusion.

    Returns
    -------
    np.ndarray
        2D uint8 binary mask (0 or 255).
    """
    if score_threshold is None:
        score_threshold = config.LOCALIZATION_SCORE_THRESHOLD

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for _, row in image_predictions_df.iterrows():
        score = float(row.get("normalized_anomaly_score", 0.0))
        is_anom = bool(row.get("is_anomaly", False))

        if score >= score_threshold or is_anom:
            x1, y1 = int(row["x1"]), int(row["y1"])
            x2, y2 = int(row["x2"]), int(row["y2"])

            x1_c, x2_c = max(0, x1), min(w, x2)
            y1_c, y2_c = max(0, y1), min(h, y2)

            if x2_c > x1_c and y2_c > y1_c:
                mask[y1_c:y2_c, x1_c:x2_c] = 255

    return mask


def apply_morphological_grouping(
    binary_mask: np.ndarray,
    kernel_size: tuple = None,
) -> np.ndarray:
    """
    Merge neighboring anomalous patches and bridge small internal voids.

    Applies:
      1. Morphological Closing (dilation followed by erosion) to fuse adjacent patches.
      2. Morphological Opening (erosion followed by dilation) to filter out isolated noise spurs.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary uint8 mask.
    kernel_size : tuple, optional
        Structuring element dimensions.

    Returns
    -------
    np.ndarray
        Spatially unified binary mask.
    """
    if kernel_size is None:
        kernel_size = config.MORPH_KERNEL_SIZE

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Closing to bridge gaps between neighboring patches
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Opening to smooth boundaries and prune single-pixel artifacts
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened


def extract_candidate_contours(binary_mask: np.ndarray) -> list:
    """
    Extract external contours from a unified binary defect mask.

    Returns
    -------
    list of OpenCV contour arrays.
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours

