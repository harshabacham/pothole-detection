"""
src/roi_detection.py
====================
Road ROI (Region of Interest) detection utilities for the Pothole Detection pipeline.

Provides functions to compute, draw, and extract a configurable rectangular ROI
from road images. The ROI is defined by fractional offsets stored in config.py,
making it easy to adjust for different camera angles and mounting positions.

Provides:
  - get_roi_bounds(image_bgr)      : Compute ROI pixel coordinates from config fractions.
  - draw_roi_on_image(image_bgr, roi) : Draw the ROI rectangle on a copy of the image.
  - extract_roi(image_bgr, roi)    : Crop and return the ROI sub-image.
"""

import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config can always be imported.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_roi_bounds(image_bgr: np.ndarray) -> dict:
    """
    Compute the ROI bounding box for a given image using fractional config values.

    The ROI is defined by four fractional offsets in config.py:
        ROI_TOP_FRAC, ROI_BOTTOM_FRAC  — vertical bounds (fraction of image height)
        ROI_LEFT_FRAC, ROI_RIGHT_FRAC  — horizontal bounds (fraction of image width)

    Returns a dict:
        x1, y1 : int  — top-left pixel coordinate of ROI (clipped to image bounds)
        x2, y2 : int  — bottom-right pixel coordinate of ROI (clipped to image bounds)
        width   : int — ROI width in pixels  (x2 - x1)
        height  : int — ROI height in pixels (y2 - y1)

    Raises
    ------
    ValueError
        If the resulting ROI has width <= 0 or height <= 0 after clipping.
    """
    if image_bgr is None:
        raise ValueError("get_roi_bounds received a None image.")

    img_h, img_w = image_bgr.shape[:2]

    # Compute raw pixel coords from fractional config values
    y1_raw = int(round(config.ROI_TOP_FRAC    * img_h))
    y2_raw = int(round(config.ROI_BOTTOM_FRAC * img_h))
    x1_raw = int(round(config.ROI_LEFT_FRAC   * img_w))
    x2_raw = int(round(config.ROI_RIGHT_FRAC  * img_w))

    # Clip to valid image bounds
    x1 = max(0, min(x1_raw, img_w))
    x2 = max(0, min(x2_raw, img_w))
    y1 = max(0, min(y1_raw, img_h))
    y2 = max(0, min(y2_raw, img_h))

    roi_width  = x2 - x1
    roi_height = y2 - y1

    if roi_width <= 0 or roi_height <= 0:
        raise ValueError(
            f"ROI has zero or negative dimensions after clipping: "
            f"x1={x1}, y1={y1}, x2={x2}, y2={y2} "
            f"(image size: {img_w}x{img_h})"
        )

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": roi_width,
        "height": roi_height,
    }


def draw_roi_on_image(
    image_bgr: np.ndarray,
    roi: dict,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Return a copy of image_bgr with the ROI rectangle drawn on it.

    Parameters
    ----------
    image_bgr : np.ndarray
        Source BGR image.
    roi : dict
        ROI bounds dict as returned by get_roi_bounds().
    color : tuple
        BGR colour for the rectangle border. Default is green (0, 255, 0).
    thickness : int
        Line thickness in pixels. Default is 2.

    Returns
    -------
    np.ndarray
        A new BGR image with the rectangle drawn; the original is not modified.
    """
    canvas = image_bgr.copy()
    cv2.rectangle(
        canvas,
        (roi["x1"], roi["y1"]),
        (roi["x2"], roi["y2"]),
        color,
        thickness,
    )
    return canvas


def extract_roi(image_bgr: np.ndarray, roi: dict) -> np.ndarray:
    """
    Crop and return the ROI region from image_bgr.

    Parameters
    ----------
    image_bgr : np.ndarray
        Source BGR image.
    roi : dict
        ROI bounds dict as returned by get_roi_bounds().

    Returns
    -------
    np.ndarray
        The cropped sub-image in BGR colour space.
    """
    return image_bgr[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]].copy()

