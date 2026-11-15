"""
src/severity_estimation.py
==========================
Visual feature extraction and severity estimation for localized road defect candidates.
Calculates region geometry, edge density, texture irregularity, and relative road contrast.
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
from src.severity_scoring import (
    calculate_severity_score,
    assign_severity_category,
    get_severity_color_bgr,
)


def compute_candidate_contrast(
    gray_image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    margin: int = None,
) -> float:
    """
    Compute photometric contrast between the candidate region and its surrounding road ring.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale representation of the full road image.
    x, y, w, h : int
        Bounding box coordinates.
    margin : int, optional
        Pixel width of the surrounding reference ring.

    Returns
    -------
    float
        Normalized contrast in [0.0, 1.0].
    """
    if margin is None:
        margin = config.CONTRAST_RING_MARGIN

    img_h, img_w = gray_image.shape[:2]

    # Inner defect region
    defect_roi = gray_image[y : y + h, x : x + w]
    if defect_roi.size == 0:
        return 0.0
    mean_defect = float(np.mean(defect_roi))

    # Outer surrounding ring
    y1_out = max(0, y - margin)
    y2_out = min(img_h, y + h + margin)
    x1_out = max(0, x - margin)
    x2_out = min(img_w, x + w + margin)

    outer_roi = gray_image[y1_out:y2_out, x1_out:x2_out].copy()

    # Create mask to exclude inner defect pixels from the surrounding background
    mask = np.ones(outer_roi.shape, dtype=bool)
    rel_y1 = y - y1_out
    rel_y2 = rel_y1 + h
    rel_x1 = x - x1_out
    rel_x2 = rel_x1 + w
    mask[rel_y1:rel_y2, rel_x1:rel_x2] = False

    surround_pixels = outer_roi[mask]
    if surround_pixels.size == 0:
        return 0.0

    mean_surround = float(np.mean(surround_pixels))
    contrast_raw = abs(mean_defect - mean_surround) / 255.0

    return float(np.clip(contrast_raw, 0.0, 1.0))


def estimate_candidate_severity(
    candidate: dict,
    gray_image: np.ndarray,
) -> dict:
    """
    Compute physical visual indicators and estimate preliminary severity for a candidate.

    Parameters
    ----------
    candidate : dict
        Candidate record from Phase 7 containing x, y, width, height, etc.
    gray_image : np.ndarray
        Grayscale road image.

    Returns
    -------
    dict
        Enriched candidate dictionary with computed visual indicators and severity scores.
    """
    cand = candidate.copy()
    x, y = int(cand["x"]), int(cand["y"])
    w, h = int(cand["width"]), int(cand["height"])
    img_h, img_w = gray_image.shape[:2]

    # 1. Geometry & Aspect Ratio
    aspect_ratio = float(w / h) if h > 0 else 1.0
    pixel_area = int(cand.get("pixel_area", w * h))

    # 2. Texture Irregularity & Edge Dynamics
    defect_roi = gray_image[y : y + h, x : x + w]
    if defect_roi.size > 0:
        texture_irreg = float(np.std(defect_roi))
        canny_edges = cv2.Canny(defect_roi, config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2)
        edge_density = float(np.count_nonzero(canny_edges)) / float(defect_roi.size)
    else:
        texture_irreg = 0.0
        edge_density = 0.0

    # 3. Relative Photometric Contrast
    contrast = compute_candidate_contrast(gray_image, x, y, w, h)

    # 4. Normalized Indicator Calibration
    # Reference area: typical single 64x64 patch = 4096 px; 4-patch defect = 16384 px
    s_size = float(np.clip(pixel_area / 16384.0, 0.0, 1.0))
    s_anom = float(np.clip(cand.get("avg_anomaly_score", 0.5), 0.0, 1.0))
    s_edge = float(np.clip(edge_density * 3.5, 0.0, 1.0))
    s_cont = float(np.clip(contrast * 3.0, 0.0, 1.0))

    # 5. Composite Severity Calculation
    severity_score = calculate_severity_score(
        relative_size=s_size,
        anomaly_strength=s_anom,
        edge_density=s_edge,
        relative_contrast=s_cont,
    )
    severity_category = assign_severity_category(severity_score)

    cand.update({
        "aspect_ratio": round(aspect_ratio, 3),
        "texture_irregularity": round(texture_irreg, 3),
        "edge_density": round(edge_density, 4),
        "road_contrast": round(contrast, 4),
        "severity_score": round(severity_score, 4),
        "severity_category": severity_category,
    })

    return cand


def draw_severity_annotations(
    image_bgr: np.ndarray,
    evaluated_candidates: list,
) -> np.ndarray:
    """
    Draw color-coded bounding boxes and badges based on the estimated severity level.
      - Green  : Low severity
      - Orange : Medium severity
      - Red    : High severity
    """
    output = image_bgr.copy()

    for cand in evaluated_candidates:
        x, y = int(cand["x"]), int(cand["y"])
        w, h = int(cand["width"]), int(cand["height"])
        cid = cand.get("candidate_id", "defect")
        score = cand.get("severity_score", 0.0)
        cat = cand.get("severity_category", "Low")
        color = get_severity_color_bgr(cat)

        # Primary box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Corner styling
        c_len = min(10, w // 4, h // 4)
        if c_len > 2:
            cv2.line(output, (x, y), (x + c_len, y), (255, 255, 255), 3)
            cv2.line(output, (x, y), (x, y + c_len), (255, 255, 255), 3)
            cv2.line(output, (x + w, y + h), (x + w - c_len, y + h), (255, 255, 255), 3)
            cv2.line(output, (x + w, y + h), (x + w, y + h - c_len), (255, 255, 255), 3)

        # Badge: [SEVERITY | ID | S:0.XX | WxH]
        label = f"{cat.upper()} | {cid} | Sev:{score:.2f} | {w}x{h}"
        font_scale = 0.40
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        badge_y1 = max(0, y - th - 6)
        badge_y2 = y
        cv2.rectangle(output, (x, badge_y1), (x + tw + 6, badge_y2), (0, 0, 0), -1)
        cv2.rectangle(output, (x, badge_y1), (x + tw + 6, badge_y2), color, 1)

        cv2.putText(
            output,
            label,
            (x + 3, badge_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output

