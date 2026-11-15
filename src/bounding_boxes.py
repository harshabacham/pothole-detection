"""
src/bounding_boxes.py
=====================
Bounding box extraction, spatial feature pooling, and non-maximum suppression (NMS)
for localized road surface anomaly candidates.
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def compute_iou(box1: dict, box2: dict) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding box dictionaries
    containing 'x', 'y', 'width', 'height'.
    """
    x1_a, y1_a = box1["x"], box1["y"]
    x2_a, y2_a = x1_a + box1["width"], y1_a + box1["height"]

    x1_b, y1_b = box2["x"], box2["y"]
    x2_b, y2_b = x1_b + box2["width"], y1_b + box2["height"]

    # Calculate intersection
    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = box1["width"] * box1["height"]
    area_b = box2["width"] * box2["height"]
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def apply_nms(candidates: list, iou_threshold: float = None) -> list:
    """
    Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping candidate detections.
    Prioritizes candidates with higher average anomaly score.
    """
    if iou_threshold is None:
        iou_threshold = config.NMS_IOU_THRESHOLD

    if not candidates:
        return []

    # Sort descending by max/avg anomaly score
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.get("avg_anomaly_score", 0.0), c.get("pixel_area", 0)),
        reverse=True,
    )

    keep = []
    while sorted_candidates:
        current = sorted_candidates.pop(0)
        keep.append(current)

        # Filter out remaining boxes that have IoU greater than threshold
        remaining = []
        for other in sorted_candidates:
            if compute_iou(current, other) < iou_threshold:
                remaining.append(other)
        sorted_candidates = remaining

    return keep


def extract_candidate_regions(
    contours: list,
    density_map: np.ndarray,
    image_shape: tuple,
    image_filename: str = "",
    min_area: int = None,
    max_area_frac: float = None,
) -> list:
    """
    Convert OpenCV contours into candidate defect bounding boxes with spatial statistics.

    Parameters
    ----------
    contours : list
        List of candidate contours from region grouping.
    density_map : np.ndarray
        2D continuous anomaly density map from Phase 6.
    image_shape : tuple
        (height, width) of the source image.
    image_filename : str, optional
        Source image reference.
    min_area : int, optional
        Minimum pixel area threshold.
    max_area_frac : float, optional
        Maximum image area fraction.

    Returns
    -------
    list of candidate dicts
    """
    if min_area is None:
        min_area = config.MIN_CANDIDATE_AREA
    if max_area_frac is None:
        max_area_frac = config.MAX_CANDIDATE_AREA_FRAC

    h, w = image_shape[:2]
    max_area = int(h * w * max_area_frac)

    candidates = []

    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_area = bw * bh
        contour_area = int(cv2.contourArea(cnt))

        # Size filtering
        if box_area < min_area:
            continue
        if box_area > max_area:
            continue

        # Extract local anomaly density
        sub_density = density_map[y : y + bh, x : x + bw]
        if sub_density.size > 0:
            avg_score = float(np.mean(sub_density))
            max_score = float(np.max(sub_density))
        else:
            avg_score = 0.0
            max_score = 0.0

        candidates.append({
            "image_filename": image_filename,
            "x": int(x),
            "y": int(y),
            "width": int(bw),
            "height": int(bh),
            "pixel_area": contour_area if contour_area > 0 else box_area,
            "box_area": int(box_area),
            "avg_anomaly_score": avg_score,
            "max_anomaly_score": max_score,
        })

    # Apply NMS to remove redundant boxes
    filtered_candidates = apply_nms(candidates)

    # Assign sequential candidate IDs
    for idx, c in enumerate(filtered_candidates, 1):
        c["candidate_id"] = f"{os.path.splitext(image_filename)[0]}_cand_{idx:02d}"

    return filtered_candidates

