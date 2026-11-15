"""
src/localization.py
===================
Master localization pipeline for road surface defect candidates.
Converts patch predictions into spatially merged bounding boxes, renders
diagnostic overlays, and persists defect candidate tables.
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
from src.heatmap import compute_spatial_anomaly_density
from src.region_grouping import (
    create_binary_anomaly_mask,
    apply_morphological_grouping,
    extract_candidate_contours,
)
from src.bounding_boxes import extract_candidate_regions


def draw_localized_candidates(
    image_bgr: np.ndarray,
    candidates: list,
    color: tuple = (0, 0, 230),  # Red in BGR
    thickness: int = 2,
) -> np.ndarray:
    """
    Render bounding boxes, candidate IDs, anomaly scores, and pixel dimensions
    on the original road image.
    """
    output = image_bgr.copy()

    for cand in candidates:
        x, y = cand["x"], cand["y"]
        w, h = cand["width"], cand["height"]
        cid = cand.get("candidate_id", "defect")
        score = cand.get("avg_anomaly_score", 0.0)

        # Draw primary bounding rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        # Corner brackets for professional CV styling
        corner_len = min(12, w // 4, h // 4)
        if corner_len > 2:
            cv2.line(output, (x, y), (x + corner_len, y), (255, 255, 255), thickness + 1)
            cv2.line(output, (x, y), (x, y + corner_len), (255, 255, 255), thickness + 1)
            cv2.line(output, (x + w, y + h), (x + w - corner_len, y + h), (255, 255, 255), thickness + 1)
            cv2.line(output, (x + w, y + h), (x + w, y + h - corner_len), (255, 255, 255), thickness + 1)

        # Badge text: ID, score, dimensions
        label = f"{cid} | S:{score:.2f} | {w}x{h}"
        font_scale = 0.42
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        # Draw filled text background above the box
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
            font_thick,
            cv2.LINE_AA,
        )

    return output


def localize_single_image(
    image_bgr: np.ndarray,
    image_predictions_df: pd.DataFrame,
    image_filename: str = "",
) -> tuple:
    """
    Execute spatial localization on a single road image.

    Returns
    -------
    tuple
        (annotated_image_bgr, candidates_list, binary_mask, grouped_mask)
    """
    h, w = image_bgr.shape[:2]

    # 1. Continuous spatial density
    density = compute_spatial_anomaly_density((h, w), image_predictions_df)

    # 2. Binary anomaly mask
    raw_mask = create_binary_anomaly_mask((h, w), image_predictions_df)

    # 3. Morphological grouping
    grouped_mask = apply_morphological_grouping(raw_mask)

    # 4. Contour extraction
    contours = extract_candidate_contours(grouped_mask)

    # 5. Bounding box pooling & NMS
    candidates = extract_candidate_regions(
        contours,
        density,
        (h, w),
        image_filename=image_filename,
    )

    # 6. Annotation overlay
    annotated = draw_localized_candidates(image_bgr, candidates)

    return annotated, candidates, raw_mask, grouped_mask


def localize_all_images(
    predictions_csv_path: str = None,
    raw_images_dir: str = None,
    output_dir: str = None,
    candidates_csv_path: str = None,
) -> tuple:
    """
    Run spatial localization across all evaluated images in the dataset.

    Returns
    -------
    tuple
        (all_candidates_df, total_candidates_found)
    """
    if predictions_csv_path is None:
        predictions_csv_path = config.ANOMALY_PREDICTIONS_CSV
    if raw_images_dir is None:
        raw_images_dir = os.path.join(config.OUTPUT_DIR, "original")
        if not os.path.exists(raw_images_dir):
            raw_images_dir = config.INPUT_DIR
    if output_dir is None:
        output_dir = config.LOCALIZED_OUTPUT_DIR
    if candidates_csv_path is None:
        candidates_csv_path = config.LOCALIZED_CANDIDATES_CSV

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(candidates_csv_path)), exist_ok=True)

    df_preds = pd.read_csv(predictions_csv_path)
    unique_images = df_preds["image_filename"].unique()

    print(f"[INFO] localization: Processing defect candidate localization for {len(unique_images)} images...")

    all_candidates = []

    for img_name in unique_images:
        stem, _ = os.path.splitext(img_name)
        img_sub = df_preds[df_preds["image_filename"] == img_name]

        img_path = os.path.join(raw_images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(config.INPUT_DIR, img_name)

        if not os.path.exists(img_path):
            continue

        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            continue

        annotated_bgr, candidates, _, _ = localize_single_image(
            orig_bgr,
            img_sub,
            image_filename=img_name,
        )

        # Save annotated image
        out_img_path = os.path.join(output_dir, f"{stem}_localized.png")
        cv2.imwrite(out_img_path, annotated_bgr)

        all_candidates.extend(candidates)

    candidates_df = pd.DataFrame(all_candidates)
    if not candidates_df.empty:
        candidates_df.to_csv(candidates_csv_path, index=False)
        print(f"[INFO] localization: Saved {len(candidates_df)} candidate regions to {candidates_csv_path}")
    else:
        # Create empty DataFrame with required schema if 0 candidates
        candidates_df = pd.DataFrame(columns=[
            "candidate_id", "image_filename", "x", "y", "width", "height",
            "pixel_area", "box_area", "avg_anomaly_score", "max_anomaly_score",
        ])
        candidates_df.to_csv(candidates_csv_path, index=False)

    return candidates_df, len(all_candidates)

