"""
src/severity_report.py
======================
Severity batch assessment and CSV reporting module.
Evaluates physical defect characteristics, saves severity visualizations,
and exports structured audit tables.
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
from src.severity_estimation import estimate_candidate_severity, draw_severity_annotations


def process_and_report_severity(
    candidates_csv_path: str = None,
    raw_images_dir: str = None,
    output_images_dir: str = None,
    report_csv_path: str = None,
) -> tuple:
    """
    Compute severity metrics for all localized candidates across the dataset,
    render severity overlays, and export the master CSV report.

    Returns
    -------
    tuple
        (severity_df, summary_dict)
    """
    if candidates_csv_path is None:
        candidates_csv_path = config.LOCALIZED_CANDIDATES_CSV
    if raw_images_dir is None:
        raw_images_dir = os.path.join(config.OUTPUT_DIR, "original")
        if not os.path.exists(raw_images_dir):
            raw_images_dir = config.INPUT_DIR
    if output_images_dir is None:
        output_images_dir = config.SEVERITY_OUTPUT_DIR
    if report_csv_path is None:
        report_csv_path = config.SEVERITY_REPORT_CSV

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_csv_path)), exist_ok=True)

    if not os.path.exists(candidates_csv_path):
        raise FileNotFoundError(f"Localized candidates table missing at: {candidates_csv_path}")

    candidates_df = pd.read_csv(candidates_csv_path)
    if candidates_df.empty:
        print("[WARNING] severity_report: Candidate table is empty. Generating blank report.")
        empty_df = pd.DataFrame(columns=[
            "candidate_id", "image_filename", "x", "y", "width", "height",
            "pixel_area", "box_area", "aspect_ratio", "avg_anomaly_score",
            "max_anomaly_score", "texture_irregularity", "edge_density",
            "road_contrast", "severity_score", "severity_category",
        ])
        empty_df.to_csv(report_csv_path, index=False)
        return empty_df, {"total": 0, "low": 0, "medium": 0, "high": 0}

    evaluated_records = []
    unique_images = candidates_df["image_filename"].unique()

    print(f"[INFO] severity_report: Evaluating severity across {len(candidates_df)} candidates in {len(unique_images)} images...")

    for img_name in unique_images:
        stem, _ = os.path.splitext(img_name)
        img_candidates = candidates_df[candidates_df["image_filename"] == img_name].to_dict(orient="records")

        img_path = os.path.join(raw_images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(config.INPUT_DIR, img_name)

        if not os.path.exists(img_path):
            continue

        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            continue
        gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

        img_evaluated = []
        for cand in img_candidates:
            ev = estimate_candidate_severity(cand, gray)
            img_evaluated.append(ev)
            evaluated_records.append(ev)

        # Draw and save annotated severity image
        annotated_bgr = draw_severity_annotations(orig_bgr, img_evaluated)
        out_path = os.path.join(output_images_dir, f"{stem}_severity.png")
        cv2.imwrite(out_path, annotated_bgr)

    report_df = pd.DataFrame(evaluated_records)
    report_df.to_csv(report_csv_path, index=False)
    print(f"[INFO] severity_report: Saved full severity report ({len(report_df)} rows) to: {report_csv_path}")

    # Summary metrics
    n_total = len(report_df)
    n_low = int((report_df["severity_category"] == "Low").sum())
    n_med = int((report_df["severity_category"] == "Medium").sum())
    n_high = int((report_df["severity_category"] == "High").sum())

    summary = {
        "total_candidates": n_total,
        "low_severity": n_low,
        "medium_severity": n_med,
        "high_severity": n_high,
    }

    return report_df, summary

