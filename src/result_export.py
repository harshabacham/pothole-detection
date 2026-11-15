"""
src/result_export.py
====================
Batch export and multi-panel comparison generator for Phase 6 visualizations.
Saves overlays, heatmaps, and 4-quadrant diagnostic comparison figures.
"""

import os
import sys
import cv2
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
from src.anomaly_visualization import draw_patch_overlay
from src.heatmap import generate_anomaly_heatmap


def create_comparison_quad(
    original_bgr: np.ndarray,
    processed_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    image_title: str = "Road Anomaly Analysis",
    save_path: str = None,
) -> None:
    """
    Generate a 2x2 four-panel comparison figure:
      - Panel 1: Original Road Image
      - Panel 2: Preprocessed Surface (Grayscale + CLAHE)
      - Panel 3: Color-Coded Anomaly Patch Overlay
      - Panel 4: Spatial Density Anomaly Heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Convert BGR to RGB for Matplotlib rendering
    rgb_orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    rgb_proc = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB) if processed_bgr.ndim == 3 else cv2.cvtColor(processed_bgr, cv2.COLOR_GRAY2RGB)
    rgb_over = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    rgb_heat = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    axes[0, 0].imshow(rgb_orig)
    axes[0, 0].set_title("1. Original Road Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_proc)
    axes[0, 1].set_title("2. Preprocessed Road Surface (Noise Reduced & CLAHE)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb_over)
    axes[1, 0].set_title("3. Color-Coded Patch Anomaly Overlay (Green/Yellow/Red)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb_heat)
    axes[1, 1].set_title("4. Spatial Anomaly Density Heatmap", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle(f"Audit Comparison: {image_title}", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def export_all_visualizations(
    predictions_csv_path: str = None,
    raw_images_dir: str = None,
    processed_dir: str = None,
) -> dict:
    """
    Iterate across all images present in predictions, generating and saving
    overlays, heatmaps, and quad comparisons.

    Returns
    -------
    dict
        Export counts and file locations.
    """
    if predictions_csv_path is None:
        predictions_csv_path = config.ANOMALY_PREDICTIONS_CSV
    if raw_images_dir is None:
        raw_images_dir = os.path.join(config.OUTPUT_DIR, "original")
        if not os.path.exists(raw_images_dir):
            raw_images_dir = config.INPUT_DIR
    if processed_dir is None:
        processed_dir = os.path.join(config.OUTPUT_DIR, "enhanced")

    os.makedirs(config.OVERLAYS_DIR, exist_ok=True)
    os.makedirs(config.HEATMAPS_DIR, exist_ok=True)
    os.makedirs(config.COMPARISONS_DIR, exist_ok=True)

    df_preds = pd.read_csv(predictions_csv_path)
    unique_images = df_preds["image_filename"].unique()

    print(f"[INFO] result_export: Exporting visualizations for {len(unique_images)} images...")
    exported_counts = {"overlays": 0, "heatmaps": 0, "comparisons": 0}

    for img_name in unique_images:
        stem, _ = os.path.splitext(img_name)
        img_sub = df_preds[df_preds["image_filename"] == img_name]

        # 1. Load Original Image
        img_path = os.path.join(raw_images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(config.INPUT_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] result_export: Image not found: {img_path}. Skipping.")
            continue

        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            continue

        # 2. Generate Overlay
        overlay_bgr = draw_patch_overlay(orig_bgr, img_sub, show_scores=True, show_normal=True)
        overlay_out = os.path.join(config.OVERLAYS_DIR, f"{stem}_overlay.png")
        cv2.imwrite(overlay_out, overlay_bgr)
        exported_counts["overlays"] += 1

        # 3. Generate Heatmap
        heatmap_blend, density, _ = generate_anomaly_heatmap(orig_bgr, img_sub)
        heatmap_out = os.path.join(config.HEATMAPS_DIR, f"{stem}_heatmap.png")
        cv2.imwrite(heatmap_out, heatmap_blend)
        exported_counts["heatmaps"] += 1

        # 4. Load Processed Image
        proc_path = os.path.join(processed_dir, f"{stem}.png")
        if os.path.exists(proc_path):
            proc_img = cv2.imread(proc_path)
        else:
            proc_img = cv2.cvtColor(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        # 5. Generate 4-Quadrant Comparison
        comp_out = os.path.join(config.COMPARISONS_DIR, f"{stem}_comparison.png")
        create_comparison_quad(
            orig_bgr,
            proc_img,
            overlay_bgr,
            heatmap_blend,
            image_title=img_name,
            save_path=comp_out,
        )
        exported_counts["comparisons"] += 1

    print(f"[INFO] result_export: Done. Exported {exported_counts['overlays']} overlays, {exported_counts['heatmaps']} heatmaps, and {exported_counts['comparisons']} comparison quads.")
    return exported_counts

