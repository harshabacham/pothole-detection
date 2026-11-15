"""
src/patch_visualization.py
==========================
Visualization utilities for ROI and patch extraction in the Pothole Detection pipeline.

Provides functions to visualize the ROI bounding box, the patch grid drawn on the
original image, patch galleries, and valid-vs-invalid patch statistics.

Provides:
  - visualize_roi_and_grid(image_bgr, roi, patch_list, patch_size, title, save_path)
  - visualize_patch_gallery(patch_list, n, use_color, title, save_path)
  - visualize_valid_vs_invalid(patch_list, save_path)

NOTE: Uses the non-interactive Agg backend so that no display server is required.
"""

import os
import sys
import math

import cv2
import numpy as np

# Use non-interactive backend BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config can always be imported.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR ndarray to RGB for matplotlib display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_roi_and_grid(
    image_bgr: np.ndarray,
    roi: dict,
    patch_list: list,
    patch_size: int,
    title: str = "",
    save_path: str = None,
) -> None:
    """
    Create a 1x3 matplotlib figure:
        Col 1: Original image
        Col 2: Image with ROI rectangle overlay (green)
        Col 3: Image with ROI rectangle AND patch grid (each patch as a blue rectangle)

    Parameters
    ----------
    image_bgr  : np.ndarray  Source BGR image.
    roi        : dict         ROI bounds dict from roi_detection.get_roi_bounds().
    patch_list : list         Patch dicts from patch_extraction.extract_patches().
    patch_size : int          Patch size used during extraction.
    title      : str          Optional super-title for the figure.
    save_path  : str, optional
        Full path (including filename) to save the figure.
        If None, the figure is shown interactively.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Col 1: Original image ---
    axes[0].imshow(_bgr_to_rgb(image_bgr))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # --- Col 2: ROI overlay ---
    roi_canvas = image_bgr.copy()
    cv2.rectangle(
        roi_canvas,
        (roi["x1"], roi["y1"]),
        (roi["x2"], roi["y2"]),
        (0, 255, 0),   # green in BGR
        2,
    )
    axes[1].imshow(_bgr_to_rgb(roi_canvas))
    axes[1].set_title("ROI Overlay")
    axes[1].axis("off")

    # --- Col 3: ROI + patch grid ---
    grid_canvas = roi_canvas.copy()
    for patch in patch_list:
        cv2.rectangle(
            grid_canvas,
            (patch["x1"], patch["y1"]),
            (patch["x2"], patch["y2"]),
            (255, 0, 0),   # blue in BGR
            1,
        )
    axes[2].imshow(_bgr_to_rgb(grid_canvas))
    axes[2].set_title(f"Patch Grid (ps={patch_size}, n={len(patch_list)})")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"[INFO] patch_visualization: Saved ROI+grid figure to '{save_path}'")
    else:
        plt.show()

    plt.close("all")


def visualize_patch_gallery(
    patch_list: list,
    n: int = 16,
    use_color: bool = False,
    title: str = "Patch Gallery",
    save_path: str = None,
) -> None:
    """
    Display up to n patches in a grid layout.

    Shows grayscale patches by default; colour patches if use_color=True.
    Each patch is annotated with a short version of its patch_id.

    Parameters
    ----------
    patch_list : list         Patch dicts from patch_extraction.extract_patches().
    n          : int          Maximum number of patches to display. Default 16.
    use_color  : bool         Show BGR colour patches when True.
    title      : str          Figure title.
    save_path  : str, optional  Path to save the figure; shows if None.
    """
    if not patch_list:
        print("[WARNING] patch_visualization: patch_list is empty; skipping gallery.")
        return

    display_patches = patch_list[:n]
    count = len(display_patches)

    # Determine grid layout (aim for roughly square)
    cols = min(count, 4)
    rows = math.ceil(count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Normalise axes to a flat list
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = list(axes.flatten())
    else:
        axes_flat = list(axes.flatten())

    for idx, patch in enumerate(display_patches):
        ax = axes_flat[idx]
        if use_color:
            ax.imshow(_bgr_to_rgb(patch["patch_bgr"]))
        else:
            ax.imshow(patch["patch_gray"], cmap="gray", vmin=0, vmax=255)

        # Short label: last two underscore-separated tokens of patch_id
        parts = patch["patch_id"].split("_")
        short_id = "_".join(parts[-2:]) if len(parts) >= 2 else patch["patch_id"]
        valid_str = "V" if patch["is_valid"] else "X"
        ax.set_title(f"{short_id} [{valid_str}]", fontsize=7)
        ax.axis("off")

    # Hide any unused axes
    for idx in range(count, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"[INFO] patch_visualization: Saved patch gallery to '{save_path}'")
    else:
        plt.show()

    plt.close("all")


def visualize_valid_vs_invalid(
    patch_list: list,
    save_path: str = None,
) -> None:
    """
    Side-by-side bar chart of valid vs invalid patch counts per source image.

    Parameters
    ----------
    patch_list : list         Patch dicts from patch_extraction.extract_patches().
    save_path  : str, optional  Path to save the figure; shows if None.
    """
    if not patch_list:
        print("[WARNING] patch_visualization: patch_list is empty; skipping validity chart.")
        return

    # Aggregate counts per image filename
    image_names = []
    seen_order = []
    counts = {}     # filename -> {"valid": int, "invalid": int}
    for patch in patch_list:
        fname = patch["image_filename"]
        if fname not in counts:
            counts[fname] = {"valid": 0, "invalid": 0}
            seen_order.append(fname)
        if patch["is_valid"]:
            counts[fname]["valid"] += 1
        else:
            counts[fname]["invalid"] += 1

    labels  = seen_order
    valids  = [counts[f]["valid"]   for f in labels]
    invalids= [counts[f]["invalid"] for f in labels]

    x = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))

    bars_v = ax.bar(x - bar_width / 2, valids,   bar_width, label="Valid",   color="#4CAF50")
    bars_i = ax.bar(x + bar_width / 2, invalids, bar_width, label="Invalid", color="#F44336")

    ax.set_xlabel("Image")
    ax.set_ylabel("Patch Count")
    ax.set_title("Valid vs Invalid Patches per Image")
    ax.set_xticks(x)
    # Shorten labels to stem only for readability
    short_labels = [os.path.splitext(f)[0] for f in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()

    # Annotate bar heights
    for bar in bars_v:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, h + 0.1,
                str(int(h)), ha="center", va="bottom", fontsize=7,
            )
    for bar in bars_i:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, h + 0.1,
                str(int(h)), ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"[INFO] patch_visualization: Saved validity chart to '{save_path}'")
    else:
        plt.show()

    plt.close("all")

