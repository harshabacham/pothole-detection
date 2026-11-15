"""
src/visualization.py
====================
Visualization helpers for the Pothole Detection pipeline.

Provides:
  - plot_preprocessing_comparison(original_bgr, processed_dict, title, save_path)
  - plot_dimension_distribution(image_records, save_path)
  - plot_sample_gallery(image_records, n, save_path)
  - generate_dataset_report_figure(validation_report, save_path)
"""

import os
import sys
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Ensure project root is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path):
    """Save the figure to *save_path* (if given) or display it interactively."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[INFO] visualization: Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_preprocessing_comparison(
    original_bgr: np.ndarray,
    processed_dict: dict,
    title: str = "Preprocessing Pipeline",
    save_path: str = None,
) -> None:
    """
    Display a 1×4 side-by-side comparison of every preprocessing stage.

    Columns: Original | Grayscale | Blurred | Enhanced

    Parameters
    ----------
    original_bgr : numpy.ndarray
        The original BGR image.
    processed_dict : dict
        Dict returned by ``preprocess_image`` with keys:
        original_bgr, gray, blurred, enhanced.
    title : str
        Super-title for the figure.
    save_path : str or None
        File path to save the figure; displays interactively if None.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Original — convert BGR → RGB for correct matplotlib rendering
    orig_rgb = cv2.cvtColor(processed_dict["original_bgr"], cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(processed_dict["gray"], cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(processed_dict["blurred"], cmap="gray")
    axes[2].set_title("Blurred")
    axes[2].axis("off")

    axes[3].imshow(processed_dict["enhanced"], cmap="gray")
    axes[3].set_title("Enhanced (CLAHE)")
    axes[3].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_dimension_distribution(
    image_records: list,
    save_path: str = None,
) -> None:
    """
    Plot histograms of image widths and heights side-by-side.

    Parameters
    ----------
    image_records : list[dict]
        Records as returned by ``load_dataset``; only successfully loaded
        images (those with a non-None ``image`` field) are used.
    save_path : str or None
        File path to save the figure; displays interactively if None.
    """
    widths  = [r["width"]  for r in image_records if r["image"] is not None]
    heights = [r["height"] for r in image_records if r["image"] is not None]

    if not widths:
        print("[WARNING] visualization: No valid images to plot dimension distribution.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Image Dimension Distribution", fontsize=14, fontweight="bold")

    ax1.hist(widths, bins=20, color="steelblue", edgecolor="white")
    ax1.set_title("Width Distribution")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Count")

    ax2.hist(heights, bins=20, color="darkorange", edgecolor="white")
    ax2.set_title("Height Distribution")
    ax2.set_xlabel("Height (px)")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_sample_gallery(
    image_records: list,
    n: int = 9,
    save_path: str = None,
) -> None:
    """
    Display up to *n* sample images from *image_records* in a square(-ish) grid.

    Parameters
    ----------
    image_records : list[dict]
        Records as returned by ``load_dataset``.
    n : int
        Maximum number of images to show.
    save_path : str or None
        File path to save the figure; displays interactively if None.
    """
    valid = [r for r in image_records if r["image"] is not None]
    samples = valid[:n]

    if not samples:
        print("[WARNING] visualization: No valid images to display in gallery.")
        return

    cols = math.ceil(math.sqrt(len(samples)))
    rows = math.ceil(len(samples) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    fig.suptitle(f"Sample Gallery ({len(samples)} images)", fontsize=14, fontweight="bold")

    # Flatten axes for uniform indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, rec in enumerate(samples):
        r, c = divmod(idx, cols)
        img_rgb = cv2.cvtColor(rec["image"], cv2.COLOR_BGR2RGB)
        axes[r][c].imshow(img_rgb)
        axes[r][c].set_title(rec["filename"], fontsize=8, wrap=True)
        axes[r][c].axis("off")

    # Hide any unused subplot cells
    for idx in range(len(samples), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def generate_dataset_report_figure(
    validation_report: dict,
    save_path: str = None,
) -> None:
    """
    Generate a bar chart summarising the key metrics from *validation_report*.

    Bars shown:
        Loaded | Failed | Duplicates | Small

    Parameters
    ----------
    validation_report : dict
        Report dict returned by ``validate_dataset``.
    save_path : str or None
        File path to save the figure; displays interactively if None.
    """
    labels = ["Loaded", "Failed", "Duplicates", "Small"]
    values = [
        validation_report.get("successfully_loaded", 0),
        len(validation_report.get("failed_images", [])),
        len(validation_report.get("duplicate_images", [])),
        len(validation_report.get("small_images", [])),
    ]
    colors = ["#4CAF50", "#F44336", "#FF9800", "#2196F3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Dataset Validation Report", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) + max(1, max(values) // 4))

    # Annotate each bar with its value
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    _save_or_show(fig, save_path)

