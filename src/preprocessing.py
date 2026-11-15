"""
src/preprocessing.py
====================
Image preprocessing utilities for the Pothole Detection pipeline.

Provides:
  - preprocess_image(image_bgr, use_clahe): Returns a dict of processed versions.
  - preprocess_and_save(input_dir, output_dir): Batch-processes all valid images
    and saves each stage to disk.
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.data_loader import load_dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(image_bgr: np.ndarray, use_clahe: bool = True) -> dict:
    """
    Apply a standard preprocessing pipeline to a single BGR image.

    Pipeline stages
    ---------------
    1. original_bgr – untouched copy of the input
    2. gray          – RGB-to-grayscale conversion
    3. blurred       – Gaussian blur applied to the grayscale image
    4. enhanced      – CLAHE (or simple histogram equalisation) on the blurred image

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Input image in BGR colour space (as returned by cv2.imread).
    use_clahe : bool
        If True, apply CLAHE for contrast enhancement; otherwise use
        cv2.equalizeHist for a global histogram equalisation.

    Returns
    -------
    dict with keys:
        original_bgr, gray, blurred, enhanced
    """
    if image_bgr is None:
        raise ValueError("preprocess_image received a None image.")

    # Stage 1 — preserve original
    original_bgr = image_bgr.copy()

    # Stage 2 — grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Stage 3 — Gaussian blur (noise reduction)
    blurred = cv2.GaussianBlur(
        gray,
        config.BLUR_KERNEL_SIZE,
        config.BLUR_SIGMA,
    )

    # Stage 4 — contrast enhancement
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )
        enhanced = clahe.apply(blurred)
    else:
        enhanced = cv2.equalizeHist(blurred)

    return {
        "original_bgr": original_bgr,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
    }


def preprocess_and_save(input_dir: str, output_dir: str) -> list:
    """
    Load all valid images from *input_dir*, preprocess each one, and save
    every pipeline stage to a corresponding sub-directory inside *output_dir*.

    Output layout
    -------------
    output_dir/
        original/   ← colour BGRs re-saved (lossless quality)
        gray/       ← grayscale PNGs
        blurred/    ← blurred grayscale PNGs
        enhanced/   ← CLAHE-enhanced grayscale PNGs

    Parameters
    ----------
    input_dir : str
        Directory containing raw images.
    output_dir : str
        Root directory for all processed output.

    Returns
    -------
    list[dict]
        One result dict per image with keys:
            filename, status, width, height
        ``status`` is "ok" on success, "skipped" when the image could not be
        loaded, or "error: <message>" if an unexpected exception occurred.
    """
    # Create output sub-directories
    stages = ["original", "gray", "blurred", "enhanced"]
    stage_dirs = {}
    for stage in stages:
        stage_path = os.path.join(output_dir, stage)
        os.makedirs(stage_path, exist_ok=True)
        stage_dirs[stage] = stage_path

    records = load_dataset(input_dir)
    results = []

    for rec in records:
        filename = rec["filename"]
        result = {"filename": filename, "status": None, "width": None, "height": None}

        if rec["image"] is None:
            print(f"[WARNING] preprocessing: Skipping unreadable image: {filename}")
            result["status"] = "skipped"
            results.append(result)
            continue

        try:
            processed = preprocess_image(rec["image"], use_clahe=config.USE_CLAHE)

            # Derive a base name without extension for lossless stages
            base, _ = os.path.splitext(filename)

            # Save original (keep original extension to preserve format)
            cv2.imwrite(os.path.join(stage_dirs["original"], filename), processed["original_bgr"])

            # Save grayscale / blurred / enhanced as PNG for lossless storage
            cv2.imwrite(os.path.join(stage_dirs["gray"],     base + ".png"), processed["gray"])
            cv2.imwrite(os.path.join(stage_dirs["blurred"],  base + ".png"), processed["blurred"])
            cv2.imwrite(os.path.join(stage_dirs["enhanced"], base + ".png"), processed["enhanced"])

            result["status"] = "ok"
            result["width"] = rec["width"]
            result["height"] = rec["height"]
            print(f"[INFO] preprocessing: Processed '{filename}' ({rec['width']}×{rec['height']})")

        except Exception as exc:
            print(f"[ERROR] preprocessing: Unexpected error for '{filename}': {exc}")
            result["status"] = f"error: {exc}"

        results.append(result)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"[INFO] preprocessing: Done. {ok_count}/{len(results)} image(s) processed successfully.")
    return results

