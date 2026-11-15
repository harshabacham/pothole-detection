"""
src/patch_extraction.py
=======================
Patch extraction utilities for the Pothole Detection pipeline.

Provides functions to compute a patch grid over a road ROI, extract individual
patches (both grayscale and colour), validate patches, save them to disk, and
run the full extraction pipeline over a directory of images.

Provides:
  - compute_grid(roi_width, roi_height, patch_size, overlap)
  - extract_patches(image_record, patch_size, overlap)
  - save_patches(patch_list, output_base_dir)
  - extract_and_save_all(input_dir, output_dir, patch_size, overlap)
"""

import os
import sys
import math

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config can always be imported.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.data_loader import load_dataset
from src.preprocessing import preprocess_image
from src.roi_detection import get_roi_bounds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_grid(
    roi_width: int,
    roi_height: int,
    patch_size: int,
    overlap: bool = False,
) -> list:
    """
    Compute a list of (col_offset, row_offset) top-left corners for patches
    within the ROI.

    When overlap=True, stride = int(patch_size * config.PATCH_STRIDE_FRAC).
    When overlap=False, stride = patch_size (non-overlapping).

    Only positions where the full patch_size x patch_size window fits within
    roi_width x roi_height are included — no padding or truncation of partial
    patches.

    Parameters
    ----------
    roi_width  : int  Width of the ROI in pixels.
    roi_height : int  Height of the ROI in pixels.
    patch_size : int  Side length of the square patch in pixels.
    overlap    : bool Whether to use overlapping patches.

    Returns
    -------
    list of (col_offset, row_offset) tuples, each int.
    """
    if overlap:
        stride = max(1, int(patch_size * config.PATCH_STRIDE_FRAC))
    else:
        stride = patch_size

    positions = []
    row_offset = 0
    while row_offset + patch_size <= roi_height:
        col_offset = 0
        while col_offset + patch_size <= roi_width:
            positions.append((col_offset, row_offset))
            col_offset += stride
        row_offset += stride

    return positions


def extract_patches(
    image_record: dict,
    patch_size: int = None,
    overlap: bool = None,
) -> list:
    """
    Extract patches from the ROI of a single image.

    Parameters
    ----------
    image_record : dict
        As returned by data_loader.load_dataset — must have 'image' (BGR ndarray),
        'filename', and 'path'.
    patch_size : int, optional
        Patch size in pixels. Defaults to config.DEFAULT_PATCH_SIZE.
    overlap : bool, optional
        Whether to use overlapping patches. Defaults to config.PATCH_OVERLAP.

    Returns
    -------
    list of patch dicts, each containing:
        patch_id        : str   "{stem}_ps{size}_r{row:03d}_c{col:03d}"
        image_filename  : str   source image filename
        image_path      : str   source image absolute path
        patch_size      : int
        row_idx         : int   row index in grid (0-based)
        col_idx         : int   col index in grid (0-based)
        x1, y1         : int   top-left of patch in original image coords
        x2, y2         : int   bottom-right of patch in original image coords
        roi_x1, roi_y1 : int   top-left of ROI in original image coords
        roi_x2, roi_y2 : int   bottom-right of ROI in original image coords
        patch_gray      : np.ndarray  grayscale patch (patch_size x patch_size)
        patch_bgr       : np.ndarray  BGR patch (patch_size x patch_size x 3)
        is_valid        : bool  True if >= MIN_PATCH_VALID_FRAC pixels are non-zero

    An empty list is returned when the image cannot be processed.
    """
    if patch_size is None:
        patch_size = config.DEFAULT_PATCH_SIZE
    if overlap is None:
        overlap = config.PATCH_OVERLAP

    image_bgr = image_record.get("image")
    filename  = image_record.get("filename", "unknown")
    path      = image_record.get("path", "")

    if image_bgr is None:
        print(f"[WARNING] patch_extraction: Skipping unreadable image: {filename}")
        return []

    img_h, img_w = image_bgr.shape[:2]

    # Check minimum size
    if img_w < patch_size or img_h < patch_size:
        print(
            f"[WARNING] patch_extraction: Image '{filename}' ({img_w}x{img_h}) is "
            f"too small for patch_size={patch_size}. Skipping."
        )
        return []

    # Compute ROI
    try:
        roi = get_roi_bounds(image_bgr)
    except ValueError as exc:
        print(f"[WARNING] patch_extraction: Could not compute ROI for '{filename}': {exc}")
        return []

    # Check that ROI is large enough for at least one patch
    if roi["width"] < patch_size or roi["height"] < patch_size:
        print(
            f"[WARNING] patch_extraction: ROI ({roi['width']}x{roi['height']}) for "
            f"'{filename}' is too small for patch_size={patch_size}. Skipping."
        )
        return []

    # Preprocess to get grayscale version
    try:
        preprocessed = preprocess_image(image_bgr, use_clahe=config.USE_CLAHE)
        gray_full = preprocessed["gray"]
    except Exception as exc:
        print(f"[WARNING] patch_extraction: Preprocessing failed for '{filename}': {exc}")
        return []

    # Compute grid positions (relative to ROI)
    grid_positions = compute_grid(roi["width"], roi["height"], patch_size, overlap)

    if not grid_positions:
        print(
            f"[WARNING] patch_extraction: No valid patch positions in ROI for "
            f"'{filename}' with patch_size={patch_size}. Skipping."
        )
        return []

    stem = os.path.splitext(filename)[0]
    patches = []

    # Track unique (row_idx, col_idx) for grid indexing
    col_positions = sorted(set(pos[0] for pos in grid_positions))
    row_positions = sorted(set(pos[1] for pos in grid_positions))
    col_idx_map = {c: i for i, c in enumerate(col_positions)}
    row_idx_map = {r: i for i, r in enumerate(row_positions)}

    for col_off, row_off in grid_positions:
        # Convert ROI-local offsets to full-image coordinates
        abs_x1 = roi["x1"] + col_off
        abs_y1 = roi["y1"] + row_off
        abs_x2 = abs_x1 + patch_size
        abs_y2 = abs_y1 + patch_size

        # Safety: skip if outside image bounds (should not happen given grid logic)
        if abs_x2 > img_w or abs_y2 > img_h:
            continue

        row_idx = row_idx_map[row_off]
        col_idx = col_idx_map[col_off]

        patch_id = f"{stem}_ps{patch_size}_r{row_idx:03d}_c{col_idx:03d}"

        patch_gray = gray_full[abs_y1:abs_y2, abs_x1:abs_x2].copy()
        patch_bgr  = image_bgr[abs_y1:abs_y2, abs_x1:abs_x2].copy()

        # Validity check: fraction of non-zero pixels in grayscale patch
        total_pixels = patch_size * patch_size
        nonzero_pixels = int(np.count_nonzero(patch_gray))
        is_valid = (nonzero_pixels / total_pixels) >= config.MIN_PATCH_VALID_FRAC

        patches.append({
            "patch_id":       patch_id,
            "image_filename": filename,
            "image_path":     path,
            "patch_size":     patch_size,
            "row_idx":        row_idx,
            "col_idx":        col_idx,
            "x1":             abs_x1,
            "y1":             abs_y1,
            "x2":             abs_x2,
            "y2":             abs_y2,
            "roi_x1":         roi["x1"],
            "roi_y1":         roi["y1"],
            "roi_x2":         roi["x2"],
            "roi_y2":         roi["y2"],
            "patch_gray":     patch_gray,
            "patch_bgr":      patch_bgr,
            "is_valid":       is_valid,
        })

    return patches


def save_patches(patch_list: list, output_base_dir: str) -> list:
    """
    Save extracted patches to disk.

    Directory layout:
        output_base_dir/
            {image_stem}/
                gray/   — grayscale patches saved as PNG
                color/  — BGR patches saved as PNG

    Filename: {patch_id}.png

    Parameters
    ----------
    patch_list       : list  Patch dicts as returned by extract_patches().
    output_base_dir  : str   Root directory for patch output.

    Returns
    -------
    The same patch_list with two new keys added to each dict:
        gray_path  : str  Absolute path where the gray patch was saved.
        color_path : str  Absolute path where the colour patch was saved.
    """
    if not patch_list:
        return patch_list

    # Pre-create output directories (grouped by image stem)
    stems_seen = set()
    for patch in patch_list:
        stem = os.path.splitext(patch["image_filename"])[0]
        if stem not in stems_seen:
            os.makedirs(os.path.join(output_base_dir, stem, "gray"),  exist_ok=True)
            os.makedirs(os.path.join(output_base_dir, stem, "color"), exist_ok=True)
            stems_seen.add(stem)

    for patch in patch_list:
        stem       = os.path.splitext(patch["image_filename"])[0]
        patch_id   = patch["patch_id"]

        gray_path  = os.path.abspath(
            os.path.join(output_base_dir, stem, "gray",  patch_id + ".png")
        )
        color_path = os.path.abspath(
            os.path.join(output_base_dir, stem, "color", patch_id + ".png")
        )

        cv2.imwrite(gray_path,  patch["patch_gray"])
        cv2.imwrite(color_path, patch["patch_bgr"])

        patch["gray_path"]  = gray_path
        patch["color_path"] = color_path

    return patch_list


def extract_and_save_all(
    input_dir: str,
    output_dir: str,
    patch_size: int = None,
    overlap: bool = None,
) -> tuple:
    """
    Full pipeline: load all images from input_dir, extract patches, save them,
    and return (all_patches_list, metadata_df).

    Parameters
    ----------
    input_dir  : str   Directory containing input images.
    output_dir : str   Root directory for saved patches.
    patch_size : int   Patch size in pixels. Defaults to config.DEFAULT_PATCH_SIZE.
    overlap    : bool  Whether to use overlapping patches. Defaults to config.PATCH_OVERLAP.

    Returns
    -------
    (all_patches : list, metadata_df : pandas.DataFrame)

    metadata_df columns:
        patch_id, image_filename, image_path, patch_size,
        row_idx, col_idx, x1, y1, x2, y2,
        roi_x1, roi_y1, roi_x2, roi_y2,
        gray_path, color_path, is_valid

    (ndarray columns patch_gray and patch_bgr are excluded from the DataFrame.)
    """
    if patch_size is None:
        patch_size = config.DEFAULT_PATCH_SIZE
    if overlap is None:
        overlap = config.PATCH_OVERLAP

    records = load_dataset(input_dir)
    if not records:
        print(f"[WARNING] patch_extraction: No images found in '{input_dir}'.")
        return [], pd.DataFrame()

    all_patches = []
    for record in records:
        if record["image"] is None:
            continue
        patches = extract_patches(record, patch_size=patch_size, overlap=overlap)
        all_patches.extend(patches)

    if not all_patches:
        print(f"[WARNING] patch_extraction: No patches extracted from '{input_dir}'.")
        return [], pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    all_patches = save_patches(all_patches, output_dir)

    # Build metadata DataFrame (drop ndarray columns)
    _ARRAY_COLS = {"patch_gray", "patch_bgr"}
    df_records = []
    for patch in all_patches:
        row = {k: v for k, v in patch.items() if k not in _ARRAY_COLS}
        df_records.append(row)

    metadata_df = pd.DataFrame(df_records)

    return all_patches, metadata_df

