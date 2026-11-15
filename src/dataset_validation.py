"""
src/dataset_validation.py
==========================
Dataset validation utilities for the Pothole Detection pipeline.

Provides:
  - validate_dataset(directory): Full validation pass over a directory of images.
    Returns a detailed validation report dict and saves it to JSON.
"""

import os
import sys
import json
import collections

# ---------------------------------------------------------------------------
# Ensure project root is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.data_loader import load_dataset, get_image_hash


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dataset(directory: str) -> dict:
    """
    Validate all images inside *directory* and produce a comprehensive report.

    Checks performed
    ----------------
    1. Corrupted / unreadable images  (cv2.imread returns None)
    2. Exact pixel-level duplicates   (MD5 of flattened bytes)
    3. Images smaller than minimum dimensions from config
    4. File-format distribution

    The report dict contains:
        total_files         – int
        successfully_loaded – int
        failed_images       – list of str paths
        duplicate_images    – list of (path1, path2) tuples
        small_images        – list of str paths
        format_counts       – dict mapping extension -> count
        dimension_stats     – dict with min/max/mean for width and height

    The report is also written to ``config.METADATA_DIR/validation_report.json``.

    Parameters
    ----------
    directory : str
        Directory to scan and validate.

    Returns
    -------
    dict
        The validation report.
    """
    records = load_dataset(directory)

    failed_images = []
    small_images = []
    format_counts = collections.Counter()
    widths = []
    heights = []

    # Hash -> first path seen; used for duplicate detection
    hash_map: dict[str, str] = {}
    duplicate_images = []

    for rec in records:
        ext = rec["format"]
        format_counts[ext] += 1

        if rec["image"] is None:
            # Corrupted or unreadable
            failed_images.append(rec["path"])
            continue

        # Dimension validation
        w, h = rec["width"], rec["height"]
        widths.append(w)
        heights.append(h)

        if w < config.MIN_IMAGE_WIDTH or h < config.MIN_IMAGE_HEIGHT:
            small_images.append(rec["path"])

        # Duplicate detection via MD5
        img_hash = get_image_hash(rec["image"])
        if img_hash in hash_map:
            duplicate_images.append((hash_map[img_hash], rec["path"]))
        else:
            hash_map[img_hash] = rec["path"]

    # Dimension statistics (only over successfully loaded images)
    if widths:
        dimension_stats = {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "mean_width": round(sum(widths) / len(widths), 2),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "mean_height": round(sum(heights) / len(heights), 2),
        }
    else:
        dimension_stats = {
            "min_width": None, "max_width": None, "mean_width": None,
            "min_height": None, "max_height": None, "mean_height": None,
        }

    report = {
        "total_files": len(records),
        "successfully_loaded": len(records) - len(failed_images),
        "failed_images": failed_images,
        "duplicate_images": [list(pair) for pair in duplicate_images],
        "small_images": small_images,
        "format_counts": dict(format_counts),
        "dimension_stats": dimension_stats,
    }

    # ------------------------------------------------------------------
    # Persist report to disk
    # ------------------------------------------------------------------
    metadata_dir = os.path.abspath(config.METADATA_DIR)
    os.makedirs(metadata_dir, exist_ok=True)
    report_path = os.path.join(metadata_dir, "validation_report.json")

    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=4)
        print(f"[INFO] validation: Report saved to {report_path}")
    except Exception as exc:
        print(f"[ERROR] validation: Could not save report: {exc}")

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    _print_summary(report)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_summary(report: dict) -> None:
    """Print a human-readable validation summary to stdout."""
    sep = "=" * 60
    print(sep)
    print("  DATASET VALIDATION SUMMARY")
    print(sep)
    print(f"  Total files scanned   : {report['total_files']}")
    print(f"  Successfully loaded   : {report['successfully_loaded']}")
    print(f"  Corrupted / failed    : {len(report['failed_images'])}")
    print(f"  Duplicate image pairs : {len(report['duplicate_images'])}")
    print(f"  Images below min size : {len(report['small_images'])}")

    print("\n  Format distribution:")
    for ext, cnt in sorted(report["format_counts"].items()):
        print(f"    {ext:8s}  {cnt}")

    ds = report["dimension_stats"]
    if ds["min_width"] is not None:
        print("\n  Dimension stats (loaded images):")
        print(f"    Width  — min: {ds['min_width']}  max: {ds['max_width']}  mean: {ds['mean_width']}")
        print(f"    Height — min: {ds['min_height']}  max: {ds['max_height']}  mean: {ds['mean_height']}")

    if report["failed_images"]:
        print("\n  Failed images:")
        for p in report["failed_images"]:
            print(f"    - {p}")

    if report["duplicate_images"]:
        print("\n  Duplicate pairs:")
        for p1, p2 in report["duplicate_images"]:
            print(f"    - {os.path.basename(p1)}  ==  {os.path.basename(p2)}")

    if report["small_images"]:
        print("\n  Small images (below minimum dimensions):")
        for p in report["small_images"]:
            print(f"    - {p}")

    print(sep)
