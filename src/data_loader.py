"""
src/data_loader.py
==================
Data loading utilities for the Pothole Detection pipeline.

Provides:
  - load_dataset(directory): Scans a directory for images and loads them.
  - get_image_hash(image_array): Returns an MD5 hash of an image array for duplicate detection.
"""

import os
import sys
import hashlib
import cv2

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so config can always be imported.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_image_hash(image_array) -> str:
    """
    Compute an MD5 hash of the flattened byte content of a NumPy image array.

    Parameters
    ----------
    image_array : numpy.ndarray
        A BGR (or grayscale) image array as returned by cv2.imread.

    Returns
    -------
    str
        Hexadecimal MD5 digest string.
    """
    raw_bytes = image_array.tobytes()
    return hashlib.md5(raw_bytes).hexdigest()


def load_dataset(directory: str) -> list:
    """
    Scan *directory* for images with extensions in config.SUPPORTED_EXTENSIONS,
    load each one with OpenCV, and return a list of image records.

    Each record is a dict with keys:
        path       – absolute file path (str)
        filename   – basename of the file (str)
        image      – BGR ndarray (numpy.ndarray) or None if loading failed
        width      – pixel width  (int) or None
        height     – pixel height (int) or None
        channels   – number of channels (int) or None
        format     – file extension in lower-case, e.g. ".jpg" (str)

    Parameters
    ----------
    directory : str
        Path to the directory to scan. May be relative to cwd.

    Returns
    -------
    list[dict]
        List of image records (one per discovered file, even if unreadable).
    """
    records = []

    # Resolve and validate the directory
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        print(f"[WARNING] data_loader: Directory not found: {abs_dir}")
        return records

    print(f"[INFO] data_loader: Scanning directory: {abs_dir}")

    # Collect files with supported extensions
    candidates = []
    for entry in os.listdir(abs_dir):
        ext = os.path.splitext(entry)[1].lower()
        if ext in config.SUPPORTED_EXTENSIONS:
            candidates.append(entry)

    candidates.sort()  # Deterministic ordering
    print(f"[INFO] data_loader: Found {len(candidates)} candidate file(s).")

    for filename in candidates:
        abs_path = os.path.join(abs_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        record = {
            "path": abs_path,
            "filename": filename,
            "image": None,
            "width": None,
            "height": None,
            "channels": None,
            "format": ext,
        }

        try:
            img = cv2.imread(abs_path)
            if img is None:
                print(f"[WARNING] data_loader: cv2.imread returned None for: {abs_path}")
            else:
                h, w = img.shape[:2]
                channels = 1 if img.ndim == 2 else img.shape[2]
                record["image"] = img
                record["width"] = w
                record["height"] = h
                record["channels"] = channels
        except Exception as exc:
            print(f"[WARNING] data_loader: Exception reading {abs_path}: {exc}")

        records.append(record)

    loaded = sum(1 for r in records if r["image"] is not None)
    print(f"[INFO] data_loader: Successfully loaded {loaded}/{len(records)} image(s).")
    return records

