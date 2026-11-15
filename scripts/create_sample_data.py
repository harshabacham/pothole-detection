"""
scripts/create_sample_data.py
==============================
Generates synthetic road-surface images for testing the Pothole Detection
pipeline.  All images are created with NumPy and OpenCV — no additional
libraries required.

Image catalogue produced in data/raw/
--------------------------------------
road_normal_001.jpg … road_normal_008.jpg  – uniform-grey road with noise
road_damaged_001.jpg … road_damaged_004.jpg – road with darker pothole blotches
corrupted.jpg                               – invalid file (non-image bytes)
tiny.jpg                                    – 20×20 valid image (below min dims)
road_normal_001_dup.jpg                     – pixel-exact duplicate of normal_001

Three images are also copied to data/test/ for later evaluation.
"""

import os
import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config is importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normal_road(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """
    Create a synthetic 'normal' road surface: uniform grey + slight Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    base_intensity = rng.integers(100, 160)          # mid-range grey
    noise = rng.normal(0, 8, (height, width)).astype(np.float32)
    channel = np.clip(base_intensity + noise, 0, 255).astype(np.uint8)
    # Stack into BGR (all channels identical → looks grey)
    return cv2.merge([channel, channel, channel])


def _make_damaged_road(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """
    Create a synthetic 'damaged' road surface: normal road with darker blotch
    regions simulating potholes or cracks.
    """
    rng = np.random.default_rng(seed)
    img = _make_normal_road(width, height, seed=seed)

    num_blotches = rng.integers(2, 6)
    for _ in range(num_blotches):
        bx = int(rng.integers(50, width  - 100))
        by = int(rng.integers(50, height - 100))
        bw = int(rng.integers(40, 140))
        bh = int(rng.integers(30, 100))
        # Clamp to image bounds
        x2 = min(bx + bw, width)
        y2 = min(by + bh, height)
        # Darken region to simulate a depression / pothole
        darkening = rng.integers(50, 100)
        roi = img[by:y2, bx:x2].astype(np.int16)
        roi = np.clip(roi - darkening, 0, 255).astype(np.uint8)
        img[by:y2, bx:x2] = roi

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_dir  = os.path.join(_PROJECT_ROOT, config.INPUT_DIR)
    test_dir = os.path.join(_PROJECT_ROOT, config.TEST_DIR)

    os.makedirs(raw_dir,  exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("[INFO] create_sample_data: Generating synthetic images …")

    saved_paths = []          # track files for later test-set copy

    # ------------------------------------------------------------------
    # 8 normal road images
    # ------------------------------------------------------------------
    for i in range(1, 9):
        img = _make_normal_road(seed=i * 7)
        filename = f"road_normal_{i:03d}.jpg"
        path = os.path.join(raw_dir, filename)
        cv2.imwrite(path, img)
        saved_paths.append(path)
        print(f"  Saved: {filename}")

    # ------------------------------------------------------------------
    # 4 damaged road images
    # ------------------------------------------------------------------
    for i in range(1, 5):
        img = _make_damaged_road(seed=i * 13)
        filename = f"road_damaged_{i:03d}.jpg"
        path = os.path.join(raw_dir, filename)
        cv2.imwrite(path, img)
        saved_paths.append(path)
        print(f"  Saved: {filename}")

    # ------------------------------------------------------------------
    # Duplicate: byte-exact copy of road_normal_001.jpg
    # (Reading and re-saving with cv2 would alter JPEG pixels slightly;
    #  a raw file copy guarantees identical decoded pixel data.)
    # ------------------------------------------------------------------
    src_001 = os.path.join(raw_dir, "road_normal_001.jpg")
    dup_path = os.path.join(raw_dir, "road_normal_001_dup.jpg")
    with open(src_001, "rb") as fsrc, open(dup_path, "wb") as fdst:
        fdst.write(fsrc.read())
    print("  Saved: road_normal_001_dup.jpg  (byte-exact duplicate of road_normal_001.jpg)")

    # ------------------------------------------------------------------
    # Corrupted file — write raw bytes that are not a valid image
    # ------------------------------------------------------------------
    corrupted_path = os.path.join(raw_dir, "corrupted.jpg")
    with open(corrupted_path, "wb") as fh:
        fh.write(b"not_an_image_just_garbage_bytes_0123456789")
    print("  Saved: corrupted.jpg  (intentionally invalid)")

    # ------------------------------------------------------------------
    # Tiny image (20×20) — below minimum dimensions
    # ------------------------------------------------------------------
    tiny_img = np.full((20, 20, 3), 128, dtype=np.uint8)
    tiny_path = os.path.join(raw_dir, "tiny.jpg")
    cv2.imwrite(tiny_path, tiny_img)
    print("  Saved: tiny.jpg  (20×20, below MIN_IMAGE_WIDTH/HEIGHT)")

    # ------------------------------------------------------------------
    # Copy 3 images to data/test/
    # ------------------------------------------------------------------
    test_files = saved_paths[:3]
    for src in test_files:
        filename = os.path.basename(src)
        dst = os.path.join(test_dir, filename)
        img = cv2.imread(src)
        cv2.imwrite(dst, img)
        print(f"  Copied to test/: {filename}")

    total_raw = 8 + 4 + 1 + 1 + 1    # normal + damaged + dup + corrupted + tiny
    print(f"\n[INFO] create_sample_data: Done — {total_raw} files in data/raw/, "
          f"3 files in data/test/.")


if __name__ == "__main__":
    main()
