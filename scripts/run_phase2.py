"""
scripts/run_phase2.py
=====================
Phase 2 orchestration script for the Pothole Detection pipeline.

Steps:
  1. Load processed images from data/processed/original/
     (Falls back to data/raw/ with a warning if the processed directory is missing.)
  2. For each patch size in PATCH_SIZES [32, 64, 128]:
       a. Extract patches from all images.
       b. Save patches to data/patches/ps{patch_size}/.
       c. Save metadata CSV to data/patches/ps{patch_size}/metadata.csv.
       d. Print summary.
  3. For the DEFAULT_PATCH_SIZE (64):
       a. Generate ROI+grid visualization for the first 3 images.
       b. Generate a patch gallery.
       c. Generate a valid-vs-invalid bar chart.
  4. Print Phase 2 completion checklist.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Add project root to sys.path before any local imports.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import config
from src.data_loader import load_dataset
from src.roi_detection import get_roi_bounds
from src.patch_extraction import extract_patches, save_patches, extract_and_save_all
from src.patch_visualization import (
    visualize_roi_and_grid,
    visualize_patch_gallery,
    visualize_valid_vs_invalid,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _section(text: str) -> None:
    print("")
    print("=" * 70)
    print(text)
    print("=" * 70)


def _resolve_input_dir() -> str:
    """
    Return the path to the image directory to use as input.
    Prefers data/processed/original/; falls back to data/raw/.
    """
    preferred = os.path.join(_PROJECT_ROOT, config.OUTPUT_DIR, "original")
    fallback  = os.path.join(_PROJECT_ROOT, config.INPUT_DIR)

    if os.path.isdir(preferred):
        # Check that the directory actually has images
        has_images = any(
            os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTENSIONS
            for f in os.listdir(preferred)
        )
        if has_images:
            print(f"[INFO] Input directory: {preferred}")
            return preferred

    print(
        f"[WARNING] Processed originals not found at '{preferred}'. "
        f"Falling back to '{fallback}'."
    )
    return fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _section("Phase 2 — ROI and Patch Extraction")

    # ------------------------------------------------------------------
    # Resolve input directory
    # ------------------------------------------------------------------
    input_dir = _resolve_input_dir()

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2 — Extract and save patches for each patch size
    # ------------------------------------------------------------------
    patch_summaries = {}   # patch_size -> list of all patches (for Step 3)

    for ps in config.PATCH_SIZES:
        _section(f"Patch Size: {ps}px")

        output_dir = os.path.join(_PROJECT_ROOT, config.PATCH_DIR, f"ps{ps}")
        os.makedirs(output_dir, exist_ok=True)

        all_patches, metadata_df = extract_and_save_all(
            input_dir=input_dir,
            output_dir=output_dir,
            patch_size=ps,
            overlap=config.PATCH_OVERLAP,
        )

        if not all_patches:
            print(f"[WARNING] No patches extracted for patch_size={ps}. Skipping.")
            patch_summaries[ps] = []
            continue

        # Save metadata CSV
        csv_path = os.path.join(output_dir, "metadata.csv")
        metadata_df.to_csv(csv_path, index=False)
        print(f"[INFO] Metadata CSV saved: {csv_path}")

        # Summary statistics
        total_patches = len(all_patches)
        valid_patches = sum(1 for p in all_patches if p["is_valid"])

        # Per-image patch count
        per_image = {}
        for p in all_patches:
            fname = p["image_filename"]
            per_image[fname] = per_image.get(fname, 0) + 1

        print(f"[INFO] patch_size={ps}px | total={total_patches} | valid={valid_patches} | invalid={total_patches - valid_patches}")
        print(f"[INFO] Patches per image:")
        for fname, cnt in sorted(per_image.items()):
            print(f"       {fname}: {cnt}")

        patch_summaries[ps] = all_patches

    # ------------------------------------------------------------------
    # Step 3 — Visualizations for DEFAULT_PATCH_SIZE
    # ------------------------------------------------------------------
    _section(f"Visualizations for DEFAULT_PATCH_SIZE={config.DEFAULT_PATCH_SIZE}px")

    metadata_dir = os.path.join(_PROJECT_ROOT, config.METADATA_DIR)
    os.makedirs(metadata_dir, exist_ok=True)

    default_patches = patch_summaries.get(config.DEFAULT_PATCH_SIZE, [])

    if not default_patches:
        print(
            f"[WARNING] No patches available for DEFAULT_PATCH_SIZE={config.DEFAULT_PATCH_SIZE}. "
            "Skipping visualizations."
        )
    else:
        # 3a — ROI+grid visualization for first 3 images
        records = load_dataset(input_dir)
        valid_records = [r for r in records if r["image"] is not None][:3]

        for rec in valid_records:
            stem = os.path.splitext(rec["filename"])[0]
            img_patches = [
                p for p in default_patches
                if p["image_filename"] == rec["filename"]
            ]

            try:
                roi = get_roi_bounds(rec["image"])
            except ValueError as exc:
                print(f"[WARNING] Could not compute ROI for '{rec['filename']}': {exc}")
                continue

            save_path = os.path.join(metadata_dir, f"roi_grid_{stem}.png")
            visualize_roi_and_grid(
                image_bgr=rec["image"],
                roi=roi,
                patch_list=img_patches,
                patch_size=config.DEFAULT_PATCH_SIZE,
                title=f"ROI and Patch Grid — {rec['filename']}",
                save_path=save_path,
            )

        # 3b — Patch gallery
        gallery_path = os.path.join(metadata_dir, f"patch_gallery_ps{config.DEFAULT_PATCH_SIZE}.png")
        visualize_patch_gallery(
            patch_list=default_patches,
            n=16,
            use_color=False,
            title=f"Patch Gallery (ps={config.DEFAULT_PATCH_SIZE}px)",
            save_path=gallery_path,
        )

        # 3c — Valid vs invalid bar chart
        validity_path = os.path.join(metadata_dir, f"patch_validity_ps{config.DEFAULT_PATCH_SIZE}.png")
        visualize_valid_vs_invalid(
            patch_list=default_patches,
            save_path=validity_path,
        )

    # ------------------------------------------------------------------
    # Step 4 — Phase 2 completion checklist
    # ------------------------------------------------------------------
    _section("Phase 2 Completion Checklist")

    checks = [
        ("config.py extended with ROI and patch settings",
         os.path.isfile(os.path.join(_PROJECT_ROOT, "config.py"))),
        ("src/roi_detection.py created",
         os.path.isfile(os.path.join(_PROJECT_ROOT, "src", "roi_detection.py"))),
        ("src/patch_extraction.py created",
         os.path.isfile(os.path.join(_PROJECT_ROOT, "src", "patch_extraction.py"))),
        ("src/patch_visualization.py created",
         os.path.isfile(os.path.join(_PROJECT_ROOT, "src", "patch_visualization.py"))),
        ("scripts/run_phase2.py created",
         os.path.isfile(os.path.join(_PROJECT_ROOT, "scripts", "run_phase2.py"))),
    ]

    # Check patch outputs
    for ps in config.PATCH_SIZES:
        csv_path = os.path.join(_PROJECT_ROOT, config.PATCH_DIR, f"ps{ps}", "metadata.csv")
        checks.append((
            f"Patches extracted and metadata saved for ps={ps}",
            os.path.isfile(csv_path),
        ))

    # Check visualization outputs
    vis_files = [
        f"patch_gallery_ps{config.DEFAULT_PATCH_SIZE}.png",
        f"patch_validity_ps{config.DEFAULT_PATCH_SIZE}.png",
    ]
    for vf in vis_files:
        vpath = os.path.join(metadata_dir, vf)
        checks.append((f"Visualization '{vf}' saved", os.path.isfile(vpath)))

    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status}  {desc}")
        if not passed:
            all_passed = False

    print("")
    if all_passed:
        print("[OK] Phase 2 complete — all checks passed.")
    else:
        print("[FAIL] Phase 2 has one or more failures. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

