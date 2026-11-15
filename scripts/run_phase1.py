"""
scripts/run_phase1.py
======================
Orchestration script for Phase 1 of the Pothole Detection project.

Steps
-----
1.  Add project root to sys.path
2.  Generate synthetic sample data (data/raw/ and data/test/)
3.  Validate the raw dataset — print report + save JSON
4.  Preprocess all valid raw images and save stage outputs
5.  Save a preprocessing comparison figure for one sample image
6.  Save a dimension-distribution histogram
7.  Save a sample-gallery figure
8.  Save a validation-report bar chart
9.  Print Phase 1 completion checklist
"""

import os
import sys

# ---------------------------------------------------------------------------
# 1. Add project root to sys.path FIRST so all local imports resolve.
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config

# ---------------------------------------------------------------------------
# Local imports (must come after sys.path manipulation)
# ---------------------------------------------------------------------------
import scripts.create_sample_data as create_sample_data
from src.dataset_validation import validate_dataset
from src.data_loader         import load_dataset
from src.preprocessing       import preprocess_image, preprocess_and_save
from src.visualization       import (
    plot_preprocessing_comparison,
    plot_dimension_distribution,
    plot_sample_gallery,
    generate_dataset_report_figure,
)


# ---------------------------------------------------------------------------
# Helper — ensure metadata directory exists
# ---------------------------------------------------------------------------
def _metadata(filename: str) -> str:
    """Return absolute path inside METADATA_DIR, creating the dir if needed."""
    md = os.path.join(_PROJECT_ROOT, config.METADATA_DIR)
    os.makedirs(md, exist_ok=True)
    return os.path.join(md, filename)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  POTHOLE DETECTION - PHASE 1 PIPELINE")
    print("=" * 70 + "\n")

    # -----------------------------------------------------------------------
    # Step 2 — Generate synthetic sample data
    # -----------------------------------------------------------------------
    print("-- STEP 2: Generating synthetic sample data -----------------------")
    create_sample_data.main()
    print()

    # -----------------------------------------------------------------------
    # Step 3 — Validate raw dataset
    # -----------------------------------------------------------------------
    print("-- STEP 3: Validating raw dataset ---------------------------------")
    raw_dir = os.path.join(_PROJECT_ROOT, config.INPUT_DIR)
    validation_report = validate_dataset(raw_dir)
    print()

    # -----------------------------------------------------------------------
    # Step 4 — Preprocess and save
    # -----------------------------------------------------------------------
    print("-- STEP 4: Preprocessing images -----------------------------------")
    processed_dir = os.path.join(_PROJECT_ROOT, config.OUTPUT_DIR)
    preprocess_results = preprocess_and_save(raw_dir, processed_dir)
    print()

    # -----------------------------------------------------------------------
    # Step 5 — Preprocessing comparison figure (first valid image)
    # -----------------------------------------------------------------------
    print("-- STEP 5: Generating preprocessing comparison figure -------------")
    all_records = load_dataset(raw_dir)
    valid_records = [r for r in all_records if r["image"] is not None]

    if valid_records:
        sample_rec = valid_records[0]
        processed_dict = preprocess_image(sample_rec["image"], use_clahe=config.USE_CLAHE)
        comp_path = _metadata("sample_comparison.png")
        plot_preprocessing_comparison(
            original_bgr=sample_rec["image"],
            processed_dict=processed_dict,
            title=f"Preprocessing — {sample_rec['filename']}",
            save_path=comp_path,
        )
    else:
        print("[WARNING] run_phase1: No valid images found for comparison plot.")
    print()

    # -----------------------------------------------------------------------
    # Step 6 — Dimension distribution
    # -----------------------------------------------------------------------
    print("-- STEP 6: Generating dimension distribution plot -----------------")
    dim_path = _metadata("dimension_distribution.png")
    plot_dimension_distribution(all_records, save_path=dim_path)
    print()

    # -----------------------------------------------------------------------
    # Step 7 — Sample gallery
    # -----------------------------------------------------------------------
    print("-- STEP 7: Generating sample gallery ------------------------------")
    gallery_path = _metadata("sample_gallery.png")
    plot_sample_gallery(all_records, n=9, save_path=gallery_path)
    print()

    # -----------------------------------------------------------------------
    # Step 8 — Validation report bar chart
    # -----------------------------------------------------------------------
    print("-- STEP 8: Generating dataset report figure -----------------------")
    report_fig_path = _metadata("dataset_report.png")
    generate_dataset_report_figure(validation_report, save_path=report_fig_path)
    print()

    # -----------------------------------------------------------------------
    # Step 9 — Phase 1 completion checklist
    # -----------------------------------------------------------------------
    ok_count = sum(1 for r in preprocess_results if r["status"] == "ok")
    metadata_dir = os.path.join(_PROJECT_ROOT, config.METADATA_DIR)

    checklist = [
        ("Directory structure created",        True),
        ("Synthetic sample data generated",    True),
        ("Dataset validation completed",       True),
        ("Validation report saved (JSON)",     os.path.isfile(os.path.join(metadata_dir, "validation_report.json"))),
        ("Images preprocessed",                ok_count > 0),
        ("Comparison figure saved",            os.path.isfile(os.path.join(metadata_dir, "sample_comparison.png"))),
        ("Dimension distribution saved",       os.path.isfile(dim_path)),
        ("Sample gallery saved",               os.path.isfile(gallery_path)),
        ("Dataset report figure saved",        os.path.isfile(report_fig_path)),
    ]

    print("=" * 70)
    print("  PHASE 1 COMPLETION CHECKLIST")
    print("=" * 70)
    all_pass = True
    for label, status in checklist:
        tick = "[OK]" if status else "[FAIL]"
        print(f"  {tick} {label}")
        if not status:
            all_pass = False

    print("=" * 70)
    if all_pass:
        print("  All Phase 1 checks passed - pipeline is ready!")
    else:
        print("  Some checks failed - review warnings above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
