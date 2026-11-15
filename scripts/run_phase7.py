"""
scripts/run_phase7.py
=====================
Phase 7 orchestration: Pothole Candidate Localization, Morphological Grouping,
Bounding Box Extraction, and Non-Maximum Suppression.
"""

import os
import sys
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.localization import localize_all_images


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 7: POTHOLE LOCALIZATION & BOUNDING BOXES")
    print("======================================================================")

    if not os.path.exists(config.ANOMALY_PREDICTIONS_CSV):
        print(f"[ERROR] Predictions CSV missing at: {config.ANOMALY_PREDICTIONS_CSV}")
        print("Please ensure Phase 5 has been executed first.")
        sys.exit(1)

    print(f"[INFO] Source predictions: {config.ANOMALY_PREDICTIONS_CSV}")
    print(f"[INFO] Localized images output dir: {config.LOCALIZED_OUTPUT_DIR}")
    print(f"[INFO] Candidate regions table: {config.LOCALIZED_CANDIDATES_CSV}")

    # 1. Run localization across dataset
    print("\n-- STEP 1: Spatial Grouping & Bounding Box Localization -------------")
    candidates_df, count = localize_all_images()

    # 2. Diagnostics & Summaries
    print("\n-- STEP 2: Candidate Region Analysis -------------------------------")
    if not candidates_df.empty:
        print(f"[INFO] Total candidate defect regions identified: {count}")
        print("\nCandidate Regions Breakdown:")
        summary = candidates_df.groupby("image_filename").agg(
            candidates=("candidate_id", "count"),
            mean_area=("pixel_area", "mean"),
            mean_score=("avg_anomaly_score", "mean"),
            max_score=("max_anomaly_score", "max"),
        ).reset_index()
        for _, r in summary.iterrows():
            print(f"  - {r['image_filename']:20s} : {r['candidates']} candidate(s) | Mean Area: {r['mean_area']:.1f}px | Avg Score: {r['mean_score']:.2f} | Max Score: {r['max_score']:.2f}")
    else:
        print("[WARNING] No candidate regions passed size and anomaly thresholds.")

    # 3. Checklist Verification
    checks = [
        ("Localized output images saved", os.path.exists(config.LOCALIZED_OUTPUT_DIR) and len(os.listdir(config.LOCALIZED_OUTPUT_DIR)) > 0),
        ("Localized candidates CSV saved", os.path.exists(config.LOCALIZED_CANDIDATES_CSV)),
        ("Bounding box metadata complete", not candidates_df.empty and "pixel_area" in candidates_df.columns),
    ]

    print("\n======================================================================")
    print("  PHASE 7 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print(f"  All Phase 7 checks passed - {count} localized candidate regions identified!")
    else:
        print("  Some Phase 7 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

