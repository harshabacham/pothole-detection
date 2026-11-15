"""
scripts/run_phase6.py
=====================
Phase 6 orchestration: Road Anomaly Overlays, Continuous Heatmap Generation,
and Quad Comparison Audits.
"""

import os
import sys

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.result_export import export_all_visualizations


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 6: ANOMALY VISUALIZATION & HEATMAPS")
    print("======================================================================")

    if not os.path.exists(config.ANOMALY_PREDICTIONS_CSV):
        print(f"[ERROR] Predictions CSV missing at: {config.ANOMALY_PREDICTIONS_CSV}")
        print("Please ensure Phase 5 has been executed first.")
        sys.exit(1)

    print(f"[INFO] Source predictions: {config.ANOMALY_PREDICTIONS_CSV}")
    print(f"[INFO] Overlays destination: {config.OVERLAYS_DIR}")
    print(f"[INFO] Heatmaps destination: {config.HEATMAPS_DIR}")
    print(f"[INFO] Comparisons destination: {config.COMPARISONS_DIR}")

    # 1. Generate all visualizations across dataset
    print("\n-- STEP 1: Rendering Visual Overlays, Heatmaps, and Comparison Quads --")
    counts = export_all_visualizations()

    # 2. Checklist verification
    checks = [
        ("Overlays directory created and populated", counts["overlays"] > 0 and os.path.exists(config.OVERLAYS_DIR)),
        ("Heatmaps directory created and populated", counts["heatmaps"] > 0 and os.path.exists(config.HEATMAPS_DIR)),
        ("Comparisons directory created and populated", counts["comparisons"] > 0 and os.path.exists(config.COMPARISONS_DIR)),
    ]

    print("\n======================================================================")
    print("  PHASE 6 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print(f"  All Phase 6 checks passed - {counts['comparisons']} full image audits generated!")
    else:
        print("  Some Phase 6 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

