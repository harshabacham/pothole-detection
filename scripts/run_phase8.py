"""
scripts/run_phase8.py
=====================
Phase 8 orchestration: Preliminary Visual Severity Estimation,
Multi-Indicator Scoring, and Severity Audit Reporting.
"""

import os
import sys
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.severity_report import process_and_report_severity


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 8: VISUAL SEVERITY ESTIMATION")
    print("======================================================================")

    if not os.path.exists(config.LOCALIZED_CANDIDATES_CSV):
        print(f"[ERROR] Localized candidates table missing at: {config.LOCALIZED_CANDIDATES_CSV}")
        print("Please ensure Phase 7 has been executed first.")
        sys.exit(1)

    print(f"[INFO] Source candidates: {config.LOCALIZED_CANDIDATES_CSV}")
    print(f"[INFO] Severity images output dir: {config.SEVERITY_OUTPUT_DIR}")
    print(f"[INFO] Severity report CSV: {config.SEVERITY_REPORT_CSV}")

    # 1. Process severity across localized candidates
    print("\n-- STEP 1: Estimating Visual Defect Indicators & Severity ----------")
    report_df, summary = process_and_report_severity()

    # 2. Diagnostics & Summaries
    print("\n-- STEP 2: Severity Distribution Summary ---------------------------")
    print(f"  Total Candidates Evaluated : {summary['total_candidates']}")
    print(f"  - Low Severity Candidates  : {summary['low_severity']} ({(summary['low_severity']/max(1, summary['total_candidates']))*100:.1f}%)")
    print(f"  - Medium Severity Candidates: {summary['medium_severity']} ({(summary['medium_severity']/max(1, summary['total_candidates']))*100:.1f}%)")
    print(f"  - High Severity Candidates : {summary['high_severity']} ({(summary['high_severity']/max(1, summary['total_candidates']))*100:.1f}%)")

    if not report_df.empty:
        print("\nTop 5 Most Severe Defect Candidates:")
        top5 = report_df.sort_values(by="severity_score", ascending=False).head(5)
        for _, r in top5.iterrows():
            print(f"  - {r['candidate_id']:25s} | Sev: {r['severity_score']:.2f} [{r['severity_category']:6s}] | Area: {r['pixel_area']:5d}px | Contrast: {r['road_contrast']:.3f} | EdgeDens: {r['edge_density']:.3f}")

    # 3. Checklist Verification
    checks = [
        ("Severity report CSV saved", os.path.exists(config.SEVERITY_REPORT_CSV)),
        ("Severity output images saved", os.path.exists(config.SEVERITY_OUTPUT_DIR) and len(os.listdir(config.SEVERITY_OUTPUT_DIR)) > 0),
        ("Calculated visual indicators valid", not report_df.empty and "road_contrast" in report_df.columns),
    ]

    print("\n======================================================================")
    print("  PHASE 8 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print(f"  All Phase 8 checks passed - Severity assessment pipeline complete!")
    else:
        print("  Some Phase 8 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

