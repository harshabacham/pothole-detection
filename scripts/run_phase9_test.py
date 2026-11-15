"""
scripts/run_phase9_test.py
==========================
Automated verification test for Phase 9 Streamlit Dashboard and Inference Pipeline.
"""

import os
import sys
import cv2
import pandas as pd
import py_compile

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.inference_pipeline import load_all_artifacts, run_inference_on_image
from src.dashboard_helpers import bgr_to_png_bytes, dataframe_to_csv_bytes, check_pipeline_readiness


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 9: STREAMLIT DASHBOARD AUDIT")
    print("======================================================================")

    # 1. Check readiness
    is_ready, missing = check_pipeline_readiness()
    if not is_ready:
        print(f"[ERROR] Pipeline readiness check failed: missing {missing}")
        sys.exit(1)
    print("[INFO] Model readiness verified - all pre-fitted artifacts exist.")

    # 2. Syntax compile check on app.py
    print("\n-- STEP 1: Syntax & Bytecode Compilation Check ---------------------")
    app_path = os.path.join(_PROJECT_ROOT, "app.py")
    py_compile.compile(app_path, doraise=True)
    print("[INFO] app.py compiled successfully with zero syntax errors.")

    # 3. Model Loading
    print("\n-- STEP 2: Loading Pre-Fitted Unsupervised Models ------------------")
    artifacts = load_all_artifacts()
    print("[INFO] Models loaded successfully: Scaler, PCA, IsolationForest.")

    # 4. End-to-End Inference Test on Damaged Sample
    print("\n-- STEP 3: Live Inference Test on Damaged Road Surface -------------")
    test_img_path = os.path.join(config.OUTPUT_DIR, "original", "road_damaged_001.jpg")
    if not os.path.exists(test_img_path):
        test_img_path = os.path.join(config.INPUT_DIR, "road_damaged_001.jpg")

    test_bgr = cv2.imread(test_img_path)
    res = run_inference_on_image(test_bgr, artifacts, image_name="road_damaged_001.jpg")

    print(f"  - Total Patches Evaluated : {res['total_patches']}")
    print(f"  - Anomalies Detected      : {res['anomalous_patches']}")
    print(f"  - Defect Candidate Regions: {res['candidate_count']}")
    print(f"  - Low Severity Count      : {res['low_severity_count']}")
    print(f"  - Medium Severity Count   : {res['medium_severity_count']}")
    print(f"  - High Severity Count     : {res['high_severity_count']}")

    # 5. Serialization and Download Buffer Test
    print("\n-- STEP 4: Download Serialization Buffer Verification --------------")
    img_bytes = bgr_to_png_bytes(res["severity_bgr"])
    csv_bytes = dataframe_to_csv_bytes(res["candidates_df"])
    print(f"[INFO] Annotated image PNG buffer: {len(img_bytes)} bytes")
    print(f"[INFO] Candidate report CSV buffer: {len(csv_bytes)} bytes")

    # 6. Checklist Verification
    checks = [
        ("app.py syntax verified", os.path.exists(app_path)),
        ("Inference pipeline operational without retraining", res["total_patches"] > 0),
        ("Overlay, Heatmap & Severity views rendered", res["severity_bgr"] is not None and res["heatmap_bgr"] is not None),
        ("Candidate dataframe populated", not res["candidates_df"].empty),
        ("Export serialization buffers functional", len(img_bytes) > 0 and len(csv_bytes) > 0),
    ]

    print("\n======================================================================")
    print("  PHASE 9 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("  All Phase 9 checks passed - Streamlit Web Dashboard is ready to launch!")
    else:
        print("  Some Phase 9 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

