"""
scripts/run_phase3.py
=====================
Phase 3 orchestration: Feature Extraction, Feature Scaling, and Feature Analysis.
"""

import os
import sys

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.features import extract_features_from_metadata_csv
from src.feature_scaling import fit_and_save
from src.feature_analysis import (
    summarize_features,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_normalized_comparison,
)


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 3: FEATURE EXTRACTION & SCALING")
    print("======================================================================")

    metadata_csv = os.path.join(config.PATCH_DIR, f"ps{config.FEATURE_PATCH_SIZE}", "metadata.csv")
    if not os.path.exists(metadata_csv):
        print(f"[ERROR] run_phase3: Metadata file not found at: {metadata_csv}")
        print("Please ensure Phase 2 has been executed first.")
        sys.exit(1)

    print(f"[INFO] Using metadata: {metadata_csv}")
    print(f"[INFO] Features target CSV: {config.FEATURES_CSV}")
    print(f"[INFO] Scaled features target CSV: {config.SCALED_FEATURES_CSV}")
    print(f"[INFO] Scaler artifact path: {config.SCALER_PATH}")

    # 1. Feature Extraction
    print("\n-- STEP 1: Extracting Multi-Group Features -------------------------")
    features_df = extract_features_from_metadata_csv(metadata_csv, config.FEATURES_CSV)

    # 2. Summary & Diagnostics
    print("\n-- STEP 2: Feature Diagnostics -------------------------------------")
    summarize_features(features_df)

    # 3. Feature Scaling (RobustScaler fitted on training features)
    print("\n-- STEP 3: Feature Scaling (RobustScaler) ---------------------------")
    scaled_df, scaler, feat_cols = fit_and_save(
        config.FEATURES_CSV,
        config.SCALED_FEATURES_CSV,
        config.SCALER_PATH,
    )

    # 4. Feature Visualizations
    print("\n-- STEP 4: Generating Feature Visualizations -----------------------")
    corr_path = os.path.join(config.METADATA_DIR, "feature_correlation.png")
    dist_path = os.path.join(config.METADATA_DIR, "feature_distributions.png")
    norm_comp_path = os.path.join(config.METADATA_DIR, "feature_normalized_comparison.png")

    plot_correlation_heatmap(features_df, max_features=30, save_path=corr_path)
    plot_feature_distributions(features_df, n_features=12, save_path=dist_path)
    plot_normalized_comparison(scaled_df, n_features=15, save_path=norm_comp_path)

    # 5. Checklist verification
    checks = [
        ("Raw features CSV created", os.path.exists(config.FEATURES_CSV)),
        ("Scaled features CSV created", os.path.exists(config.SCALED_FEATURES_CSV)),
        ("Scaler artifact saved (joblib)", os.path.exists(config.SCALER_PATH)),
        ("Feature correlation heatmap saved", os.path.exists(corr_path)),
        ("Feature distributions plot saved", os.path.exists(dist_path)),
        ("Normalized feature comparison saved", os.path.exists(norm_comp_path)),
    ]

    print("\n======================================================================")
    print("  PHASE 3 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("  All Phase 3 checks passed - Feature pipeline complete!")
    else:
        print("  Some Phase 3 checks failed. Please check logs.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

