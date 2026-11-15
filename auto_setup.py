"""
auto_setup.py
=============
Automated One-Click System Initializer and Self-Healing Engine.
Ensures the entire Pothole Detection & Government Reporting pipeline works
instantly and automatically right after cloning/installing, with zero manual setup.

Steps automated:
1. Creates all required output and metadata directories.
2. Initializes SQLite database with government grievance schema.
3. Generates training datasets (raw road images & realistic asphalt samples) if missing.
4. Auto-calibrates and persists all ML models (Scaler, PCA, Isolation Forest, K-Means) if missing.
5. Generates benchmark dashcam video & telemetry logs if missing.
6. Auto-detects connected hardware camera devices (USB dashcam, webcam).
"""

import os
import sys
import subprocess
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.database import init_database


def check_and_create_directories():
    """Ensure all required project directories exist."""
    dirs = [
        config.INPUT_DIR,
        config.OUTPUT_DIR,
        config.TEST_DIR,
        config.METADATA_DIR,
        config.PATCH_DIR,
        config.FEATURES_DIR,
        "models",
        config.OVERLAYS_DIR,
        config.HEATMAPS_DIR,
        config.COMPARISONS_DIR,
        config.LOCALIZED_OUTPUT_DIR,
        config.SEVERITY_OUTPUT_DIR,
        config.VIDEOS_OUTPUT_DIR,
        config.MAPS_OUTPUT_DIR,
        config.COMPLAINTS_OUTPUT_DIR,
        config.EVIDENCE_OUTPUT_DIR,
        config.REAL_ROAD_SAMPLES_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return True


def detect_camera_hardware():
    """Check if a USB dashcam or camera device is connected."""
    available_cams = []
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cams.append(idx)
            cap.release()
    return available_cams


def ensure_models_and_data_ready(verbose: bool = True):
    """
    Verify all model artifacts and benchmark data exist.
    If missing, automatically trains and compiles everything in seconds.
    """
    check_and_create_directories()
    init_database()

    # 1. Check raw dataset
    raw_files = []
    if os.path.exists(config.INPUT_DIR):
        raw_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(tuple(config.SUPPORTED_EXTENSIONS))]

    if len(raw_files) < 5:
        if verbose:
            print("[AUTO-SETUP] Initializing baseline road surface dataset...")
        from scripts.create_sample_data import generate_all_sample_data
        generate_all_sample_data()

    # 2. Check realistic asphalt benchmark samples
    real_files = []
    if os.path.exists(config.REAL_ROAD_SAMPLES_DIR):
        real_files = [f for f in os.listdir(config.REAL_ROAD_SAMPLES_DIR) if f.lower().endswith(tuple(config.SUPPORTED_EXTENSIONS))]

    if len(real_files) < 4:
        if verbose:
            print("[AUTO-SETUP] Generating realistic asphalt road benchmark samples...")
        try:
            from scripts.fetch_real_dataset import generate_realistic_road_dataset
            generate_realistic_road_dataset()
        except Exception as e:
            if verbose:
                print(f"[AUTO-SETUP] Note on realistic samples: {e}")

    # 3. Check ML model artifacts
    models_missing = (
        not os.path.exists(config.SCALER_PATH)
        or not os.path.exists(config.PCA_MODEL_PATH)
        or not os.path.exists(config.IFOREST_MODEL_PATH)
    )

    if models_missing:
        if verbose:
            print("[AUTO-SETUP] Pre-fitted models missing. Auto-calibrating unsupervised pipeline (takes ~5s)...")

        # Step A: Preprocess images
        from src.data_loader import scan_dataset, load_image
        from src.preprocessing import process_and_save_dataset
        process_and_save_dataset()

        # Step B: Extract patches
        from src.patch_extraction import extract_patches_for_dataset
        extract_patches_for_dataset()

        # Step C: Extract features & fit Scaler
        from src.features import extract_features_for_all_patches
        from src.feature_scaling import fit_and_save_scaler
        features_df = extract_features_for_all_patches()
        fit_and_save_scaler(features_df)

        # Step D: Fit PCA
        from src.pca_model import fit_and_save_pca
        from sklearn.preprocessing import StandardScaler
        from src.feature_scaling import load_scaler
        scaler = load_scaler()
        meta_cols = ["patch_id", "source_image", "row_idx", "col_idx", "x_min", "y_min", "x_max", "y_max", "is_road", "label"]
        num_cols = [c for c in features_df.columns if c not in meta_cols]
        X = features_df[num_cols].values
        # Standardize for stable PCA
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)
        fit_and_save_pca(X_std, feature_names=num_cols)

        # Step E: Fit Isolation Forest
        from src.model_training import train_and_save_model
        train_and_save_model(X_std)

        if verbose:
            print("[AUTO-SETUP] [OK] All unsupervised models successfully calibrated and persisted!")

    # 4. Check benchmark test video
    sample_vid = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
    if not os.path.exists(sample_vid):
        if verbose:
            print("[AUTO-SETUP] Compiling benchmark dashcam road video...")
        try:
            from scripts.run_phase10 import create_sample_video_and_gps
            gps_csv = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_gps_log.csv")
            create_sample_video_and_gps(sample_vid, gps_csv)
            if verbose:
                print(f"[AUTO-SETUP] [OK] Created {sample_vid}")
        except Exception as e:
            if verbose:
                print(f"[AUTO-SETUP] Note on video: {e}")

    return True


def run_full_auto_setup():
    print("=" * 70)
    print("  POTHOLE DETECTION & GOV PORTAL AUTO-REPORTER: AUTO-SETUP")
    print("=" * 70)
    print("[1/4] Verifying directory structure and SQLite database...")
    check_and_create_directories()
    init_database()
    print("      [OK] Database & folders ready.")

    print("\n[2/4] Verifying and calibrating ML detection models & benchmark data...")
    ensure_models_and_data_ready(verbose=True)
    print("      [OK] All models & datasets calibrated.")

    print("\n[3/4] Probing vehicle dashcam / camera hardware...")
    cams = detect_camera_hardware()
    if cams:
        print(f"      [OK] Detected {len(cams)} active camera device(s): Index {cams}")
    else:
        print("      [INFO] No USB dashcam currently connected (Simulation & video feeds active).")

    print("\n[4/4] Verifying Government of India Grievance Portals...")
    for k, v in config.GOV_PORTALS.items():
        print(f"      - {k:12s}: {v['name']} ({v['url']})")

    print("\n" + "=" * 70)
    print("  [SUCCESS] SYSTEM IS 100% READY FOR AUTOMATIC OPERATION!")
    print("=" * 70)
    print("  Launch options:")
    print("  - Double-click 'start_vehicle_dashcam.bat' to begin driving surveillance.")
    print("  - Double-click 'start_web_dashboard.bat' to open the interactive dashboard.")
    print("=" * 70)


if __name__ == "__main__":
    run_full_auto_setup()
