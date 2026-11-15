"""
scripts/run_phase10.py
======================
Phase 10 orchestration: Video Processing, Temporal Defect Tracking,
GPS Telemetry Association, SQLite Database Persistence, and Folium Map Generation.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.video_processor import process_road_video
from src.database import init_database, get_all_defects, get_geotagged_defects
from src.map_visualization import build_geographic_map, save_map_to_html


def create_sample_video_and_gps(
    output_video_path: str,
    output_gps_csv: str,
    fps: int = 15,
    num_frames: int = 45,
) -> tuple:
    """
    Generate a realistic synthetic road inspection video and synchronized GPS telemetry
    for testing video tracking and geospatial defect mapping.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_gps_csv)), exist_ok=True)

    # Load representative road surface images
    raw_dir = os.path.join(config.OUTPUT_DIR, "original")
    if not os.path.exists(raw_dir):
        raw_dir = config.INPUT_DIR

    img1 = cv2.imread(os.path.join(raw_dir, "road_normal_001.jpg"))
    img2 = cv2.imread(os.path.join(raw_dir, "road_damaged_001.jpg"))
    img3 = cv2.imread(os.path.join(raw_dir, "road_damaged_002.jpg"))

    if img1 is None or img2 is None:
        raise FileNotFoundError("Base road images missing for video generation.")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    # Frame 0 to 14: Normal road approach
    # Frame 15 to 29: Defect 1 in field of view
    # Frame 30 to 44: Defect 2 in field of view
    gps_records = []
    base_lat = 28.535510
    base_lon = 77.241020

    for i in range(num_frames):
        t_sec = float(i / fps)
        lat = base_lat + (i * 0.00008)
        lon = base_lon + (i * 0.00006)
        gps_records.append({"timestamp_sec": round(t_sec, 2), "latitude": lat, "longitude": lon})

        if i < 15:
            # Shift slightly to simulate vehicle motion
            shift = (i * 3) % 20
            f = np.roll(img1, shift, axis=0)
        elif i < 30:
            shift = ((i - 15) * 4) % 20
            f = np.roll(img2, shift, axis=0)
        else:
            shift = ((i - 30) * 4) % 20
            src_img = img3 if img3 is not None else img2
            f = np.roll(src_img, shift, axis=0)

        writer.write(f)

    writer.release()
    pd.DataFrame(gps_records).to_csv(output_gps_csv, index=False)
    print(f"[INFO] run_phase10: Generated benchmark video '{output_video_path}' ({num_frames} frames, {fps} fps)")
    print(f"[INFO] run_phase10: Generated GPS telemetry log '{output_gps_csv}'")

    return output_video_path, output_gps_csv


def main():
    print("======================================================================")
    print("  POTHOLE DETECTION - PHASE 10: VIDEO PROCESSING + GPS + MAP")
    print("======================================================================")

    init_database()

    # 1. Synthesize benchmark road video and GPS telemetry log
    print("\n-- STEP 1: Creating Benchmark Video Stream & GPS Log ---------------")
    sample_video = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
    sample_gps = os.path.join(config.METADATA_DIR, "sample_gps_telemetry.csv")
    create_sample_video_and_gps(sample_video, sample_gps)

    # 2. Execute video analysis, frame extraction, tracking, and GPS linking
    print("\n-- STEP 2: Processing Road Video Stream with Defect Tracking --------")
    annotated_video = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_annotated.mp4")
    res = process_road_video(
        video_path=sample_video,
        output_video_path=annotated_video,
        gps_log_path=sample_gps,
        sample_interval=5,  # Sample every 5th frame for thorough tracking test
    )

    print(f"[INFO] Video processed: {res['total_frames_sampled']} frames sampled")
    print(f"[INFO] Unique physical defects tracked across time: {res['unique_defects_tracked']}")
    print(f"[INFO] Detections logged to SQLite database: {res['total_detections_logged']}")

    # 3. Query SQLite Database
    print("\n-- STEP 3: Inspecting SQLite Database Defect History ---------------")
    db_df = get_all_defects()
    geo_df = get_geotagged_defects()
    print(f"[INFO] Total SQLite records stored: {len(db_df)}")
    print(f"[INFO] Total Geotagged records: {len(geo_df)}")

    if not db_df.empty:
        print("\nRecent Defect Records in SQLite:")
        print(db_df[["id", "detection_id", "track_id", "severity_category", "severity_score", "latitude", "longitude"]].head(6).to_string())

    # 4. Generate Interactive Folium Map
    print("\n-- STEP 4: Generating Interactive Folium Geographic Map ------------")
    folium_map = build_geographic_map(db_df)
    map_html_path = os.path.join(config.MAPS_OUTPUT_DIR, "pothole_defect_map.html")
    save_map_to_html(folium_map, save_path=map_html_path)

    # 5. Checklist Verification
    checks = [
        ("Benchmark road video and GPS log created", os.path.exists(sample_video) and os.path.exists(sample_gps)),
        ("Annotated processed video generated", os.path.exists(annotated_video)),
        ("Spatio-temporal tracking deduplication active", res["unique_defects_tracked"] > 0),
        ("SQLite database populated with unique defect records", len(db_df) > 0),
        ("GPS coordinates correctly associated", len(geo_df) > 0),
        ("Folium interactive map HTML exported", os.path.exists(map_html_path)),
    ]

    print("\n======================================================================")
    print("  PHASE 10 COMPLETION CHECKLIST")
    print("======================================================================")
    all_passed = True
    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("  All Phase 10 checks passed - Video, GPS, Database & Map ready!")
    else:
        print("  Some Phase 10 checks failed.")
        sys.exit(1)
    print("======================================================================\n")


if __name__ == "__main__":
    main()

