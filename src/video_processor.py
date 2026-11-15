"""
src/video_processor.py
======================
Comprehensive video analysis engine.
Integrates frame extraction, pre-fitted unsupervised ML inference, spatio-temporal
defect tracking, GPS telemetry linking, SQLite logging, and annotated video rendering.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.frame_extraction import get_video_properties, extract_video_frames
from src.inference_pipeline import load_all_artifacts, run_inference_on_image
from src.tracking import DefectTracker
from src.gps_handler import load_external_gps_log, interpolate_gps
from src.database import init_database, insert_defect
from src.severity_scoring import get_severity_color_bgr


def process_road_video(
    video_path: str,
    output_video_path: str = None,
    gps_log_path: str = None,
    sample_interval: int = None,
    artifacts: dict = None,
) -> dict:
    """
    Process an entire road video file through the unsupervised anomaly detection pipeline.

    Parameters
    ----------
    video_path : str
        Path to source MP4 or AVI video.
    output_video_path : str, optional
        Path for rendered output video.
    gps_log_path : str, optional
        Path to external GPS telemetry CSV.
    sample_interval : int, optional
        Analyze every N frames.
    artifacts : dict, optional
        Pre-loaded models. If None, loaded automatically.

    Returns
    -------
    dict
        Processing summary, unique tracks, and database status.
    """
    init_database()

    if artifacts is None:
        artifacts = load_all_artifacts()

    if sample_interval is None:
        sample_interval = config.VIDEO_SAMPLE_INTERVAL_FRAMES

    props = get_video_properties(video_path)
    stem = os.path.splitext(props["filename"])[0]

    if output_video_path is None:
        os.makedirs(config.VIDEOS_OUTPUT_DIR, exist_ok=True)
        output_video_path = os.path.join(config.VIDEOS_OUTPUT_DIR, f"{stem}_annotated.mp4")

    # Load external GPS if provided
    gps_df = None
    if gps_log_path and os.path.exists(gps_log_path):
        gps_df = load_external_gps_log(gps_log_path)
        print(f"[INFO] video_processor: Loaded {len(gps_df)} GPS telemetry records.")

    # Extract sampled frames
    sampled_frames = extract_video_frames(
        video_path,
        sample_interval=sample_interval,
        target_width=config.IMAGE_WIDTH,
        target_height=config.IMAGE_HEIGHT,
    )

    if not sampled_frames:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = max(1.0, float(props["fps"] / sample_interval))
    out_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        out_fps,
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
    )

    tracker = DefectTracker()
    recorded_track_ids = set()
    total_detections_logged = 0

    print(f"[INFO] video_processor: Analyzing {len(sampled_frames)} sampled frames with tracker...")

    for i, item in enumerate(sampled_frames, 1):
        frame_idx = item["frame_idx"]
        t_sec = item["timestamp_sec"]
        frame_bgr = item["frame_bgr"]

        # Run inference on frame without retraining
        res = run_inference_on_image(
            frame_bgr,
            artifacts,
            image_name=f"{stem}_f{frame_idx:05d}.jpg",
        )

        candidates = res["candidates_df"].to_dict(orient="records") if not res["candidates_df"].empty else []

        # Update tracker
        tracked_candidates = tracker.update(candidates, frame_idx, t_sec)

        # Annotate frame
        annotated_frame = frame_bgr.copy()

        for cand in tracked_candidates:
            x, y, w, h = cand["x"], cand["y"], cand["width"], cand["height"]
            t_id = cand["track_id"]
            cat = cand.get("severity_category", "Low")
            score = cand.get("severity_score", 0.0)
            color = get_severity_color_bgr(cat)

            # Draw box & text
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            label = f"{t_id} | {cat} ({score:.2f})"
            cv2.putText(annotated_frame, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            # GPS interpolation
            lat, lon = None, None
            if gps_df is not None:
                lat, lon = interpolate_gps(gps_df, t_sec)

            # Record unique defect track in SQLite database
            if t_id not in recorded_track_ids:
                recorded_track_ids.add(t_id)
                insert_defect(
                    detection_id=f"{stem}_{t_id}",
                    track_id=t_id,
                    latitude=lat,
                    longitude=lon,
                    timestamp=datetime.now().isoformat(),
                    severity_category=cat,
                    severity_score=score,
                    anomaly_score=cand.get("avg_anomaly_score", 0.0),
                    pixel_area=cand.get("pixel_area", 0),
                    image_path=output_video_path,
                    source_media=props["filename"],
                )
                total_detections_logged += 1

        # Frame telemetry watermark
        watermark = f"Time: {t_sec:.1f}s | Frame: {frame_idx} | Defects Tracked: {len(tracked_candidates)}"
        cv2.putText(annotated_frame, watermark, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, watermark, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        out_writer.write(annotated_frame)

    out_writer.release()
    unique_tracks = tracker.finalize()

    print(f"[INFO] video_processor: Video processing complete. Output saved to: {output_video_path}")
    print(f"[INFO] video_processor: {len(unique_tracks)} unique physical defects tracked and {total_detections_logged} logged to database.")

    return {
        "output_video_path": output_video_path,
        "total_frames_sampled": len(sampled_frames),
        "unique_defects_tracked": len(unique_tracks),
        "total_detections_logged": total_detections_logged,
        "tracks": unique_tracks,
    }

