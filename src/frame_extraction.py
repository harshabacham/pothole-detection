"""
src/frame_extraction.py
=======================
Video ingest, metadata inspection, and frame extraction utilities for the
Pothole Detection pipeline. Supports MP4 and AVI containers.
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def get_video_properties(video_path: str) -> dict:
    """
    Query fundamental video properties including frame rate, resolution, and frame counts.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video stream for file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = float(total_frames / fps) if fps > 0 else 0.0

    cap.release()

    return {
        "video_path": video_path,
        "filename": os.path.basename(video_path),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": round(duration_sec, 2),
    }


def extract_video_frames(
    video_path: str,
    sample_interval: int = None,
    target_width: int = None,
    target_height: int = None,
    max_frames: int = None,
) -> list:
    """
    Extract and resize frames at a specified interval from a video container.

    Parameters
    ----------
    video_path : str
        Path to video file.
    sample_interval : int, optional
        Sample every N-th frame (e.g. 10).
    target_width : int, optional
        Target resized width (default from config).
    target_height : int, optional
        Target resized height (default from config).
    max_frames : int, optional
        Maximum sampled frames to return (useful for previewing).

    Returns
    -------
    list of dicts
        Each dict has 'frame_idx', 'timestamp_sec', 'frame_bgr'
    """
    if sample_interval is None:
        sample_interval = config.VIDEO_SAMPLE_INTERVAL_FRAMES
    if target_width is None:
        target_width = config.IMAGE_WIDTH
    if target_height is None:
        target_height = config.IMAGE_HEIGHT

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    sampled_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = float(frame_idx / fps)
            # Consistent resizing for the downstream ML pipeline
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame.copy()

            sampled_frames.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                "frame_bgr": frame_resized,
            })

            if max_frames and len(sampled_frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"[INFO] frame_extraction: Sampled {len(sampled_frames)} frames from {video_path} (interval={sample_interval})")
    return sampled_frames

