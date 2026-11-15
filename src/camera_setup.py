"""
src/camera_setup.py
===================
Simplified Dashcam Connection & Visual Camera Selection Wizard.

Handles:
1. Automatic detection of all USB dashcams, webcams, and capture devices (Index 0, 1, 2...).
2. Captures visual preview snapshots so the user can easily distinguish between
   the laptop webcam (interior) and the external front dashcam (road view).
3. Saves preferred camera selection to persistent configuration.
4. Support for Wireless Smartphone Windshield Mounts (IP Webcam / RTSP stream).
"""

import os
import sys
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config

_SETTINGS_FILE = os.path.join(config.METADATA_DIR, "camera_settings.json")


def get_available_cameras(max_probe: int = 4) -> list:
    """
    Probe connected camera hardware and capture a live test thumbnail for each.

    Returns
    -------
    list of dict:
        [{"index": 0, "status": "Active", "preview_bgr": np.ndarray, "width": 640, "height": 480}, ...]
    """
    cameras = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                cameras.append({
                    "index": idx,
                    "label": f"Camera Device {idx}",
                    "width": w,
                    "height": h,
                    "preview_bgr": frame,
                })
            cap.release()
    return cameras


def get_saved_camera_config() -> dict:
    """Read saved camera configuration, or return defaults."""
    default_cfg = {
        "source_type": "usb_camera",  # "usb_camera", "phone_stream", "video_file"
        "camera_index": config.DASHCAM_CAMERA_INDEX,
        "stream_url": "http://192.168.1.100:8080/video",
        "video_file": os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4"),
    }
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                default_cfg.update(saved)
        except Exception:
            pass
    return default_cfg


def save_camera_config(camera_index: int = None, stream_url: str = None, source_type: str = None):
    """Persist preferred camera configuration to disk."""
    cfg = get_saved_camera_config()
    if camera_index is not None:
        cfg["camera_index"] = int(camera_index)
    if stream_url is not None:
        cfg["stream_url"] = str(stream_url)
    if source_type is not None:
        cfg["source_type"] = str(source_type)

    os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
    with open(_SETTINGS_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def get_active_dashcam_source():
    """
    Automatically resolve the active camera source.
    Returns: integer index (e.g. 0 or 1), or stream URL / video path string.
    """
    cfg = get_saved_camera_config()
    src_type = cfg.get("source_type", "usb_camera")

    if src_type == "phone_stream":
        return cfg.get("stream_url", "http://192.168.1.100:8080/video")
    elif src_type == "video_file":
        v_path = cfg.get("video_file", os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4"))
        return v_path if os.path.exists(v_path) else 0
    else:
        # USB camera index
        return cfg.get("camera_index", 0)
