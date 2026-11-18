"""
src/yolo_detector.py
====================
Deep Learning YOLOv8 Pothole Detection Engine.
Integrates Ultralytics YOLOv8 for rapid and high-precision pothole detection
from webcam, phone cameras, and video streams.
"""

import os
import sys
import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config

_YOLO_MODEL = None
_YOLO_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "models", "y8best.pt")
_HF_WEIGHTS_URL = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"


def get_yolo_model(weights_path: str = None):
    """
    Load or retrieve the cached YOLOv8 pothole detection model.
    Auto-downloads weights if not found locally.
    """
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    if weights_path is None:
        weights_path = _YOLO_WEIGHTS_PATH

    # Check if weights exist; if not, download them
    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        try:
            import urllib.request
            print(f"[PotholeAI] Downloading YOLOv8 pothole weights from HuggingFace to {weights_path}...")
            urllib.request.urlretrieve(_HF_WEIGHTS_URL, weights_path)
            print("[PotholeAI] YOLOv8 weights downloaded successfully.")
        except Exception as dl_err:
            print(f"[PotholeAI] Failed to auto-download YOLOv8 weights: {dl_err}")
            return None

    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(weights_path)
        return _YOLO_MODEL
    except Exception as err:
        print(f"[PotholeAI] Failed to initialize YOLO model: {err}")
        return None


def detect_potholes_yolo(
    image_bgr: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model=None,
    image_filename: str = "",
) -> list:
    """
    Run YOLOv8 inference on a single image and return candidate dictionaries
    compatible with the rest of the PotholeAI pipeline (severity assessment,
    MoRTH compliance, 3D depth, and CPGRAMS dispatch).

    Returns
    -------
    list of dict:
        Candidate dictionaries with x, y, width, height, pixel_area, confidence, etc.
    """
    if model is None:
        model = get_yolo_model()

    if model is None or image_bgr is None or image_bgr.size == 0:
        return []

    h_img, w_img = image_bgr.shape[:2]

    try:
        results = model.predict(
            source=image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
    except Exception as exc:
        print(f"[PotholeAI] YOLO prediction error: {exc}")
        return []

    candidates = []
    idx = 1
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1 = max(0, int(xyxy[0]))
            y1 = max(0, int(xyxy[1]))
            x2 = min(w_img, int(xyxy[2]))
            y2 = min(h_img, int(xyxy[3]))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = bw * bh

            c_id = f"{os.path.splitext(image_filename)[0]}_cand_{idx:02d}" if image_filename else f"cand_{idx:02d}"

            candidates.append({
                "image_filename": image_filename,
                "candidate_id": c_id,
                "x": x1,
                "y": y1,
                "width": bw,
                "height": bh,
                "pixel_area": area,
                "box_area": area,
                "avg_anomaly_score": conf,
                "max_anomaly_score": conf,
                "confidence": conf,
                "detector_type": "YOLOv8",
                "class_id": cls_id,
            })
            idx += 1

    return candidates
