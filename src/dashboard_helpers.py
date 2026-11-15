"""
src/dashboard_helpers.py
========================
Streamlit helper utilities: model caching, image color space conversions,
byte serialization for downloads, and metric rendering components.
"""

import os
import sys
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.inference_pipeline import load_all_artifacts


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR image array to RGB for Streamlit rendering.
    """
    if image_bgr is None:
        return None
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    """
    Encode an OpenCV BGR image array to PNG byte stream for download buttons.
    """
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG buffer.")
    return buffer.tobytes()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialize a pandas DataFrame into UTF-8 encoded CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def check_pipeline_readiness() -> tuple:
    """
    Verify whether all required ML model artifacts exist on disk.
    """
    missing = []
    if not os.path.exists(config.SCALER_PATH):
        missing.append("RobustScaler (models/scaler.joblib)")
    if not os.path.exists(config.PCA_MODEL_PATH):
        missing.append("PCA Model (models/pca_model.joblib)")
    if not os.path.exists(config.IFOREST_MODEL_PATH):
        missing.append("Isolation Forest (models/isolation_forest.joblib)")

    is_ready = (len(missing) == 0)
    return is_ready, missing

