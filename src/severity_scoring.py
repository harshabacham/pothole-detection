"""
src/severity_scoring.py
=======================
Transparent mathematical scoring and categorical classification for
road surface defect candidates.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def calculate_severity_score(
    relative_size: float,
    anomaly_strength: float,
    edge_density: float,
    relative_contrast: float,
) -> float:
    """
    Compute a continuous severity score in [0.0, 1.0] using a transparent weighted formula.

    Formula
    -------
    Severity = w_size * S_size + w_anom * S_anom + w_edge * S_edge + w_contrast * S_contrast

    Parameters
    ----------
    relative_size : float
        Normalized size indicator in [0.0, 1.0] (region area relative to image).
    anomaly_strength : float
        Normalized anomaly score in [0.0, 1.0].
    edge_density : float
        Canny edge density within the defect candidate in [0.0, 1.0].
    relative_contrast : float
        Normalized intensity difference between defect and surrounding road.

    Returns
    -------
    float
        Continuous severity index in [0.0, 1.0].
    """
    s_size = float(np.clip(relative_size, 0.0, 1.0))
    s_anom = float(np.clip(anomaly_strength, 0.0, 1.0))
    s_edge = float(np.clip(edge_density, 0.0, 1.0))
    s_cont = float(np.clip(relative_contrast, 0.0, 1.0))

    score = (
        config.SEVERITY_WEIGHT_SIZE * s_size
        + config.SEVERITY_WEIGHT_ANOMALY * s_anom
        + config.SEVERITY_WEIGHT_EDGE * s_edge
        + config.SEVERITY_WEIGHT_CONTRAST * s_cont
    )

    return float(np.clip(score, 0.0, 1.0))


def assign_severity_category(score: float) -> str:
    """
    Map a continuous severity score to a discrete level: 'Low', 'Medium', or 'High'.

    Parameters
    ----------
    score : float
        Severity score in [0.0, 1.0].

    Returns
    -------
    str
        'Low', 'Medium', or 'High'
    """
    if score >= config.SEVERITY_THRESH_HIGH:
        return "High"
    elif score >= config.SEVERITY_THRESH_MEDIUM:
        return "Medium"
    else:
        return "Low"


def get_severity_color_bgr(category: str) -> tuple:
    """
    Return the visual bounding box color (BGR) corresponding to the severity category.
    """
    cat = str(category).capitalize()
    if cat == "High":
        return config.COLOR_SEV_HIGH_BGR
    elif cat == "Medium":
        return config.COLOR_SEV_MEDIUM_BGR
    else:
        return config.COLOR_SEV_LOW_BGR

