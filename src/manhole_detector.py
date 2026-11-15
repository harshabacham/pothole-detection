"""
src/manhole_detector.py
=======================
Civil Infrastructure Asset & Sunken Manhole / Utility Trench Classifier.

CRITICAL REAL-WORLD PRODUCTION PROBLEM:
Circular cast-iron manhole covers and water utility valves are legally planned
civil infrastructure. Submitting a flush manhole cover as a road pothole is an embarrassing
false positive.

HOWEVER, SUNKEN MANHOLES ARE LETHAL HAZARDS:
When asphalt around a cast-iron manhole subsides or collapses, creating a sudden vertical
drop (> 2.5 to 5.0 cm), it becomes a severe accident hazard under IRC:82 road safety norms.
Similarly, poorly patched utility pipeline trenches (telecom/water excavations) create
dangerous transverse road steps.

DISCRIMINATION CRITERIA:
1. Circularity & Radial Symmetry:
   - Evaluates circularity metric = 4 * pi * (Area / Perimeter^2).
   - Manhole covers exhibit circularity >= 0.78 with concentric metallic rim edges.
2. Rim-to-Pavement Metric Elevation Differential:
   - Flush Manhole (Drop < 2.5 cm)  : MANHOLE_FLUSH_LEGAL (Suppress complaint).
   - Sunken Manhole (Drop >= 2.5 cm): IRC:SUNKEN_MANHOLE_HAZARD (File complaint to Water/Sewerage Board).
3. Trench Excavation Geometry:
   - Long straight rectangular trench cuts: IRC:UTILITY_TRENCH_EXCAVATION.
"""

import os
import sys
import math
import cv2
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class InfrastructureAssetClassifier:
    """
    Classifies civil infrastructure fixtures (flush manholes, sunken manholes,
    and utility trench cuts) vs natural road surface potholes.
    """

    def __init__(self, min_circularity: float = 0.76):
        self.min_circularity = min_circularity

    def evaluate_asset(
        self,
        patch_bgr: np.ndarray,
        measured_depth_cm: float = 0.0,
        bbox: tuple = None,
    ) -> dict:
        """
        Evaluate candidate anomaly patch.

        Returns
        -------
        dict with classification, is_manhole, is_sunken_hazard, target_agency, and suppress_grievance.
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return {
                "classification": "POTHOLE",
                "is_manhole": False,
                "is_sunken_hazard": False,
                "suppress_grievance": False,
                "target_authority": "Municipal PWD",
            }

        h, w = patch_bgr.shape[:2]
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY) if len(patch_bgr.shape) == 3 else patch_bgr.copy()

        # 1. Circularity & Circular Hough Transform
        circularity, has_circle_rim = self._check_circularity_and_rim(gray)

        # 2. Rectangular Utility Trench Test
        is_trench = self._check_utility_trench_geometry(gray, w, h)

        # -------------------------------------------------------------------
        # Decision Tree
        # -------------------------------------------------------------------
        if has_circle_rim or circularity >= self.min_circularity:
            # It is a manhole cover fixture
            if measured_depth_cm >= 2.5:
                # Sunken Manhole: The rim has subsided into the asphalt. Genuine road hazard!
                return {
                    "classification": "IRC:SUNKEN_MANHOLE_HAZARD",
                    "is_manhole": True,
                    "is_sunken_hazard": True,
                    "depth_cm": round(measured_depth_cm, 1),
                    "circularity": round(circularity, 2),
                    "suppress_grievance": False,  # DO file a grievance!
                    "target_authority": "Municipal Water Supply and Sewerage Board (DJB/BWSSB/MCGM)",
                    "action_required": "Raise cast-iron manhole frame flush to wearing course with polymer asphalt leveling ring.",
                }
            else:
                # Flush Manhole: Normal road fixture. Suppress false complaint!
                return {
                    "classification": "MANHOLE_FLUSH_LEGAL",
                    "is_manhole": True,
                    "is_sunken_hazard": False,
                    "depth_cm": round(measured_depth_cm, 1),
                    "circularity": round(circularity, 2),
                    "suppress_grievance": True,  # Suppress false alarm!
                    "target_authority": "Civil Infrastructure Asset",
                    "action_required": "None. Fixture is flush with road grade.",
                }

        elif is_trench:
            return {
                "classification": "IRC:UTILITY_TRENCH_EXCAVATION",
                "is_manhole": False,
                "is_sunken_hazard": True,
                "depth_cm": round(measured_depth_cm, 1),
                "circularity": round(circularity, 2),
                "suppress_grievance": False,
                "target_authority": "City Utility Coordination Division (Telecom / Gas Pipeline)",
                "action_required": "Compacted bitumen restoration of utility cable excavation cut under PWD street-cutting norms.",
            }

        # Otherwise standard pothole
        return {
            "classification": "POTHOLE",
            "is_manhole": False,
            "is_sunken_hazard": False,
            "depth_cm": round(measured_depth_cm, 1),
            "circularity": round(circularity, 2),
            "suppress_grievance": False,
            "target_authority": "Public Works Department (PWD / NHAI)",
            "action_required": "Cold/hot mix bitumen patching and mechanical roller compaction.",
        }

    def _check_circularity_and_rim(self, gray: np.ndarray) -> tuple:
        """
        Check contour isoperimetric circularity quotient and detect concentric circular rims.
        """
        # Threshold
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        # Check Hough Circles
        min_radius = max(8, min(gray.shape) // 6)
        max_radius = max(min_radius + 5, min(gray.shape) // 2)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_radius,
            param1=100,
            param2=28,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        has_circle_rim = circles is not None and len(circles[0]) > 0

        # Contour circularity quotient = 4 * pi * Area / (Perimeter^2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_circ = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 100 and perimeter > 0:
                circ = 4.0 * math.pi * (area / (perimeter * perimeter))
                if circ > max_circ:
                    max_circ = circ

        return float(min(1.0, max_circ)), has_circle_rim

    def _check_utility_trench_geometry(self, gray: np.ndarray, w: int, h: int) -> bool:
        """
        Detect rectangular slot cuts typical of fiber optic or water pipe excavations.
        """
        aspect = float(w) / max(1.0, float(h))
        # Trenches are typically elongated slots (aspect > 3.0 or aspect < 0.35)
        # with straight parallel edges
        if aspect > 3.5 or aspect < 0.28:
            return True
        return False
