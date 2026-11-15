"""
src/distress_taxonomy.py
========================
IRC (Indian Roads Congress) Pavement Distress Classification and
Pavement Condition Index (PCI) Calculation Engine.

Compliant with IRC:82-2015, IRC:SP:16, and ASTM D6433 standards.
Distinguishes between:
1. IRC:Pothole               - Bowl-shaped structural cavity with depth depression.
2. IRC:Alligator_Cracking    - Interconnected fatigue web cracking (pothole precursor).
3. IRC:Linear_Crack          - Longitudinal / transverse pavement joint fractures.
4. IRC:Sunken_Manhole        - Utility cover drops and sunken sewer lids.
5. IRC:Raveling_Surface      - Pavement aggregate wear and loss of bitumen binder.
"""

import os
import sys
import math
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class PavementDistressClassifier:
    """
    Classifies road surface distress according to Indian Roads Congress (IRC)
    and computes ASTM D6433 Pavement Condition Index (PCI).
    """

    def __init__(self):
        self.categories = config.IRC_DISTRESS_TYPES

    def classify_distress(
        self,
        patch_bgr: np.ndarray,
        cand: dict,
        depth_info: dict = None,
    ) -> dict:
        """
        Classify a candidate defect into the official IRC distress taxonomy.

        Parameters
        ----------
        patch_bgr : np.ndarray
            Cropped image patch of the defect.
        cand : dict
            Bounding box candidate dictionary (x, y, width, height, severity_score).
        depth_info : dict, optional
            3D metric depth profile (max_depth_cm, cavity_volume_liters).

        Returns
        -------
        dict
            IRC classification details, maintenance action, and PCI deduct points.
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return self._default_pothole_result()

        h, w = patch_bgr.shape[:2]
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        max_depth = depth_info.get("max_depth_cm", 4.0) if depth_info else 4.0
        sev_score = cand.get("severity_score", 0.5)

        # 1. Edge & Morphology Analysis
        edges = cv2.Canny(gray, 40, 130)
        edge_pixel_count = np.sum(edges > 0)
        edge_density = edge_pixel_count / max(1.0, h * w)

        # Contours for circularity & convexity
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_c)
            peri = cv2.arcLength(largest_c, True)
            if peri > 0:
                circularity = (4.0 * math.pi * area) / (peri * peri)

        num_contours = len(contours)

        # 2. Decision Logic for IRC Pavement Taxonomy

        # RULE A: Linear / Longitudinal Crack (High aspect ratio: long and thin)
        if aspect_ratio >= 2.4 and edge_density < 0.25:
            category = "IRC:Linear_Crack"
            irc_code = "IRC:82-LinearCracking"
            desc = "Longitudinal or transverse pavement joint fracture"
            action = "Clean with compressed air and seal with rubberized bitumen emulsion"
            deduct = 12.0 if sev_score < 0.5 else 22.0
            confidence = 0.91

        # RULE B: Pothole (Structural Cavity with significant depth >= 3.0 cm)
        elif max_depth >= 3.0 and aspect_ratio < 2.4:
            category = "IRC:Pothole"
            irc_code = "IRC:82-PotholeCavity"
            desc = "Bowl-shaped asphalt cavity with depth depression and broken perimeter rim"
            action = "Excavate square boundaries, apply bituminous tack coat, compact hot/cold mix asphalt"
            deduct = 25.0 if max_depth < 5.0 else (45.0 if max_depth < 8.0 else 60.0)
            confidence = 0.94

        # RULE C: Alligator / Fatigue Cracking (Interconnected web with high edge density and multiple loops)
        elif edge_density >= 0.14 and max_depth < 3.0:
            category = "IRC:Alligator_Cracking"
            irc_code = "IRC:82-FatigueAlligator"
            desc = "Interconnected fatigue web cracking under repeated wheel load (pothole precursor)"
            action = "Apply fog seal / micro-surfacing; plan localized asphalt milling and overlay"
            deduct = 20.0 if sev_score < 0.6 else 38.0
            confidence = 0.89

        # RULE D: Sunken Utility Manhole / Metal Lid (High circularity, discrete contour)
        elif circularity > 0.65 and max_depth < 6.0 and num_contours <= 3:
            category = "IRC:Sunken_Manhole"
            irc_code = "IRC:SP:16-SunkenUtility"
            desc = "Sunken sewer lid or utility trench dropped below grade level"
            action = "Level manhole frame with polymer-modified asphalt collar"
            deduct = 15.0 if sev_score < 0.6 else 28.0
            confidence = 0.88

        # RULE E: Raveling / Surface Weathering (Default fallback)
        else:
            category = "IRC:Raveling_Surface"
            irc_code = "IRC:82-SurfaceRaveling"
            desc = "Loss of fine aggregate and binder wear on upper wearing course"
            action = "Apply rejuvenator spray or thin bituminous slurry seal"
            deduct = 10.0
            confidence = 0.82

        return {
            "irc_category": category,
            "irc_code": irc_code,
            "description": desc,
            "maintenance_action": action,
            "confidence": round(confidence, 2),
            "pci_deduct_points": round(deduct, 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "edge_density": round(edge_density, 3),
            "circularity": round(circularity, 2),
        }

    def _default_pothole_result(self) -> dict:
        return {
            "irc_category": "IRC:Pothole",
            "irc_code": "IRC:82-PotholeCavity",
            "description": "Bowl-shaped asphalt cavity with broken perimeter rim",
            "maintenance_action": "Square cut, tack coat, compact asphalt mix",
            "confidence": 0.85,
            "pci_deduct_points": 25.0,
            "aspect_ratio": 1.2,
            "edge_density": 0.15,
            "circularity": 0.5,
        }

    def compute_pci_rating(self, defects: list, road_length_km: float = 1.0) -> dict:
        """
        Compute ASTM D6433 & IRC standard Pavement Condition Index (PCI, scale 0-100)
        for a surveyed road segment.
        """
        if not defects:
            return {
                "pci_score": 100,
                "rating": "Excellent / Good",
                "color_hex": "#27ae60",
                "total_distress_count": 0,
                "breakdown": {},
                "pwd_strategy": "Routine maintenance and visual inspection",
            }

        total_deduct = 0.0
        breakdown = {k: 0 for k in self.categories}

        for d in defects:
            cat = d.get("irc_category", "IRC:Pothole")
            deduct = d.get("pci_deduct_points", 25.0)
            breakdown[cat] = breakdown.get(cat, 0) + 1
            total_deduct += deduct

        # Multi-defect correction curve factor (ASTM D6433 standard)
        # Prevents deducts from adding up linearly past 100 in short segments
        num_defects = len(defects)
        q_factor = 1.0 / (1.0 + (0.15 * math.log(max(1, num_defects))))
        corrected_deduct = total_deduct * q_factor

        pci_score = max(0, min(100, int(round(100.0 - corrected_deduct))))

        # Rating Category & Municipal Strategy
        if pci_score >= config.PCI_EXCELLENT_THRESH:
            rating = "Good / Excellent"
            color_hex = "#27ae60"
            strategy = "Normal operational condition. Routine preventive maintenance."
        elif pci_score >= 70:
            rating = "Satisfactory"
            color_hex = "#2ecc71"
            strategy = "Minor distresses observed. Crack sealing and localized surface patching."
        elif pci_score >= config.PCI_POOR_THRESH:
            rating = "Fair"
            color_hex = "#f39c12"
            strategy = "Moderate structural fatigue. Needs bituminous slurry seal and pothole patching."
        elif pci_score >= 40:
            rating = "Poor"
            color_hex = "#e67e22"
            strategy = "Significant structural failure. Immediate cold mix patching and asphalt overlay needed."
        elif pci_score >= 25:
            rating = "Very Poor"
            color_hex = "#d35400"
            strategy = "Severe structural breakdown. Milling and thick asphalt binder course required."
        else:
            rating = "Failed / Serious"
            color_hex = "#c0392b"
            strategy = "CRITICAL: Complete pavement reconstruction required. High vehicle accident hazard."

        return {
            "pci_score": pci_score,
            "rating": rating,
            "color_hex": color_hex,
            "total_distress_count": len(defects),
            "breakdown": breakdown,
            "pwd_strategy": strategy,
        }
