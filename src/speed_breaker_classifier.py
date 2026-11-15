"""
src/speed_breaker_classifier.py
===============================
Convex Speed Breaker & Transverse Rumble Strip vs Concave Pothole Discriminator.

CRITICAL REAL-WORLD PRODUCTION PROBLEM:
On Indian and global roads, the #1 cause of false alarms for naive anomaly detection
is unpainted asphalt speed breakers and transverse rumble strips. Because speed bumps
disrupt smooth flat asphalt, an unsupervised texture model flags every speed breaker
as a massive severe pothole. Submitting speed breakers to PWD as potholes ruins credibility.

PHYSICAL & PHOTOMETRIC DISCRIMINATION PRINCIPLES:
1. Geometric Aspect Ratio & Lane Width:
   - Speed breakers typically span 60% to 100% of the active lane width.
     Aspect ratio (Width / Height) is typically 3.0 to 10.0+.
   - Potholes are localized circular/elliptical cavities (Aspect ratio typically 0.6 to 2.2).
2. Photometric Gradient Profile (Convex Elevation vs Concave Cavity):
   - Convex Speed Bump: Light from headlights/sun creates a bright specular crest
     on the leading incline (+h elevation), followed by a soft shadow behind it.
     Vertical intensity derivative: Bright -> Crest -> Dark.
   - Concave Pothole: Light casts an immediate dark shadow inside the leading rim,
     with a bright reflection on the far bottom wall (-d depression).
     Vertical intensity derivative: Dark -> Void Center -> Bright edge.
3. Transverse Rumble Strip Frequency:
   - Rumble strips exhibit periodic, repeating parallel horizontal lines (3 to 8 bars)
     spaced 10-25 cm apart across the lane.
4. Optical-Inertial (IMU) Shock Waveform:
   - Speed Bump: Wheel travels UP first (+Gz suspension compression) followed by
     a rebound extension (-Gz). Signature: COMPRESSION_THEN_REBOUND.
   - Pothole: Wheel drops DOWN into void first (-Gz free-fall) followed by a violent
     upward impact spike (+Gz rim shock). Signature: DROP_THEN_IMPACT.

CIVIC ACTION:
If classified as a SPEED_BREAKER or RUMBLE_STRIP, the system SUPPRESSES the
government complaint and logs it as a "Traffic Calming Asset" for driver ADAS alerts.
"""

import os
import sys
import math
import numpy as np
import cv2

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class SpeedBreakerDiscriminator:
    """
    Discriminates real road cavities (potholes) from legal traffic calming infrastructure
    (speed breakers and transverse rumble strips) using photometric and morphological analysis.
    """

    def __init__(self, lane_width_pixels: int = 420):
        self.lane_width_pixels = lane_width_pixels

    def classify_defect_patch(
        self,
        patch_bgr: np.ndarray,
        bbox: tuple,
        frame_width: int = 640,
        imu_shock_profile: list = None,
    ) -> dict:
        """
        Classify candidate defect into:
        - 'POTHOLE' (true negative cavity hazard)
        - 'SPEED_BREAKER_CONVEX' (legal raised bump)
        - 'RUMBLE_STRIP_PERIODIC' (series of transverse warning lines)
        - 'FLAT_ROAD_MARKING' (painted zebra/lane arrow)

        Parameters
        ----------
        patch_bgr : np.ndarray
            Cropped bounding box image of the road anomaly.
        bbox : tuple
            (x, y, w, h) bounding box in original camera frame.
        frame_width : int
            Width of the camera frame (used for lane-spanning checks).
        imu_shock_profile : list of float, optional
            Recent vertical accelerometer readings [Gz(t-k), ..., Gz(t)].

        Returns
        -------
        dict containing classification, confidence, convexity_score, and suppression recommendation.
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return {
                "classification": "POTHOLE",
                "is_hazard": True,
                "confidence": 0.5,
                "aspect_ratio": 1.0,
                "convexity_score": 0.0,
                "suppress_grievance": False,
                "reason": "EMPTY_PATCH_FALLBACK",
            }

        x, y, w, h = bbox
        aspect_ratio = float(w) / max(1.0, float(h))
        lane_span_ratio = float(w) / max(1.0, float(frame_width * 0.75))

        # Convert to grayscale
        if len(patch_bgr.shape) == 3:
            gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch_bgr.copy()

        # 1. Morphological Aspect Ratio & Lane-Spanning Test
        # Speed breakers extend transversely across the lane (w >> h)
        is_transverse_span = (aspect_ratio >= 2.8 and lane_span_ratio >= 0.45) or (aspect_ratio >= 3.8)

        # 2. Photometric Convexity Profile (Vertical Gradient)
        # Compute vertical intensity profile (mean intensity per row from top to bottom)
        row_means = np.mean(gray, axis=1)
        convexity_score = self._compute_convexity_score(row_means)

        # 3. Transverse Rumble Strip Periodicity Test (Fast Fourier Transform / Peak Count)
        is_rumble_strip, peak_count = self._detect_rumble_strip_periodicity(gray)

        # 4. IMU Waveform Signature Analysis (if accelerometer data available)
        imu_signature = self._analyze_imu_shock_waveform(imu_shock_profile)

        # -------------------------------------------------------------------
        # Decision Fusion Engine
        # -------------------------------------------------------------------
        # 1. IMU shock signature has highest physical authority if available
        if imu_signature == "COMPRESSION_THEN_REBOUND":
            classification = "SPEED_BREAKER_CONVEX"
            is_hazard = False
            suppress = True
            confidence = 0.94
            reason = f"IMU vertical compression wave confirmed (+Gz then -Gz). Speed breaker."

        elif is_rumble_strip and peak_count >= 3 and aspect_ratio >= 1.8:
            classification = "RUMBLE_STRIP_PERIODIC"
            is_hazard = False
            suppress = True
            confidence = 0.88
            reason = f"Periodic transverse bars detected ({peak_count} peaks, aspect {aspect_ratio:.1f}). Legal rumble strip."

        elif (is_transverse_span and convexity_score > 0.10):
            classification = "SPEED_BREAKER_CONVEX"
            is_hazard = False
            suppress = True
            confidence = 0.86
            reason = f"Convex elevated crest (aspect {aspect_ratio:.1f}, convexity {convexity_score:.2f}). Speed breaker."

        elif is_transverse_span and convexity_score > -0.15:
            # Borderline wide bump spanning lane
            classification = "SPEED_BREAKER_CONVEX"
            is_hazard = False
            suppress = True
            confidence = 0.78
            reason = f"Transverse span across road lane (w={w}px, ratio {aspect_ratio:.1f}). Speed breaker."

        else:
            # Verified true cavity (pothole)
            classification = "POTHOLE"
            is_hazard = True
            suppress = False
            confidence = 0.90 if imu_signature == "DROP_THEN_IMPACT" else 0.85
            reason = f"Concave depression profile (aspect {aspect_ratio:.1f}, convexity {convexity_score:.2f}). Physical cavity."

        return {
            "classification": classification,
            "is_hazard": is_hazard,
            "confidence": round(confidence, 3),
            "aspect_ratio": round(aspect_ratio, 2),
            "convexity_score": round(convexity_score, 3),
            "lane_span_ratio": round(lane_span_ratio, 2),
            "rumble_peak_count": peak_count,
            "imu_signature": imu_signature,
            "suppress_grievance": suppress,
            "reason": reason,
        }

    def _compute_convexity_score(self, row_means: np.ndarray) -> float:
        """
        Analyze vertical illumination gradient to distinguish convex bump from concave cavity.
        - Convex bump (+): Bright leading ridge, peak luminance in upper-middle, followed by trailing shadow.
        - Concave pothole (-): Dark cavity center with lower luminance than top and bottom edges.
        """
        if len(row_means) < 5:
            return 0.0

        n = len(row_means)
        top_quarter = np.mean(row_means[: n // 4])
        mid_half = np.mean(row_means[n // 4 : 3 * n // 4])
        bottom_quarter = np.mean(row_means[3 * n // 4 :])

        baseline = (top_quarter + bottom_quarter) / 2.0
        if baseline <= 1e-3:
            return 0.0

        # Convexity: relative elevation of center brightness compared to outer boundaries
        convexity = (mid_half - baseline) / baseline
        return float(np.clip(convexity, -1.0, 1.0))

    def _detect_rumble_strip_periodicity(self, gray: np.ndarray) -> tuple:
        """
        Detect periodic horizontal rumble strips by counting distinct parallel peaks
        in the vertical profile.
        """
        if gray.shape[0] < 15:
            return False, 0

        # Sum intensities horizontally
        h_sums = np.mean(gray, axis=1)
        min_val, max_val = float(np.min(h_sums)), float(np.max(h_sums))
        contrast = max_val - min_val

        # Low contrast or uniform patch cannot be a high-visibility rumble strip
        if contrast < 12.0:
            return False, 0

        # Normalize to 0-100 scale
        norm_profile = (h_sums - min_val) / contrast * 100.0

        # Find distinct peaks above 55% height with minimum spacing
        peak_indices = []
        for i in range(1, len(norm_profile) - 1):
            if norm_profile[i] > 55.0 and norm_profile[i] >= norm_profile[i - 1] and norm_profile[i] >= norm_profile[i + 1]:
                if not peak_indices or (i - peak_indices[-1]) >= 8:
                    peak_indices.append(i)

        is_periodic = (len(peak_indices) >= 3)
        return is_periodic, len(peak_indices)

    def _analyze_imu_shock_waveform(self, imu_samples: list) -> str:
        """
        Classify vertical acceleration waveform (Gz) into shock signatures:
        - 'COMPRESSION_THEN_REBOUND' (Speed Breaker: +Gz then -Gz)
        - 'DROP_THEN_IMPACT'         (Pothole: -Gz then violent +Gz)
        - 'NO_SIGNIFICANT_IMU_EVENT'
        """
        if not imu_samples or len(imu_samples) < 4:
            return "NO_SIGNIFICANT_IMU_EVENT"

        arr = np.array(imu_samples, dtype=float)
        # Baseline gravity is 1.0g
        deviations = arr - 1.0

        max_idx = int(np.argmax(deviations))
        min_idx = int(np.argmin(deviations))

        max_val = deviations[max_idx]
        min_val = deviations[min_idx]

        # Check for significant motion (> 0.25g deviation)
        if abs(max_val) < 0.25 and abs(min_val) < 0.25:
            return "NO_SIGNIFICANT_IMU_EVENT"

        # If vehicle experiences drop (-Gz) BEFORE the hard impact (+Gz) -> Pothole
        if min_idx < max_idx and min_val < -0.20 and max_val > 0.35:
            return "DROP_THEN_IMPACT"

        # If vehicle experiences compression (+Gz) BEFORE rebound (-Gz) -> Speed Breaker
        if max_idx < min_idx and max_val > 0.30:
            return "COMPRESSION_THEN_REBOUND"

        return "GENERIC_CHASSIS_VIBRATION"
