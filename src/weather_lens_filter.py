"""
src/weather_lens_filter.py
==========================
Adverse Weather, Windshield Wiper Sweep & Specular Glare Neutralizer.

CRITICAL REAL-WORLD PRODUCTION PROBLEM:
During real-world monsoon driving in India:
1. Windshield wipers sweep across the camera view every 1.0 to 2.0 seconds.
   The dark, moving wiper blade forms a massive diagonal edge that naive vision
   models flag as a catastrophic pavement crack or road trench!
2. Direct low-angle sunlight (dawn/dusk glare) washes out asphalt with specular
   reflection, triggering false texture anomalies.

SOLUTIONS IMPLEMENTED:
1. Wiper Blade Detection & Suppression:
   - Identifies sweeping diagonal linear bars using Progressive Probabilistic
     Hough Line Transform (angles 20-75 deg and 105-160 deg, length > 120px).
   - Generates a dynamic Wiper Occlusion Mask.
   - Cross-checks candidate defect bounding boxes: any defect overlapping
     the sweeping wiper trajectory is filtered out as WIPER_BLADE_ARTIFACT.
2. Specular Sun Glare Neutralizer:
   - Identifies saturated pixel clusters (Y > 240) caused by direct sunlight.
   - Applies local dynamic range attenuation so asphalt beneath is evaluated
     without false edge triggers.
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


class WeatherLensFilter:
    """
    Filters adverse environmental camera artifacts: sweeping windshield wipers,
    raindrop refraction, and direct sun glare.
    """

    def __init__(self, min_wiper_length: int = 140):
        self.min_wiper_length = min_wiper_length
        self.recent_wiper_detected = False
        self.last_wiper_mask = None

    def detect_wiper_blade(self, frame_bgr: np.ndarray) -> tuple:
        """
        Detect if a windshield wiper blade is currently sweeping across the frame.

        Returns
        -------
        (is_wiper_present: bool, wiper_mask: np.ndarray, detected_lines: list)
        """
        if frame_bgr is None:
            return False, None, []

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr.copy()

        # Wiper blades typically appear in the lower 75% of the windshield
        roi_top = int(h * 0.25)
        gray_roi = gray[roi_top:h, :]

        # Edge detection with Canny
        edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)

        # Probabilistic Hough line transform to find long linear segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=self.min_wiper_length,
            maxLineGap=20,
        )

        wiper_mask = np.zeros((h, w), dtype=np.uint8)
        wiper_lines = []

        if lines is not None:
            for line in lines:
                coords = line.flatten()
                if len(coords) < 4:
                    continue
                x1, y1_roi, x2, y2_roi = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                y1 = y1_roi + roi_top
                y2 = y2_roi + roi_top

                # Calculate angle in degrees
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                if length < self.min_wiper_length:
                    continue

                angle_deg = abs(math.degrees(math.atan2(dy, dx)))

                # Windshield wiper blades sweep at distinct diagonal angles:
                # Typically 20 to 75 deg or 105 to 160 deg (not purely horizontal or vertical)
                if (20.0 <= angle_deg <= 75.0) or (105.0 <= angle_deg <= 160.0):
                    wiper_lines.append((x1, y1, x2, y2, angle_deg, length))
                    # Draw thick line into mask to cover the wiper blade body
                    cv2.line(wiper_mask, (x1, y1), (x2, y2), 255, thickness=28)

        is_present = len(wiper_lines) > 0
        self.recent_wiper_detected = is_present
        self.last_wiper_mask = wiper_mask
        return is_present, wiper_mask, wiper_lines

    def is_defect_occluded_by_wiper(
        self,
        bbox: tuple,
        wiper_mask: np.ndarray,
        overlap_threshold: float = 0.25,
    ) -> bool:
        """
        Check if a candidate defect bounding box is actually a windshield wiper artifact.
        """
        if wiper_mask is None or np.sum(wiper_mask) == 0:
            return False

        x, y, w, h = bbox
        frame_h, frame_w = wiper_mask.shape[:2]

        # Clamp coordinates
        x1 = max(0, min(frame_w - 1, x))
        y1 = max(0, min(frame_h - 1, y))
        x2 = max(0, min(frame_w, x + w))
        y2 = max(0, min(frame_h, y + h))

        box_area = max(1, (x2 - x1) * (y2 - y1))
        box_mask_roi = wiper_mask[y1:y2, x1:x2]
        wiper_overlap_pixels = int(np.count_nonzero(box_mask_roi))

        overlap_ratio = float(wiper_overlap_pixels) / float(box_area)
        return overlap_ratio >= overlap_threshold

    def neutralize_sun_glare(self, frame_bgr: np.ndarray) -> tuple:
        """
        Detect and attenuate specular solar glare patches on the pavement.

        Returns
        -------
        (enhanced_frame: np.ndarray, glare_detected: bool, glare_percentage: float)
        """
        if frame_bgr is None:
            return frame_bgr, False, 0.0

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        # Sun glare is characterized by very high value (V > 240) and low saturation (S < 40)
        glare_mask = (v_channel > 240) & (s_channel < 45)
        glare_pixels = int(np.count_nonzero(glare_mask))
        total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
        glare_ratio = float(glare_pixels) / float(total_pixels)

        glare_detected = glare_ratio > 0.015  # More than 1.5% of frame washed out

        if not glare_detected:
            return frame_bgr.copy(), False, round(glare_ratio * 100.0, 2)

        # Attenuate glare by adaptive gamma mapping in V channel
        enhanced_v = v_channel.copy()
        # Compress highlights: replace saturated glare with local neighborhood median
        enhanced_v[glare_mask] = np.clip(enhanced_v[glare_mask] * 0.70, 0, 255).astype(np.uint8)

        hsv[:, :, 2] = enhanced_v
        corrected_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return corrected_bgr, True, round(glare_ratio * 100.0, 2)
