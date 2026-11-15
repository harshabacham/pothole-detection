"""
src/depth_estimator.py
======================
Monocular Metric Depth & Pothole Volumetric Repair Estimation Engine.

Computes:
1. Distance to defect ahead of the vehicle (meters) using ground plane projective geometry.
2. 3D cavity depression depth profile (centimeters).
3. Estimated pothole cavity volume (liters).
4. Required asphalt patch repair mass (kilograms) for PWD engineering maintenance.
5. Time-to-impact (TTI) for driver ADAS warning.
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


class MetricDepthEstimator:
    """
    Estimates real-world metric depth, distance, cavity depression,
    and repair material mass from vehicle monocular camera frames.
    """

    def __init__(
        self,
        camera_height_m: float = None,
        camera_pitch_deg: float = None,
        camera_hfov_deg: float = None,
        asphalt_density: float = None,
    ):
        self.height_m = camera_height_m if camera_height_m is not None else config.CAMERA_HEIGHT_METERS
        self.pitch_deg = camera_pitch_deg if camera_pitch_deg is not None else config.CAMERA_PITCH_DEG
        self.hfov_deg = camera_hfov_deg if camera_hfov_deg is not None else config.CAMERA_HFOV_DEG
        self.asphalt_density = asphalt_density if asphalt_density is not None else config.ASPHALT_DENSITY_KG_PER_LITER

    def compute_distance_ahead(self, y_pixel: int, image_height: int = 480, image_width: int = 640) -> float:
        """
        Compute metric distance in meters from car bumper to road pixel y.
        Uses flat-earth projective pinhole camera model.
        """
        # Focal length in pixels
        f_x = (image_width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))
        f_y = f_x  # square pixel assumption
        y_c = image_height / 2.0

        # Downward angle from camera optical axis to pixel y
        dy = y_pixel - y_c
        angle_from_axis_deg = math.degrees(math.atan2(dy, f_y))
        total_pitch_deg = self.pitch_deg + angle_from_axis_deg

        if total_pitch_deg <= 0.5:
            return 99.0  # horizon or above

        distance_m = self.height_m / math.tan(math.radians(total_pitch_deg))
        return max(1.0, min(80.0, distance_m))

    def profile_pothole_3d(
        self,
        frame_bgr: np.ndarray,
        cand: dict,
        vehicle_speed_kmh: float = 45.0,
    ) -> dict:
        """
        Compute complete 3D volumetric profile, depth in centimeters,
        asphalt repair mass in kg, and time-to-impact for a detected candidate.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = cand["x"], cand["y"], cand["width"], cand["height"]

        # Ensure bounds
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(4, min(w, w_img - x))
        h = max(4, min(h, h_img - y))

        # Bottom of bounding box represents contact point with road
        y_bottom = y + h
        dist_ahead_m = self.compute_distance_ahead(y_bottom, image_height=h_img, image_width=w_img)

        # Time-to-impact (TTI)
        speed_mps = max(1.0, (vehicle_speed_kmh * 1000.0) / 3600.0)
        tti_sec = dist_ahead_m / speed_mps

        # Crop patch & surrounding ring for baseline
        patch_bgr = frame_bgr[y:y + h, x:x + w]
        gray_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Surrounding road reference
        pad = min(16, min(x, y, w_img - (x + w), h_img - (y + h)))
        if pad > 2:
            outer = frame_bgr[y - pad:y + h + pad, x - pad:x + w + pad]
            outer_gray = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY).astype(np.float32)
            bg_mean = float(np.median(outer_gray))
        else:
            bg_mean = float(np.percentile(gray_patch, 85))

        bg_mean = max(10.0, bg_mean)

        # Photometric depth depression map
        # Darker cavity pixels and high edge gradients indicate depth depression
        sobel_x = cv2.Sobel(gray_patch, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_patch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        norm_grad = cv2.normalize(grad_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

        intensity_deficit = np.clip((bg_mean - gray_patch) / bg_mean, 0.0, 1.0)
        depression_norm = (0.75 * intensity_deficit) + (0.25 * norm_grad)

        # Geometric scale factor: pixels to real-world centimeters based on distance
        # 1 pixel at distance D covers approx (D * tan(FOV/W)) meters
        pixel_size_cm = (dist_ahead_m * math.tan(math.radians(self.hfov_deg / 2.0)) / (w_img / 2.0)) * 100.0
        pixel_area_cm2 = max(0.01, pixel_size_cm ** 2)

        # Physical depth calibration:
        # A fully depressed anomaly corresponds to 5 - 15 cm deep cavity
        depth_scale_cm = 8.5 * (1.0 + (cand.get("severity_score", 0.5) * 0.8))
        depth_map_cm = depression_norm * depth_scale_cm

        max_depth_cm = float(np.percentile(depth_map_cm, 98))
        mean_depth_cm = float(np.mean(depth_map_cm[depth_map_cm > 1.0])) if np.any(depth_map_cm > 1.0) else float(np.mean(depth_map_cm))

        # Real-world surface area in cm^2 and cavity volume in Liters (dm^3)
        pothole_mask = (depth_map_cm > 1.0)
        active_pixels = int(np.sum(pothole_mask))
        surface_area_cm2 = active_pixels * pixel_area_cm2

        # Volume = Sum(depth_cm * pixel_area_cm2) in cm^3 -> divide by 1000 for Liters
        volume_cm3 = np.sum(depth_map_cm[pothole_mask]) * pixel_area_cm2
        volume_liters = max(0.1, volume_cm3 / 1000.0)

        # Required repair mass (kg) = Volume (liters) * 2.4 kg/L
        repair_mass_kg = volume_liters * self.asphalt_density

        # Color-coded 3D depth map for visualization (Jet: Blue=surface, Red=deep cavity)
        depth_vis_norm = cv2.normalize(depth_map_cm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_JET)

        return {
            "distance_ahead_m": round(dist_ahead_m, 1),
            "time_to_impact_sec": round(tti_sec, 2),
            "max_depth_cm": round(max_depth_cm, 1),
            "mean_depth_cm": round(mean_depth_cm, 1),
            "surface_area_cm2": round(surface_area_cm2, 1),
            "cavity_volume_liters": round(volume_liters, 2),
            "asphalt_repair_mass_kg": round(repair_mass_kg, 2),
            "depth_colormap": depth_colormap,
            "depth_map_cm": depth_map_cm,
        }

