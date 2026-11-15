"""
src/night_vision.py
===================
Night-Vision Illumination Enhancement and Adverse Weather (Puddle Filter) Engine.

Capabilities:
1. Automatic low-light / night driving scene detector.
2. Adaptive illumination enhancer for dark asphalt road surfaces without headlight glare saturation.
3. Specular reflection & rain puddle disambiguation (distinguishes flat water reflections from deep structural potholes).
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


class NightVisionEnhancer:
    """
    Real-time night-vision processor and rain puddle discriminator.
    """

    def __init__(
        self,
        luminance_thresh: float = None,
        puddle_filter_active: bool = None,
    ):
        self.lum_thresh = luminance_thresh if luminance_thresh is not None else config.NIGHT_LUMINANCE_THRESHOLD
        self.puddle_filter = puddle_filter_active if puddle_filter_active is not None else config.PUDDLE_FILTER_ACTIVE

    def is_night_scene(self, frame_bgr: np.ndarray) -> tuple:
        """
        Detect whether the current driving frame is low-light / nighttime.

        Returns
        -------
        tuple: (bool is_night, float mean_luminance)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        is_night = (mean_lum < self.lum_thresh)
        return is_night, round(mean_lum, 1)

    def enhance_night_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Adaptively enhance dark asphalt pavement while suppressing headlight glare.
        Uses YCrCb color space with localized dynamic range compression.
        """
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 1. CLAHE on luminance channel with gentle clip limit
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)

        # 2. Gamma correction for shadow boost
        norm_y = y_eq.astype(np.float32) / 255.0
        # Dark regions boosted by gamma < 1.0, bright headlights protected
        gamma = 0.72
        y_gamma = np.power(norm_y, gamma) * 255.0
        y_enhanced = np.clip(y_gamma, 0, 255).astype(np.uint8)

        # 3. Bilateral filter to reduce camera sensor noise in low-light
        y_denoised = cv2.bilateralFilter(y_enhanced, d=5, sigmaColor=35, sigmaSpace=35)

        merged = cv2.merge([y_denoised, cr, cb])
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        return enhanced_bgr

    def is_surface_puddle(self, patch_bgr: np.ndarray) -> tuple:
        """
        Differentiate flat water puddles / reflections from real structural potholes.

        Water puddles:
        - Flat specular sheen with low internal structural variance.
        - High reflection of sky/lights.

        Potholes:
        - Sharp jagged perimeter edges.
        - Granular gravel texture inside the cavity.

        Returns
        -------
        tuple: (bool is_puddle, float specular_score, float texture_energy)
        """
        if not self.puddle_filter:
            return False, 0.0, 1.0

        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Internal texture energy (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Specular reflection highlight check (near-white pixels from water reflection)
        specular_pixels = np.sum(gray > 220)
        specular_ratio = float(specular_pixels) / float(max(1, h * w))

        # 3. Flatness of internal patch (standard deviation)
        std_dev = float(np.std(gray))

        # Puddle condition: High specular reflection or ultra-smooth mirror surface with low laplacian
        is_puddle = (specular_ratio > 0.12 and lap_var < 80.0) or (std_dev < 10.0 and lap_var < 40.0)

        return is_puddle, round(specular_ratio, 3), round(lap_var, 1)

