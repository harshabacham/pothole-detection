"""
src/texture_features.py
========================
Texture feature extraction using Local Binary Patterns (LBP) and
Gray Level Co-occurrence Matrix (GLCM).
"""

import os
import sys
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def extract_lbp_features(gray_patch: np.ndarray) -> dict:
    """
    Compute LBP histogram and statistical features for a grayscale patch.

    Parameters
    ----------
    gray_patch : np.ndarray
        2D uint8 array representing the grayscale patch.

    Returns
    -------
    dict
        Feature dictionary containing lbp_mean, lbp_std, lbp_energy,
        lbp_entropy, and individual histogram bin values.
    """
    n_bins = int(config.LBP_N_POINTS + 2) if config.LBP_METHOD == "uniform" else int(2 ** config.LBP_N_POINTS)
    default_res = {
        "lbp_mean": 0.0,
        "lbp_std": 0.0,
        "lbp_energy": 0.0,
        "lbp_entropy": 0.0,
    }
    for b in range(n_bins):
        default_res[f"lbp_hist_{b}"] = 0.0

    if gray_patch is None or gray_patch.size == 0:
        return default_res

    try:
        patch_u8 = np.clip(gray_patch, 0, 255).astype(np.uint8)
        lbp = local_binary_pattern(
            patch_u8,
            P=config.LBP_N_POINTS,
            R=config.LBP_RADIUS,
            method=config.LBP_METHOD,
        )

        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        else:
            hist = np.zeros(n_bins, dtype=np.float64)

        lbp_mean = float(np.mean(lbp))
        lbp_std = float(np.std(lbp))
        lbp_energy = float(np.sum(hist ** 2))
        lbp_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))

        res = {
            "lbp_mean": lbp_mean,
            "lbp_std": lbp_std,
            "lbp_energy": lbp_energy,
            "lbp_entropy": lbp_entropy,
        }
        for b in range(n_bins):
            res[f"lbp_hist_{b}"] = float(hist[b])

        return res

    except Exception as exc:
        print(f"[WARNING] texture_features: LBP extraction failed: {exc}")
        return default_res


def extract_glcm_features(gray_patch: np.ndarray) -> dict:
    """
    Compute Gray Level Co-occurrence Matrix (GLCM) texture features.

    Parameters
    ----------
    gray_patch : np.ndarray
        2D uint8 array representing the grayscale patch.

    Returns
    -------
    dict
        Feature dictionary containing glcm_contrast, glcm_dissimilarity,
        glcm_homogeneity, glcm_energy, glcm_correlation, and glcm_asm.
    """
    default_res = {
        "glcm_contrast": 0.0,
        "glcm_dissimilarity": 0.0,
        "glcm_homogeneity": 0.0,
        "glcm_energy": 0.0,
        "glcm_correlation": 0.0,
        "glcm_asm": 0.0,
    }

    if gray_patch is None or gray_patch.size == 0:
        return default_res

    try:
        patch_u8 = np.clip(gray_patch, 0, 255).astype(np.uint8)
        glcm = graycomatrix(
            patch_u8,
            distances=config.GLCM_DISTANCES,
            angles=config.GLCM_ANGLES,
            levels=config.GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )

        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
        res = {}
        for prop in props:
            vals = graycoprops(glcm, prop)
            val = float(np.nanmean(vals))
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            res[f"glcm_{prop.lower()}"] = val

        return res

    except Exception as exc:
        print(f"[WARNING] texture_features: GLCM extraction failed: {exc}")
        return default_res

