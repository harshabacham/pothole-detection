"""
src/features.py
===============
Main feature extraction module. Combines intensity, texture (LBP + GLCM),
edge, and color features into a single numerical feature vector per patch.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.texture_features import extract_lbp_features, extract_glcm_features


def extract_intensity_features(gray_patch: np.ndarray) -> dict:
    """
    Compute basic intensity statistics from a grayscale patch.

    Parameters
    ----------
    gray_patch : np.ndarray
        2D uint8 array representing the grayscale patch.

    Returns
    -------
    dict
        Dictionary containing intensity statistics.
    """
    default_res = {
        "intensity_mean": 0.0,
        "intensity_std": 0.0,
        "intensity_var": 0.0,
        "intensity_min": 0.0,
        "intensity_max": 0.0,
        "intensity_range": 0.0,
        "intensity_skew": 0.0,
        "intensity_kurtosis": 0.0,
    }

    if gray_patch is None or gray_patch.size == 0:
        return default_res

    try:
        arr = gray_patch.astype(np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        var_val = float(np.var(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        range_val = max_val - min_val

        if std_val > 1e-6:
            skew_val = float(stats.skew(arr.ravel()))
            kurt_val = float(stats.kurtosis(arr.ravel()))
            if np.isnan(skew_val) or np.isinf(skew_val):
                skew_val = 0.0
            if np.isnan(kurt_val) or np.isinf(kurt_val):
                kurt_val = 0.0
        else:
            skew_val = 0.0
            kurt_val = 0.0

        return {
            "intensity_mean": mean_val,
            "intensity_std": std_val,
            "intensity_var": var_val,
            "intensity_min": min_val,
            "intensity_max": max_val,
            "intensity_range": range_val,
            "intensity_skew": skew_val,
            "intensity_kurtosis": kurt_val,
        }
    except Exception as exc:
        print(f"[WARNING] features: Intensity extraction failed: {exc}")
        return default_res


def extract_edge_features(gray_patch: np.ndarray) -> dict:
    """
    Compute edge-based features using Canny edge detector and Sobel gradients.

    Parameters
    ----------
    gray_patch : np.ndarray
        2D uint8 array representing the grayscale patch.

    Returns
    -------
    dict
        Dictionary containing edge density, Sobel statistics, and gradient energy.
    """
    default_res = {
        "edge_density": 0.0,
        "sobel_mean": 0.0,
        "sobel_std": 0.0,
        "sobel_max": 0.0,
        "gradient_energy": 0.0,
    }

    if gray_patch is None or gray_patch.size == 0:
        return default_res

    try:
        patch_u8 = np.clip(gray_patch, 0, 255).astype(np.uint8)

        # Canny edge density
        canny = cv2.Canny(patch_u8, config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2)
        edge_density = float(np.count_nonzero(canny)) / float(patch_u8.size)

        # Sobel gradients
        patch_f32 = patch_u8.astype(np.float32)
        sobel_x = cv2.Sobel(patch_f32, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch_f32, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        sobel_mean = float(np.mean(grad_mag))
        sobel_std = float(np.std(grad_mag))
        sobel_max = float(np.max(grad_mag))
        grad_energy = float(np.mean(grad_mag ** 2))

        return {
            "edge_density": edge_density,
            "sobel_mean": sobel_mean,
            "sobel_std": sobel_std,
            "sobel_max": sobel_max,
            "gradient_energy": grad_energy,
        }
    except Exception as exc:
        print(f"[WARNING] features: Edge extraction failed: {exc}")
        return default_res


def extract_color_features(bgr_patch: np.ndarray) -> dict:
    """
    Compute color features from the original BGR patch.

    Parameters
    ----------
    bgr_patch : np.ndarray
        3D uint8 array representing the BGR color patch.

    Returns
    -------
    dict
        Dictionary containing BGR and HSV channel statistics.
    """
    default_res = {
        "color_b_mean": 0.0, "color_b_std": 0.0,
        "color_g_mean": 0.0, "color_g_std": 0.0,
        "color_r_mean": 0.0, "color_r_std": 0.0,
        "color_h_mean": 0.0, "color_h_std": 0.0,
        "color_s_mean": 0.0, "color_s_std": 0.0,
        "color_v_mean": 0.0, "color_v_std": 0.0,
    }

    if bgr_patch is None or bgr_patch.size == 0 or bgr_patch.ndim != 3:
        return default_res

    try:
        b_ch = bgr_patch[:, :, 0].astype(np.float64)
        g_ch = bgr_patch[:, :, 1].astype(np.float64)
        r_ch = bgr_patch[:, :, 2].astype(np.float64)

        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        h_ch = hsv_patch[:, :, 0].astype(np.float64)
        s_ch = hsv_patch[:, :, 1].astype(np.float64)
        v_ch = hsv_patch[:, :, 2].astype(np.float64)

        return {
            "color_b_mean": float(np.mean(b_ch)), "color_b_std": float(np.std(b_ch)),
            "color_g_mean": float(np.mean(g_ch)), "color_g_std": float(np.std(g_ch)),
            "color_r_mean": float(np.mean(r_ch)), "color_r_std": float(np.std(r_ch)),
            "color_h_mean": float(np.mean(h_ch)), "color_h_std": float(np.std(h_ch)),
            "color_s_mean": float(np.mean(s_ch)), "color_s_std": float(np.std(s_ch)),
            "color_v_mean": float(np.mean(v_ch)), "color_v_std": float(np.std(v_ch)),
        }
    except Exception as exc:
        print(f"[WARNING] features: Color extraction failed: {exc}")
        return default_res


def extract_all_features(
    gray_patch: np.ndarray,
    bgr_patch: np.ndarray,
    patch_meta: dict,
) -> dict:
    """
    Combine all feature groups into a single flat dictionary.

    Parameters
    ----------
    gray_patch : np.ndarray
        Grayscale patch (H x W, uint8).
    bgr_patch : np.ndarray
        Color patch (H x W x 3, uint8).
    patch_meta : dict
        Metadata dictionary for the patch.

    Returns
    -------
    dict
        Combined flat dictionary of metadata and numeric features.
    """
    res = {
        "patch_id": patch_meta.get("patch_id", ""),
        "image_filename": patch_meta.get("image_filename", ""),
        "patch_size": patch_meta.get("patch_size", config.FEATURE_PATCH_SIZE),
        "row_idx": patch_meta.get("row_idx", 0),
        "col_idx": patch_meta.get("col_idx", 0),
        "x1": patch_meta.get("x1", 0),
        "y1": patch_meta.get("y1", 0),
        "x2": patch_meta.get("x2", 0),
        "y2": patch_meta.get("y2", 0),
    }

    # Extract all feature categories
    res.update(extract_intensity_features(gray_patch))
    res.update(extract_lbp_features(gray_patch))
    res.update(extract_glcm_features(gray_patch))
    res.update(extract_edge_features(gray_patch))
    res.update(extract_color_features(bgr_patch))

    return res


def extract_features_from_metadata_csv(
    metadata_csv_path: str,
    output_csv_path: str,
) -> pd.DataFrame:
    """
    Load patch metadata CSV, load each patch from disk, extract feature vectors,
    and save the combined feature dataset.

    Parameters
    ----------
    metadata_csv_path : str
        Path to the patch metadata CSV.
    output_csv_path : str
        Path where the extracted features CSV will be saved.

    Returns
    -------
    pd.DataFrame
        Extracted features DataFrame.
    """
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv_path}")

    df_meta = pd.read_csv(metadata_csv_path)
    print(f"[INFO] features: Loaded {len(df_meta)} rows from {metadata_csv_path}")

    records = []
    total_patches = len(df_meta)

    for idx, row in df_meta.iterrows():
        gray_p = row.get("gray_path", "")
        color_p = row.get("color_path", "")

        gray_patch = None
        bgr_patch = None

        if isinstance(gray_p, str) and os.path.exists(gray_p):
            gray_patch = cv2.imread(gray_p, cv2.IMREAD_GRAYSCALE)
        if isinstance(color_p, str) and os.path.exists(color_p):
            bgr_patch = cv2.imread(color_p, cv2.IMREAD_COLOR)

        if gray_patch is None and bgr_patch is not None:
            gray_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
        elif bgr_patch is None and gray_patch is not None:
            bgr_patch = cv2.cvtColor(gray_patch, cv2.COLOR_GRAY2BGR)

        meta_dict = row.to_dict()
        feat_dict = extract_all_features(gray_patch, bgr_patch, meta_dict)
        records.append(feat_dict)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_patches:
            print(f"[INFO] features: Extracted features for {idx + 1}/{total_patches} patches")

    features_df = pd.DataFrame(records)

    # Sanitize NaN and Inf
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    nan_count = features_df[numeric_cols].isna().sum().sum()
    inf_count = np.isinf(features_df[numeric_cols].to_numpy()).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"[WARNING] features: Found {nan_count} NaNs and {inf_count} Infs. Replacing with 0.0.")
        features_df[numeric_cols] = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    features_df.to_csv(output_csv_path, index=False)
    print(f"[INFO] features: Successfully saved {len(features_df)} feature vectors to {output_csv_path}")

    return features_df

