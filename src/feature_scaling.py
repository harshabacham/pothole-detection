"""
src/feature_scaling.py
=======================
Feature scaling utilities. Fits a scaler on training data only and
provides reusable transform/inverse-transform.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config

META_COLUMNS = [
    "patch_id",
    "image_filename",
    "patch_size",
    "row_idx",
    "col_idx",
    "x1",
    "y1",
    "x2",
    "y2",
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of numeric feature columns excluding metadata.
    """
    return [col for col in df.columns if col not in META_COLUMNS and np.issubdtype(df[col].dtype, np.number)]


def fit_scaler(features_df: pd.DataFrame, scaler_save_path: str = None) -> tuple:
    """
    Fit a RobustScaler on the numeric feature columns of features_df.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing metadata and numeric feature columns.
    scaler_save_path : str, optional
        File path to save the fitted scaler object using joblib.

    Returns
    -------
    tuple
        (fitted_scaler, feature_columns_list)
    """
    feat_cols = get_feature_columns(features_df)
    if not feat_cols:
        raise ValueError("No numeric feature columns found to scale.")

    scaler = RobustScaler()
    scaler.fit(features_df[feat_cols].to_numpy())

    if scaler_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(scaler_save_path)), exist_ok=True)
        joblib.dump({"scaler": scaler, "feature_columns": feat_cols}, scaler_save_path)
        print(f"[INFO] feature_scaling: Saved fitted scaler to {scaler_save_path}")

    return scaler, feat_cols


def transform_features(features_df: pd.DataFrame, scaler, feature_cols: list = None) -> pd.DataFrame:
    """
    Transform numeric feature columns using a fitted RobustScaler.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing features to transform.
    scaler : RobustScaler
        Pre-fitted scaler.
    feature_cols : list, optional
        List of feature columns to transform. If None, derived automatically.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled features and untouched metadata columns.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    scaled_df = features_df.copy()
    scaled_vals = scaler.transform(scaled_df[feature_cols].to_numpy())
    scaled_df[feature_cols] = scaled_vals
    return scaled_df


def load_scaler(scaler_path: str):
    """
    Load a fitted scaler and feature column list from disk.

    Parameters
    ----------
    scaler_path : str
        Path to the saved scaler joblib file.

    Returns
    -------
    tuple
        (scaler, feature_columns_list)
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    data = joblib.load(scaler_path)
    if isinstance(data, dict) and "scaler" in data:
        return data["scaler"], data["feature_columns"]
    return data, None


def fit_and_save(features_csv: str, scaled_csv: str, scaler_path: str) -> tuple:
    """
    Full scaling pipeline:
    1. Load raw features CSV.
    2. Fit RobustScaler on numeric features.
    3. Transform features.
    4. Save scaled features CSV.
    5. Save fitted scaler artifact.

    Returns
    -------
    tuple
        (scaled_df, scaler, feature_columns)
    """
    df = pd.read_csv(features_csv)
    print(f"[INFO] feature_scaling: Loaded {len(df)} rows from {features_csv}")

    scaler, feat_cols = fit_scaler(df, scaler_save_path=scaler_path)
    scaled_df = transform_features(df, scaler, feature_cols=feat_cols)

    os.makedirs(os.path.dirname(os.path.abspath(scaled_csv)), exist_ok=True)
    scaled_df.to_csv(scaled_csv, index=False)
    print(f"[INFO] feature_scaling: Saved scaled features to {scaled_csv}")

    return scaled_df, scaler, feat_cols

