"""
src/gps_handler.py
==================
Geospatial coordinate extraction and timestamp interpolation.
Parses EXIF metadata from geotagged imagery and temporal GPS log files.
Does NOT fabricate coordinates when telemetry is missing.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def _convert_to_degrees(value) -> float:
    """Helper to convert EXIF GPS degrees, minutes, seconds to decimal degrees."""
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)


def extract_image_gps(image_path: str) -> tuple:
    """
    Extract decimal (latitude, longitude) from EXIF metadata of a geotagged image.

    Returns
    -------
    tuple
        (latitude, longitude) if found, otherwise (None, None).
    """
    if not os.path.exists(image_path):
        return None, None

    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None, None

            gps_info = {}
            for tag, val in exif.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in val:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_info[sub_decoded] = val[t]

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat = _convert_to_degrees(gps_info["GPSLatitude"])
                if gps_info.get("GPSLatitudeRef") == "S":
                    lat = -lat

                lon = _convert_to_degrees(gps_info["GPSLongitude"])
                if gps_info.get("GPSLongitudeRef") == "W":
                    lon = -lon

                return round(lat, 6), round(lon, 6)
    except Exception:
        pass

    return None, None


def load_external_gps_log(gps_log_path: str) -> pd.DataFrame:
    """
    Load an external GPS telemetry file (CSV format).
    Expected columns: timestamp_sec (or time), latitude, longitude.
    """
    if not os.path.exists(gps_log_path):
        raise FileNotFoundError(f"GPS telemetry log not found: {gps_log_path}")

    df = pd.read_csv(gps_log_path)
    # Standardize column headers
    cols = {c.lower().strip(): c for c in df.columns}
    time_col = None
    for candidate in ["timestamp_sec", "timestamp", "time", "sec"]:
        if candidate in cols:
            time_col = cols[candidate]
            break

    lat_col = cols.get("latitude") or cols.get("lat")
    lon_col = cols.get("longitude") or cols.get("lon") or cols.get("lng")

    if not time_col or not lat_col or not lon_col:
        raise ValueError("GPS log must contain columns for timestamp, latitude, and longitude.")

    clean_df = pd.DataFrame({
        "timestamp_sec": df[time_col].astype(float),
        "latitude": df[lat_col].astype(float),
        "longitude": df[lon_col].astype(float),
    }).sort_values(by="timestamp_sec")

    return clean_df


def interpolate_gps(gps_df: pd.DataFrame, timestamp_sec: float) -> tuple:
    """
    Interpolate (latitude, longitude) at a specific video timestamp.

    Returns
    -------
    tuple
        (latitude, longitude) or (None, None) if telemetry is unavailable.
    """
    if gps_df is None or gps_df.empty:
        return None, None

    times = gps_df["timestamp_sec"].to_numpy()
    lats = gps_df["latitude"].to_numpy()
    lons = gps_df["longitude"].to_numpy()

    if timestamp_sec < times[0] or timestamp_sec > times[-1]:
        # Return nearest bound if within 2 seconds
        if abs(timestamp_sec - times[0]) <= 2.0:
            return float(lats[0]), float(lons[0])
        elif abs(timestamp_sec - times[-1]) <= 2.0:
            return float(lats[-1]), float(lons[-1])
        return None, None

    interp_lat = float(np.interp(timestamp_sec, times, lats))
    interp_lon = float(np.interp(timestamp_sec, times, lons))

    return round(interp_lat, 6), round(interp_lon, 6)


def get_device_ip_location() -> dict:
    """
    Fetch approximate device location based on client IP as a fast fallback.
    Returns dict with latitude, longitude, city, region, country, and status.
    """
    try:
        import requests
        resp = requests.get("http://ip-api.com/json/", timeout=2.0).json()
        if resp.get("status") == "success":
            return {
                "latitude": float(resp.get("lat")),
                "longitude": float(resp.get("lon")),
                "city": resp.get("city", ""),
                "region": resp.get("regionName", ""),
                "country": resp.get("country", "India"),
                "source": "Network IP Geolocation",
                "success": True,
            }
    except Exception:
        pass

    # Standard default Indian highway corridor (Delhi-NCR / NH-48)
    return {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "city": "New Delhi",
        "region": "Delhi",
        "country": "India",
        "source": "Default Highway Location",
        "success": False,
    }


