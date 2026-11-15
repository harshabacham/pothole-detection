"""
src/map_visualization.py
========================
Geographic defect map and spatial heatmap generator using Folium.
Renders clustered severity markers, defect popups, and density layers.
"""

import os
import sys
import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def get_marker_color(severity_category: str) -> str:
    """Return Folium marker color based on severity."""
    cat = str(severity_category).capitalize()
    if cat == "High":
        return "red"
    elif cat == "Medium":
        return "orange"
    else:
        return "green"


def build_geographic_map(
    defects_df: pd.DataFrame,
    center: list = None,
    zoom_start: int = None,
    severity_filter: list = None,
) -> folium.Map:
    """
    Construct a Folium interactive map with clustered defect markers and a density heatmap.

    Parameters
    ----------
    defects_df : pd.DataFrame
        DataFrame with columns 'latitude', 'longitude', 'severity_category',
        'severity_score', 'detection_id', 'timestamp'.
    center : list, optional
        [latitude, longitude] center.
    zoom_start : int, optional
        Initial zoom level.
    severity_filter : list, optional
        List of categories to include, e.g. ['Low', 'Medium', 'High'].

    Returns
    -------
    folium.Map
        Interactive Folium map object.
    """
    if zoom_start is None:
        zoom_start = config.DEFAULT_MAP_ZOOM

    # Filter invalid GPS
    valid_df = defects_df.dropna(subset=["latitude", "longitude"]).copy() if not defects_df.empty else pd.DataFrame()

    if severity_filter and not valid_df.empty:
        valid_df = valid_df[valid_df["severity_category"].isin(severity_filter)]

    if not valid_df.empty:
        mean_lat = float(valid_df["latitude"].mean())
        mean_lon = float(valid_df["longitude"].mean())
        map_center = [mean_lat, mean_lon]
    else:
        map_center = center if center is not None else config.DEFAULT_MAP_CENTER

    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    if valid_df.empty:
        folium.Marker(
            location=map_center,
            popup="<i>No verified GPS defect records logged yet.</i>",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)
        return m

    # 1. Clustered Markers Layer
    marker_cluster = MarkerCluster(name="Defect Clusters").add_to(m)

    for _, row in valid_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cat = str(row.get("severity_category", "Low"))
        score = float(row.get("severity_score", 0.0))
        det_id = str(row.get("detection_id", "N/A"))
        t_stamp = str(row.get("timestamp", "N/A"))
        media = str(row.get("source_media", "N/A"))

        color = get_marker_color(cat)

        popup_html = f"""
        <div style="font-family:sans-serif; min-width:180px;">
            <h5 style="margin:0 0 5px 0; color:#2c3e50;">{det_id}</h5>
            <b>Severity:</b> <span style="color:{color}; font-weight:bold;">{cat}</span> (Score: {score:.2f})<br>
            <b>Time:</b> {t_stamp}<br>
            <b>Source:</b> {media}<br>
            <b>Coordinates:</b> {lat:.5f}, {lon:.5f}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{det_id} [{cat} Severity]",
        ).add_to(marker_cluster)

    # 2. Thermal Density Heatmap Layer
    heat_data = [
        [float(r["latitude"]), float(r["longitude"]), float(r.get("severity_score", 0.5))]
        for _, r in valid_df.iterrows()
    ]
    if heat_data:
        HeatMap(
            heat_data,
            name="Severity Density Heatmap",
            min_opacity=0.3,
            radius=18,
            blur=12,
            max_zoom=16,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def save_map_to_html(map_obj: folium.Map, save_path: str = None) -> str:
    """Save folium map to HTML document."""
    if save_path is None:
        save_path = os.path.join(config.MAPS_OUTPUT_DIR, "defect_map.html")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    map_obj.save(save_path)
    print(f"[INFO] map_visualization: Saved interactive map to: {save_path}")
    return save_path

