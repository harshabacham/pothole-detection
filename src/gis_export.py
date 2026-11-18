"""
src/gis_export.py
=================
Geographic Information Systems (GIS) & Smart City Export Engine.

Converts detected road anomalies, 3D metric telemetry, and SQLite defect logs into:
1. Standard OGC GeoJSON FeatureCollections for QGIS and ArcGIS.
2. 3D Google Earth KML files with depth-extruded polygonal pillars ($cm$)
   and rich HTML popup inspection cards.
3. Smart City Command and Control Center (ICCC) JSON feeds.
"""

import os
import sys
import json
import math
import html
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def export_defects_to_geojson(
    defects_df: pd.DataFrame,
    output_path: str = None,
) -> str:
    """
    Export defect records into a GeoJSON FeatureCollection.
    """
    if output_path is None:
        output_path = config.GEOJSON_OUTPUT_PATH

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    features = []
    valid_df = defects_df.dropna(subset=["latitude", "longitude"]).copy() if not defects_df.empty else pd.DataFrame()

    for _, row in valid_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cat = str(row.get("severity_category", "Low"))
        score = float(row.get("severity_score", 0.0))
        anom_score = float(row.get("anomaly_score", 0.0))
        det_id = str(row.get("detection_id", "N/A"))
        t_id = str(row.get("track_id", "N/A"))
        t_stamp = str(row.get("timestamp", datetime.now().isoformat()))
        media = str(row.get("source_media", "N/A"))
        area = int(row.get("pixel_area", 0))
        depth = float(row.get("max_depth_cm", 4.5)) if pd.notnull(row.get("max_depth_cm")) else 4.5
        vol = float(row.get("cavity_volume_liters", 2.8)) if pd.notnull(row.get("cavity_volume_liters")) else 2.8
        ref = str(row.get("grievance_ref_no") or row.get("reference_no") or "N/A")

        color_hex = "#27ae60" if cat == "Low" else ("#d35400" if cat == "Medium" else "#c0392b")

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "detection_id": det_id,
                "track_id": t_id,
                "grievance_ref": ref,
                "severity_category": cat,
                "severity_score": round(score, 4),
                "anomaly_score": round(anom_score, 4),
                "max_depth_cm": round(depth, 1),
                "cavity_volume_liters": round(vol, 1),
                "pixel_area": area,
                "timestamp": t_stamp,
                "source_media": media,
                "marker_color": color_hex,
                "marker_size": "medium",
            },
        }
        features.append(feature)

    geojson_doc = {
        "type": "FeatureCollection",
        "name": "Pothole_and_Road_Surface_Anomalies",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2)

    print(f"[INFO] gis_export: Exported {len(features)} defect features to GeoJSON: {output_path}")
    return output_path


def geojson_to_bytes(defects_data) -> bytes:
    """Serialize defect records or GeoJSON filepath into GeoJSON byte stream for in-browser download."""
    if isinstance(defects_data, bytes):
        return defects_data
    if isinstance(defects_data, str):
        if os.path.exists(defects_data):
            try:
                with open(defects_data, "rb") as f:
                    return f.read()
            except Exception:
                pass
        try:
            return defects_data.encode("utf-8")
        except Exception:
            pass
    if isinstance(defects_data, dict):
        return json.dumps(defects_data, indent=2).encode("utf-8")

    if not isinstance(defects_data, pd.DataFrame):
        return json.dumps({"type": "FeatureCollection", "features": []}, indent=2).encode("utf-8")

    defects_df = defects_data
    has_coords = "latitude" in defects_df.columns and "longitude" in defects_df.columns
    valid_df = defects_df.dropna(subset=["latitude", "longitude"]).copy() if (not defects_df.empty and has_coords) else pd.DataFrame()
    features = []

    for _, row in valid_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cat = str(row.get("severity_category", "Low"))
        score = float(row.get("severity_score", 0.0))
        det_id = str(row.get("detection_id", "N/A"))
        t_id = str(row.get("track_id", "N/A"))
        color_hex = "#27ae60" if cat == "Low" else ("#d35400" if cat == "Medium" else "#c0392b")

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "detection_id": det_id,
                "track_id": t_id,
                "severity": cat,
                "severity_score": round(score, 4),
                "marker_color": color_hex,
            },
        })

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features,
    }
    return json.dumps(geojson_doc, indent=2).encode("utf-8")


def export_defects_to_kml(
    defects_df: pd.DataFrame,
    output_path: str = None,
    extrude_3d: bool = True,
) -> str:
    """
    Export defect records into a 3D Google Earth KML document.

    If extrude_3d is True, creates 3D extruded polygonal cylinders where the
    altitude/height above pavement directly represents cavity depth in centimeters.
    """
    if output_path is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(config.OUTPUT_DIR, "pothole_3d_google_earth.kml")

    kml_str = _generate_kml_content(defects_df, extrude_3d=extrude_3d)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kml_str)

    print(f"[INFO] gis_export: Exported 3D Google Earth KML to: {output_path}")
    return output_path


def kml_to_bytes(defects_df: pd.DataFrame, extrude_3d: bool = True) -> bytes:
    """Serialize defect records into Google Earth KML byte stream."""
    return _generate_kml_content(defects_df, extrude_3d=extrude_3d).encode("utf-8")


def _generate_kml_content(defects_df: pd.DataFrame, extrude_3d: bool = True) -> str:
    """Internal helper to construct standard OGC KML with 3D extrusions."""
    has_coords = isinstance(defects_df, pd.DataFrame) and "latitude" in defects_df.columns and "longitude" in defects_df.columns
    valid_df = defects_df.dropna(subset=["latitude", "longitude"]).copy() if (has_coords and not defects_df.empty) else pd.DataFrame()

    kml_placemarks = []

    for _, row in valid_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cat = str(row.get("severity_category", "Low"))
        score = float(row.get("severity_score", 0.0))
        det_id = str(row.get("detection_id", "N/A"))
        t_id = str(row.get("track_id", "N/A"))
        ref_no = str(row.get("grievance_ref_no") or row.get("reference_no") or "UNREPORTED")
        depth_cm = float(row.get("max_depth_cm", 4.5)) if pd.notnull(row.get("max_depth_cm")) else 4.5
        vol_l = float(row.get("cavity_volume_liters", 2.8)) if pd.notnull(row.get("cavity_volume_liters")) else 2.8
        t_stamp = str(row.get("timestamp", datetime.now().isoformat()))

        # KML color format: aabbggrr (Alpha, Blue, Green, Red)
        if cat == "High":
            kml_color = "cc1400d9"  # Deep Red
            style_id = "potholeHigh"
        elif cat == "Medium":
            kml_color = "cc0070e6"  # Orange
            style_id = "potholeMedium"
        else:
            kml_color = "cc33a832"  # Green
            style_id = "potholeLow"

        # Scale height in meters for 3D visual extrusion in Google Earth
        # 1 cm depth = 1.5m visual pillar height so it pops on satellite terrain
        height_m = max(2.0, depth_cm * 1.5)

        # Generate a small 8-point 3D circular polygon around the GPS coordinate
        radius_deg = 0.00004  # ~4.5 meters radius
        coords_str = []
        for step in range(9):
            angle = step * (2.0 * math.pi / 8.0)
            d_lat = radius_deg * math.cos(angle)
            d_lon = radius_deg * math.sin(angle)
            coords_str.append(f"{lon + d_lon:.7f},{lat + d_lat:.7f},{height_m:.1f}")
        polygon_coords = " ".join(coords_str)

        desc_html = html.escape(f"""
            <div style="font-family: Arial, sans-serif; min-width: 250px;">
                <h3 style="margin-top:0; color:#c0392b;">🚨 Pothole Hazard Profile</h3>
                <p><b>Detection ID:</b> {det_id} (Track: {t_id})</p>
                <p><b>Grievance Ref:</b> <code>{ref_no}</code></p>
                <p><b>Severity Category:</b> <span style="font-weight:bold;">{cat.upper()}</span> (Score: {score:.2f})</p>
                <hr style="border:0; border-top:1px solid #ddd;"/>
                <p>📏 <b>Max Metric Depth:</b> {depth_cm:.1f} cm</p>
                <p>🪣 <b>Void Volume:</b> {vol_l:.1f} Liters</p>
                <p>📍 <b>GPS:</b> {lat:.6f}, {lon:.6f}</p>
                <p>🕒 <b>Timestamp:</b> {t_stamp}</p>
                <p><i>Autonomous Indian Road Safety Dashcam AI</i></p>
            </div>
        """)

        placemark = f"""
        <Placemark>
            <name>{html.escape(f"[{cat.upper()}] Pothole #{det_id}")}</name>
            <description><![CDATA[{desc_html}]]></description>
            <styleUrl>#{style_id}</styleUrl>
            <Polygon>
                <extrude>{1 if extrude_3d else 0}</extrude>
                <altitudeMode>relativeToGround</altitudeMode>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>{polygon_coords}</coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        </Placemark>
        """
        kml_placemarks.append(placemark)

    placemarks_xml = "\n".join(kml_placemarks)

    kml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Indian Highway Pothole 3D Surveillance Survey</name>
    <description>3D Volumetric Road Anomaly Surface Model for Google Earth Pro</description>
    
    <Style id="potholeHigh">
      <PolyStyle>
        <color>cc1400d9</color>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
    </Style>

    <Style id="potholeMedium">
      <PolyStyle>
        <color>cc0070e6</color>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>ff0080ff</color>
        <width>2</width>
      </LineStyle>
    </Style>

    <Style id="potholeLow">
      <PolyStyle>
        <color>cc33a832</color>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>ff00ff00</color>
        <width>2</width>
      </LineStyle>
    </Style>

    {placemarks_xml}
  </Document>
</kml>
"""
    return kml_template


def export_smart_city_wms_feed(defects_df: pd.DataFrame) -> dict:
    """
    Generate an Integrated Command and Control Center (ICCC) Smart City API payload.
    Compliant with Ministry of Housing and Urban Affairs (MoHUA) Smart Cities Mission standards.
    """
    has_coords = isinstance(defects_df, pd.DataFrame) and "latitude" in defects_df.columns and "longitude" in defects_df.columns
    valid_df = defects_df.dropna(subset=["latitude", "longitude"]).copy() if (has_coords and not defects_df.empty) else pd.DataFrame()

    total_records = len(valid_df)
    critical_records = int((valid_df["severity_category"] == "High").sum()) if "severity_category" in valid_df.columns else 0

    incidents = []
    for _, r in valid_df.iterrows():
        incidents.append({
            "incident_id": str(r.get("detection_id", "N/A")),
            "docket_ref": str(r.get("grievance_ref_no") or r.get("reference_no") or "PENDING"),
            "coordinates": {
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            },
            "pavement_metrics": {
                "severity_category": str(r.get("severity_category", "Medium")),
                "severity_score": float(r.get("severity_score", 0.5)),
                "max_depth_cm": float(r.get("max_depth_cm", 4.0)) if pd.notnull(r.get("max_depth_cm")) else 4.0,
                "cavity_volume_liters": float(r.get("cavity_volume_liters", 2.5)) if pd.notnull(r.get("cavity_volume_liters")) else 2.5,
            },
            "reported_timestamp": str(r.get("timestamp", datetime.now().isoformat())),
            "status": str(r.get("grievance_status", "ACTIVE_HAZARD")),
        })

    return {
        "smart_city_system": "National Highway & Urban Road Surface Monitoring Grid",
        "iccc_protocol_version": "2.4",
        "generated_at": datetime.now().isoformat(),
        "total_active_hazards": total_records,
        "critical_priority_hazards": critical_records,
        "incidents": incidents,
    }
