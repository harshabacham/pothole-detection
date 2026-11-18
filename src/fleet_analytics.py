"""
src/fleet_analytics.py
======================
Multi-Drive Pothole Growth Velocity and Fleet Crowdsourcing Engine.

Tracks the temporal evolution and degradation of road surface anomalies
across repeated drives over the same GPS coordinates.

Computes:
1. Spatial clustering of multi-vehicle sightings (Haversine formula within 12m).
2. Pothole Growth Velocity: delta volume (L/week), delta depth (cm/week), and % area expansion.
3. Imminent collapse & critical axle-breaking hazard alerts.
4. Municipal Fleet Summary & Degradation Timelines.
"""

import os
import sys
import math
import sqlite3
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.database import get_db_connection


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS coordinates in meters."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 999999.0

    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class PotholeDegradationTracker:
    """
    Tracks and models the physical growth rate and degradation velocity
    of road defects observed across sequential vehicle drives over time.
    """

    def __init__(self, db_path: str = None, cluster_radius_m: float = None):
        self.db_path = db_path if db_path is not None else config.DATABASE_PATH
        self.cluster_radius_m = (
            cluster_radius_m if cluster_radius_m is not None
            else config.FLEET_GEO_CLUSTER_RADIUS_METERS
        )
        self._init_timeline_table()

    def _init_timeline_table(self):
        """Create timeline tracking schema in SQLite if not present."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS defect_clusters (
            cluster_id TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            road_name TEXT,
            observation_count INTEGER DEFAULT 1,
            first_seen_timestamp TEXT,
            last_seen_timestamp TEXT,
            initial_depth_cm REAL,
            current_depth_cm REAL,
            initial_volume_l REAL,
            current_volume_l REAL,
            initial_area_px INTEGER,
            current_area_px INTEGER,
            growth_rate_volume_l_per_week REAL DEFAULT 0.0,
            growth_percent_total REAL DEFAULT 0.0,
            irc_category TEXT,
            degradation_status TEXT DEFAULT 'STABLE'
        );
        """)
        conn.commit()
        conn.close()

    def find_nearest_cluster(self, lat: float, lon: float) -> tuple:
        """
        Find existing cluster within cluster_radius_m of (lat, lon).
        Returns (cluster_id, cluster_dict, distance_m) or (None, None, None).
        """
        if lat is None or lon is None:
            return None, None, None

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM defect_clusters").fetchall()
        conn.close()

        best_cluster = None
        min_dist = float("inf")

        for r in rows:
            dist_m = haversine_distance_meters(lat, lon, r["latitude"], r["longitude"])
            if dist_m < self.cluster_radius_m and dist_m < min_dist:
                min_dist = dist_m
                best_cluster = dict(r)

        if best_cluster is not None:
            return best_cluster["cluster_id"], best_cluster, round(min_dist, 1)
        return None, None, None

    def record_sighting(
        self,
        lat: float,
        lon: float,
        depth_cm: float = 4.5,
        volume_l: float = 3.0,
        pixel_area: int = 4000,
        road_name: str = "Local Road",
        irc_category: str = "IRC:Pothole",
        timestamp: str = None,
    ) -> dict:
        """
        Ingest a defect sighting, update growth metrics, and classify degradation velocity.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if lat is None or lon is None:
            return {
                "cluster_id": "UNGEOTAGGED",
                "is_new_cluster": False,
                "observation_count": 1,
                "growth_percent": 0.0,
                "growth_rate_l_per_week": 0.0,
                "degradation_status": "STABLE",
                "days_tracked": 0.0,
            }

        cluster_id, cluster_obj, _ = self.find_nearest_cluster(lat, lon)
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        now_dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.now()

        if cluster_obj is None:
            # Spawn new cluster
            cid = f"CLUSTER_{int(lat*10000)}_{int(lon*10000)}"
            cursor.execute("""
            INSERT INTO defect_clusters (
                cluster_id, latitude, longitude, road_name, observation_count,
                first_seen_timestamp, last_seen_timestamp, initial_depth_cm,
                current_depth_cm, initial_volume_l, current_volume_l,
                initial_area_px, current_area_px, growth_rate_volume_l_per_week,
                growth_percent_total, irc_category, degradation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid, lat, lon, road_name, 1,
                timestamp, timestamp, depth_cm,
                depth_cm, volume_l, volume_l,
                pixel_area, pixel_area, 0.0,
                0.0, irc_category, "STABLE"
            ))
            conn.commit()
            conn.close()

            return {
                "cluster_id": cid,
                "is_new_cluster": True,
                "revisit_count": 1,
                "growth_percent": 0.0,
                "growth_rate_l_week": 0.0,
                "degradation_status": "STABLE",
                "alert_message": "Initial defect sighting registered.",
            }

        # Existing cluster: update degradation metrics
        cid = cluster_obj["cluster_id"]
        obs_count = cluster_obj["observation_count"] + 1

        first_dt = datetime.fromisoformat(cluster_obj["first_seen_timestamp"])
        delta_days = max(0.1, (now_dt - first_dt).total_seconds() / 86400.0)
        delta_weeks = max(0.014, delta_days / 7.0)

        init_vol = cluster_obj["initial_volume_l"]
        init_area = max(1, cluster_obj["initial_area_px"])

        delta_vol = max(0.0, volume_l - init_vol)
        vol_growth_per_week = round(delta_vol / delta_weeks, 2)
        growth_pct = round(((pixel_area - init_area) / float(init_area)) * 100.0, 1)

        # Degradation status classification
        if depth_cm >= config.CRITICAL_COLLAPSE_DEPTH_CM or volume_l >= 15.0:
            status = "CRITICAL_COLLAPSE_RISK"
            msg = f"CRITICAL: Cavity depth ({depth_cm:.1f} cm) exceeds safety limit! Immediate tire rupture risk."
        elif growth_pct >= config.VELOCITY_ALERT_GROWTH_PERCENT or vol_growth_per_week >= 1.5:
            status = "RAPID_DETERIORATION"
            msg = f"HIGH ALERT: Pothole expanded by +{growth_pct}% ({vol_growth_per_week} L/wk). Rapid monsoon erosion."
        elif growth_pct > 5.0:
            status = "STEADY_GROWTH"
            msg = f"Defect expanding steadily (+{growth_pct}% over {int(delta_days)} days)."
        else:
            status = "STABLE"
            msg = "Surface condition stable since previous drive."

        cursor.execute("""
        UPDATE defect_clusters
        SET observation_count = ?,
            last_seen_timestamp = ?,
            current_depth_cm = ?,
            current_volume_l = ?,
            current_area_px = ?,
            growth_rate_volume_l_per_week = ?,
            growth_percent_total = ?,
            irc_category = ?,
            degradation_status = ?
        WHERE cluster_id = ?
        """, (
            obs_count, timestamp, depth_cm, volume_l,
            pixel_area, vol_growth_per_week, growth_pct,
            irc_category, status, cid
        ))
        conn.commit()
        conn.close()

        return {
            "cluster_id": cid,
            "is_new_cluster": False,
            "revisit_count": obs_count,
            "growth_percent": growth_pct,
            "growth_rate_l_week": vol_growth_per_week,
            "degradation_status": status,
            "alert_message": msg,
        }

    def get_all_clusters(self) -> pd.DataFrame:
        """Retrieve all clustered degradation entities as a pandas DataFrame."""
        conn = get_db_connection(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM defect_clusters ORDER BY current_depth_cm DESC", conn)
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        return df

    def get_fleet_summary(self) -> dict:
        """Generate high-level fleet degradation summary statistics."""
        df = self.get_all_clusters()
        if df.empty:
            return {
                "total_clusters": 0,
                "rapid_growth_count": 0,
                "critical_collapse_count": 0,
                "avg_growth_percent": 0.0,
                "top_deteriorating_roads": [],
            }

        crit_count = int((df["degradation_status"] == "CRITICAL_COLLAPSE_RISK").sum())
        rapid_count = int((df["degradation_status"] == "RAPID_DETERIORATION").sum())
        avg_growth = float(df["growth_percent_total"].mean())

        # Top deteriorating locations
        top_df = df[df["growth_percent_total"] > 0].sort_values("growth_percent_total", ascending=False).head(5)
        top_roads = top_df[["road_name", "growth_percent_total", "current_depth_cm", "degradation_status"]].to_dict(orient="records")

        return {
            "total_clusters": len(df),
            "rapid_growth_count": rapid_count,
            "critical_collapse_count": crit_count,
            "avg_growth_percent": round(avg_growth, 1),
            "top_deteriorating_roads": top_roads,
        }

