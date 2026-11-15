"""
src/v2x_hazard_broadcaster.py
=============================
Connected Vehicle (V2X) Collaborative Hazard Radar & Dynamic Braking Warning Engine.

REAL-WORLD PRODUCTION CAPABILITY:
Reporting a pothole to the government resolves it in 3 to 7 days, but what about the
car or motorcycle driving 300 meters behind at 80 km/h RIGHT NOW?

This module turns every detected pothole into an active V2X hazard beacon:
1. Physics-Based Stopping Distance Calculation:
   - Calculates dynamic stopping distance based on vehicle speed and pavement friction:
     d_stop = (v * t_reaction) + (v^2 / (2 * mu * g))
   - Warns the driver with sufficient lead time to decelerate safely without
     panic-braking or swerving into adjacent traffic.
2. Directional Heading Filter:
   - Computes azimuth bearing from vehicle to hazard.
   - Suppresses warnings for potholes on the opposite carriageway or behind the car.
3. Multi-Vehicle Fleet Sync:
   - Queries the crowdsourced SQLite registry (`grievance_registry`) in real-time.
   - Pushes early-warning audio alerts into the cabin via `InCabinAdasAudio`.
"""

import os
import sys
import math
import sqlite3
from datetime import datetime

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.database import get_db_connection
from src.adas_audio import InCabinAdasAudio


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two GPS coordinates."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 999999.0
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate forward azimuth bearing in degrees from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


class V2XHazardRadar:
    """
    Real-time connected vehicle hazard early-warning radar.
    """

    def __init__(self, db_path: str = None, enable_audio: bool = True):
        self.db_path = db_path or config.DATABASE_PATH
        self.enable_audio = enable_audio
        self.adas_audio = InCabinAdasAudio() if enable_audio else None
        self.cooldown_tracker = {}  # {registry_id: last_alerted_timestamp}
        self.alert_cooldown_seconds = 15.0
        self._cached_rows = None
        self._last_db_query_time = datetime.min

    @staticmethod
    def calculate_stopping_distance_m(speed_kmh: float, is_wet: bool = False) -> dict:
        """
        Compute physics-based emergency stopping distance:
        d_stop = d_reaction + d_braking
        """
        v_ms = max(0.0, speed_kmh / 3.6)
        t_reaction = 1.2  # Driver perception-reaction time in seconds
        mu = 0.40 if is_wet else 0.70  # Tire-road friction coefficient
        g = 9.81

        d_reaction = v_ms * t_reaction
        d_braking = (v_ms ** 2) / (2.0 * mu * g)
        d_total = d_reaction + d_braking

        # Safe early warning distance (at least 3.2x stopping distance or min 250m)
        warning_horizon = max(250.0, d_total * 3.2)

        return {
            "speed_kmh": round(speed_kmh, 1),
            "reaction_distance_m": round(d_reaction, 1),
            "braking_distance_m": round(d_braking, 1),
            "total_stopping_distance_m": round(d_total, 1),
            "safe_warning_horizon_m": round(warning_horizon, 1),
        }

    def scan_ahead_for_hazards(
        self,
        current_lat: float,
        current_lon: float,
        current_speed_kmh: float = 60.0,
        current_heading_deg: float = None,
        is_wet: bool = False,
    ) -> list:
        """
        Query the active hazard geofence ahead of the vehicle.

        Returns
        -------
        list of matching active hazards within the warning horizon, ordered by proximity.
        """
        if current_lat is None or current_lon is None:
            return []

        stopping_info = self.calculate_stopping_distance_m(current_speed_kmh, is_wet=is_wet)
        warning_horizon_m = stopping_info["safe_warning_horizon_m"]

        active_alerts = []
        now = datetime.now()

        # Cache database lookups for 2.0s to avoid per-frame disk I/O bottlenecks
        if (now - self._last_db_query_time).total_seconds() > 2.0 or self._cached_rows is None:
            conn = get_db_connection(self.db_path)
            cur = conn.cursor()
            try:
                cur.execute("""
                    SELECT registry_id, latitude, longitude, road_name,
                           severity_category, max_depth_cm, cavity_volume_liters,
                           citizen_corroboration_count, original_reference_no
                    FROM grievance_registry
                    WHERE is_resolved = 0
                """)
                self._cached_rows = [dict(r) for r in cur.fetchall()]
            except Exception:
                self._cached_rows = []
            finally:
                conn.close()
            self._last_db_query_time = now

        rows = self._cached_rows

        for r in rows:
            dist = haversine_distance_m(current_lat, current_lon, r["latitude"], r["longitude"])
            if dist > warning_horizon_m:
                continue

            # Heading filter: check if the hazard is ahead of the vehicle
            if current_heading_deg is not None:
                bearing_to_hazard = calculate_bearing_deg(current_lat, current_lon, r["latitude"], r["longitude"])
                angle_diff = abs((bearing_to_hazard - current_heading_deg + 180.0) % 360.0 - 180.0)
                # Only alert if within 60 degrees of vehicle forward heading
                if angle_diff > 60.0:
                    continue

            # Check alert cooldown to prevent repetitive ear fatigue
            reg_id = r["registry_id"]
            last_alert = self.cooldown_tracker.get(reg_id)
            if last_alert and (now - last_alert).total_seconds() < self.alert_cooldown_seconds:
                continue

            time_to_arrival_sec = round(dist / max(1.0, current_speed_kmh / 3.6), 1)

            # Recommend safe target speed
            depth = r.get("max_depth_cm", 5.0)
            rec_speed = 30.0 if depth > 6.0 else 45.0

            alert_payload = {
                "registry_id": reg_id,
                "reference_no": r.get("original_reference_no", "CPGRAMS-HAZARD"),
                "road_name": r.get("road_name", "Highway"),
                "distance_m": round(dist, 1),
                "time_to_arrival_sec": time_to_arrival_sec,
                "severity_category": r.get("severity_category", "High"),
                "max_depth_cm": depth,
                "recommended_speed_kmh": rec_speed,
                "citizen_corroborations": r.get("citizen_corroboration_count", 1),
                "stopping_distance_info": stopping_info,
            }
            active_alerts.append(alert_payload)
            self.cooldown_tracker[reg_id] = now

            # Trigger spoken audio warning
            if self.enable_audio and self.adas_audio:
                self.adas_audio.trigger_pothole_voice_alert(
                    severity_category=alert_payload["severity_category"],
                    depth_cm=depth,
                    distance_m=dist,
                )

        # Sort closest hazard first
        active_alerts.sort(key=lambda x: x["distance_m"])
        return active_alerts
