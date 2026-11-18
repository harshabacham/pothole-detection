"""
src/crowd_dedup.py
==================
Crowdsourced Spatial Deduplication & Citizen Corroboration Engine.

Critical Production Problem Solved:
    If 100 vehicles with this system drive past the same pothole on the same
    highway, the government portal must NOT receive 100 duplicate complaints.
    That would be spam, not civic technology.

Solution:
    1. Maintains a persistent spatial grievance registry (SQLite) of all
       previously-reported potholes, keyed by GPS coordinates.
    2. Before dispatching a new complaint, queries the registry using the
       Haversine formula (configurable radius, default 15 meters).
    3. If a matching grievance already exists:
       - SUPPRESSES the duplicate government portal submission.
       - INCREMENTS the "citizen_corroboration_count" on the original complaint.
       - RECORDS the new vehicle's evidence photo, timestamp, and severity reading
         as a corroboration entry (strengthens the existing case).
       - When corroboration count crosses thresholds (5, 15, 50 citizens),
         automatically ESCALATES the priority tier to the government.
    4. If no matching grievance exists within the radius:
       - Files a fresh complaint as normal.
       - Registers it in the deduplication registry for future vehicles.

This turns redundant spam into a powerful crowdsourced civic pressure signal:
    "52 citizens independently confirmed this pothole at Greater Kailash II"
    is far more actionable than 52 identical complaint PDFs.
"""

import os
import sys
import math
import sqlite3
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.database import get_db_connection


# ---------------------------------------------------------------------------
# Haversine distance (local copy to avoid circular imports)
# ---------------------------------------------------------------------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two GPS points."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 999999.0
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Escalation Tier Thresholds
# ---------------------------------------------------------------------------
_ESCALATION_TIERS = [
    (50, "CRITICAL_MASS",   "CRITICAL: 50+ citizens independently confirmed this hazard. Demands immediate PWD intervention."),
    (15, "HIGH_PRIORITY",   "HIGH PRIORITY: 15+ citizen corroborations. Escalated for urgent road maintenance scheduling."),
    (5,  "COMMUNITY_ALERT", "COMMUNITY ALERT: 5+ citizens have confirmed this pothole. Prioritized in complaint queue."),
    (1,  "FILED",           "Grievance filed and registered in government portal."),
]


class CrowdsourcedSpatialDeduplicator:
    """
    Ensures that multiple vehicles reporting the same physical pothole
    result in ONE government complaint with increasing citizen corroboration,
    not N duplicate submissions.
    """

    def __init__(
        self,
        db_path: str = None,
        dedup_radius_m: float = 15.0,
        cooldown_hours: float = 72.0,
    ):
        """
        Parameters
        ----------
        db_path : str
            Path to SQLite database (defaults to project pothole_database.db).
        dedup_radius_m : float
            GPS radius in meters within which two reports are considered the
            same physical pothole (default 15m ~ one road lane width).
        cooldown_hours : float
            Hours after which a REPAIRED pothole can be re-reported as a new
            defect if it recurs (default 72 hours = 3 days).
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.dedup_radius_m = dedup_radius_m
        self.cooldown_hours = cooldown_hours
        self._init_registry_table()

    def _init_registry_table(self):
        """Create the crowdsourced grievance registry schema."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS grievance_registry (
            registry_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            road_name TEXT,
            district TEXT,
            state TEXT,
            postcode TEXT,
            full_address TEXT,
            original_detection_id TEXT,
            original_reference_no TEXT,
            target_portal TEXT,
            severity_category TEXT,
            max_depth_cm REAL,
            cavity_volume_liters REAL,
            irc_category TEXT,
            citizen_corroboration_count INTEGER DEFAULT 1,
            escalation_tier TEXT DEFAULT 'FILED',
            first_reported_at TEXT,
            last_corroborated_at TEXT,
            is_resolved INTEGER DEFAULT 0,
            resolved_at TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS corroboration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            registry_id TEXT NOT NULL,
            vehicle_detection_id TEXT,
            latitude REAL,
            longitude REAL,
            severity_score REAL,
            max_depth_cm REAL,
            evidence_image_path TEXT,
            corroborated_at TEXT,
            FOREIGN KEY (registry_id) REFERENCES grievance_registry(registry_id)
        );
        """)

        conn.commit()
        conn.close()

    def find_existing_grievance(self, lat: float, lon: float) -> dict:
        """
        Search the registry for an active (unresolved) grievance within
        dedup_radius_m of the given GPS coordinates.

        Returns
        -------
        dict or None
            The matching registry row as a dict, or None if no match.
        """
        if lat is None or lon is None:
            return None

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM grievance_registry WHERE is_resolved = 0"
        ).fetchall()
        conn.close()

        cutoff = datetime.now() - timedelta(hours=self.cooldown_hours)
        best_match = None
        best_dist = float("inf")

        for row in rows:
            dist = _haversine_m(lat, lon, row["latitude"], row["longitude"])
            if dist < self.dedup_radius_m and dist < best_dist:
                # Check cooldown: don't match against very old resolved entries
                last_time_str = row["last_corroborated_at"] or row["first_reported_at"]
                if last_time_str:
                    try:
                        last_dt = datetime.fromisoformat(last_time_str)
                        if last_dt < cutoff:
                            continue  # Stale entry — could be re-reported as new
                    except (ValueError, TypeError):
                        pass
                best_match = dict(row)
                best_dist = dist

        return best_match

    def register_new_grievance(
        self,
        lat: float,
        lon: float,
        detection_id: str,
        reference_no: str,
        target_portal: str,
        severity_category: str = "Medium",
        max_depth_cm: float = 5.0,
        cavity_volume_liters: float = 3.5,
        irc_category: str = "IRC:Pothole",
        road_name: str = "Local Road",
        district: str = "District",
        state: str = "India",
        postcode: str = "",
        full_address: str = "",
        evidence_image_path: str = None,
    ) -> dict:
        """
        Register a brand-new pothole grievance in the deduplication registry.
        Called only when no existing match was found.
        """
        now_str = datetime.now().isoformat()
        lat_tag = f"{int(lat * 10000)}" if lat is not None else "UNGEOTAGGED"
        lon_tag = f"{int(lon * 10000)}" if lon is not None else "0"
        registry_id = f"GRV_{lat_tag}_{lon_tag}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO grievance_registry (
            registry_id, latitude, longitude, road_name, district, state,
            postcode, full_address, original_detection_id, original_reference_no,
            target_portal, severity_category, max_depth_cm, cavity_volume_liters,
            irc_category, citizen_corroboration_count, escalation_tier,
            first_reported_at, last_corroborated_at, is_resolved
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            registry_id, lat, lon, road_name, district, state,
            postcode, full_address, detection_id, reference_no,
            target_portal, severity_category, max_depth_cm, cavity_volume_liters,
            irc_category, 1, "FILED",
            now_str, now_str, 0,
        ))

        # Log first citizen corroboration
        cursor.execute("""
        INSERT INTO corroboration_log (
            registry_id, vehicle_detection_id, latitude, longitude,
            severity_score, max_depth_cm, evidence_image_path, corroborated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            registry_id, detection_id, lat, lon,
            0.0, max_depth_cm, evidence_image_path, now_str,
        ))

        conn.commit()
        conn.close()

        return {
            "action": "NEW_GRIEVANCE_FILED",
            "registry_id": registry_id,
            "reference_no": reference_no,
            "citizen_count": 1,
            "escalation_tier": "FILED",
            "should_dispatch": True,
            "message": f"New pothole grievance registered. Reference: {reference_no}",
        }

    def add_citizen_corroboration(
        self,
        existing_grievance: dict,
        new_detection_id: str,
        lat: float,
        lon: float,
        severity_score: float = 0.0,
        max_depth_cm: float = 5.0,
        evidence_image_path: str = None,
    ) -> dict:
        """
        Add a citizen corroboration to an existing grievance instead of
        filing a duplicate complaint.
        """
        now_str = datetime.now().isoformat()
        registry_id = existing_grievance["registry_id"]
        old_count = existing_grievance["citizen_corroboration_count"]
        new_count = old_count + 1

        # Determine new escalation tier
        new_tier = "FILED"
        tier_message = ""
        for threshold, tier, msg in _ESCALATION_TIERS:
            if new_count >= threshold:
                new_tier = tier
                tier_message = msg
                break

        # Update severity if the new reading is worse
        updated_depth = max(
            existing_grievance.get("max_depth_cm") or 0.0,
            max_depth_cm,
        )

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        UPDATE grievance_registry
        SET citizen_corroboration_count = ?,
            escalation_tier = ?,
            last_corroborated_at = ?,
            max_depth_cm = ?,
            severity_category = CASE
                WHEN ? > COALESCE(max_depth_cm, 0) THEN
                    CASE WHEN ? >= 8.0 THEN 'High'
                         WHEN ? >= 4.0 THEN 'Medium'
                         ELSE severity_category END
                ELSE severity_category END
        WHERE registry_id = ?
        """, (
            new_count, new_tier, now_str,
            updated_depth,
            max_depth_cm, max_depth_cm, max_depth_cm,
            registry_id,
        ))

        cursor.execute("""
        INSERT INTO corroboration_log (
            registry_id, vehicle_detection_id, latitude, longitude,
            severity_score, max_depth_cm, evidence_image_path, corroborated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            registry_id, new_detection_id, lat, lon,
            severity_score, max_depth_cm, evidence_image_path, now_str,
        ))

        conn.commit()
        conn.close()

        escalation_changed = (new_tier != existing_grievance.get("escalation_tier", "FILED"))
        original_ref = existing_grievance.get("original_reference_no", "N/A")

        return {
            "action": "DUPLICATE_SUPPRESSED",
            "registry_id": registry_id,
            "reference_no": original_ref,
            "citizen_count": new_count,
            "escalation_tier": new_tier,
            "escalation_changed": escalation_changed,
            "should_dispatch": False,  # Do NOT submit another complaint
            "message": (
                f"Duplicate complaint SUPPRESSED. "
                f"Citizen #{new_count} corroborated existing grievance {original_ref}. "
                f"{tier_message}"
            ),
        }

    def check_and_decide(
        self,
        lat: float,
        lon: float,
        detection_id: str,
        severity_score: float = 0.0,
        max_depth_cm: float = 5.0,
        cavity_volume_liters: float = 3.5,
        irc_category: str = "IRC:Pothole",
        evidence_image_path: str = None,
    ) -> dict:
        """
        Master decision gate: should this detection be dispatched as a new
        government complaint, or should it be suppressed as a duplicate?

        Returns
        -------
        dict with keys:
            action : "NEW_GRIEVANCE_FILED" or "DUPLICATE_SUPPRESSED"
            should_dispatch : bool — whether to actually call the gov portal API
            reference_no : str — existing reference if duplicate
            citizen_count : int — total independent citizen confirmations
            escalation_tier : str — current priority tier
            message : str — human-readable status
        """
        existing = self.find_existing_grievance(lat, lon)

        if existing is not None:
            return self.add_citizen_corroboration(
                existing_grievance=existing,
                new_detection_id=detection_id,
                lat=lat,
                lon=lon,
                severity_score=severity_score,
                max_depth_cm=max_depth_cm,
                evidence_image_path=evidence_image_path,
            )

        # No match — will be registered after actual dispatch succeeds
        return {
            "action": "NEW_GRIEVANCE_PENDING",
            "registry_id": None,
            "reference_no": None,
            "citizen_count": 0,
            "escalation_tier": "PENDING",
            "should_dispatch": True,
            "message": "No prior grievance found within radius. New complaint will be filed.",
        }

    def get_registry_summary(self) -> dict:
        """
        High-level summary of the crowdsourced grievance registry.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        total = cursor.execute("SELECT COUNT(*) FROM grievance_registry").fetchone()[0]
        active = cursor.execute(
            "SELECT COUNT(*) FROM grievance_registry WHERE is_resolved = 0"
        ).fetchone()[0]
        total_corroborations = cursor.execute(
            "SELECT COALESCE(SUM(citizen_corroboration_count), 0) FROM grievance_registry"
        ).fetchone()[0]
        duplicates_suppressed = max(0, total_corroborations - total)
        critical_mass = cursor.execute(
            "SELECT COUNT(*) FROM grievance_registry WHERE escalation_tier = 'CRITICAL_MASS'"
        ).fetchone()[0]
        high_priority = cursor.execute(
            "SELECT COUNT(*) FROM grievance_registry WHERE escalation_tier = 'HIGH_PRIORITY'"
        ).fetchone()[0]

        conn.close()

        return {
            "total_unique_potholes": total,
            "active_grievances": active,
            "total_citizen_reports": total_corroborations,
            "duplicate_submissions_prevented": duplicates_suppressed,
            "critical_mass_escalations": critical_mass,
            "high_priority_escalations": high_priority,
        }

