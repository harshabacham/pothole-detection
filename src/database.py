"""
src/database.py
===============
SQLite persistence engine for defect records, unique spatio-temporal tracks,
and Government Grievance tracking status.
"""

import os
import sys
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


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Obtain an SQLite connection, ensuring parent directory exists."""
    if db_path is None:
        db_path = config.DATABASE_PATH

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = None) -> None:
    """
    Initialize SQLite schema for storing unique defect reports and grievance statuses.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS defect_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id TEXT UNIQUE,
        track_id TEXT,
        latitude REAL,
        longitude REAL,
        timestamp TEXT,
        severity_category TEXT,
        severity_score REAL,
        anomaly_score REAL,
        pixel_area INTEGER,
        image_path TEXT,
        source_media TEXT,
        grievance_status TEXT DEFAULT 'Unreported',
        grievance_ref_no TEXT,
        dispatched_portal TEXT,
        dispatched_at TEXT
    );
    """)

    # Automatic column migration for existing tables
    existing_cols = [r["name"] for r in cursor.execute("PRAGMA table_info(defect_records)").fetchall()]
    new_cols = [
        ("grievance_status", "TEXT DEFAULT 'Unreported'"),
        ("grievance_ref_no", "TEXT"),
        ("dispatched_portal", "TEXT"),
        ("dispatched_at", "TEXT"),
        ("bev_area_sqcm", "REAL"),
        ("lateral_offset_m", "REAL"),
        ("wheel_track_hazard", "TEXT"),
        ("imu_shock_g", "REAL"),
        ("threat_index", "REAL"),
        ("fusion_status", "TEXT"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE defect_records ADD COLUMN {col_name} {col_type};")
            except Exception:
                pass

    conn.commit()
    conn.close()


def insert_defect(
    detection_id: str,
    track_id: str = None,
    latitude: float = None,
    longitude: float = None,
    timestamp: str = None,
    severity_category: str = "Low",
    severity_score: float = 0.0,
    anomaly_score: float = 0.0,
    pixel_area: int = 0,
    image_path: str = None,
    source_media: str = None,
    db_path: str = None,
) -> bool:
    """
    Insert a defect record into SQLite with deduplication on detection_id.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO defect_records (
            detection_id, track_id, latitude, longitude, timestamp,
            severity_category, severity_score, anomaly_score, pixel_area,
            image_path, source_media
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(detection_id) DO UPDATE SET
            severity_score = excluded.severity_score,
            anomaly_score = excluded.anomaly_score,
            pixel_area = excluded.pixel_area
        """, (
            detection_id,
            track_id,
            latitude,
            longitude,
            timestamp,
            severity_category,
            round(severity_score, 4),
            round(anomaly_score, 4),
            int(pixel_area),
            image_path,
            source_media,
        ))
        conn.commit()
        success = True
    except Exception as exc:
        print(f"[ERROR] database: Insertion failed for {detection_id}: {exc}")
        success = False
    finally:
        conn.close()

    return success


def update_grievance_status(
    detection_id: str,
    status: str = "Dispatched",
    ref_no: str = None,
    portal: str = "CPGRAMS",
    db_path: str = None,
) -> bool:
    """
    Update government complaint submission status for a defect entry.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    success = False

    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        UPDATE defect_records
        SET grievance_status = ?,
            grievance_ref_no = ?,
            dispatched_portal = ?,
            dispatched_at = ?
        WHERE detection_id = ?
        """, (status, ref_no, portal, now_str, detection_id))
        conn.commit()
        success = (cursor.rowcount > 0)
    except Exception as exc:
        print(f"[ERROR] database: Failed to update grievance for {detection_id}: {exc}")
    finally:
        conn.close()

    return success


def disapprove_defect(
    detection_id: str,
    reason: str = "Disapproved by user (False Positive)",
    db_path: str = None,
) -> bool:
    """
    Mark a recorded defect as Disapproved / Rejected by the user.
    Prevents it from being submitted to government portals.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    success = False

    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        UPDATE defect_records
        SET grievance_status = 'Disapproved',
            grievance_ref_no = 'REJECTED_BY_USER',
            dispatched_portal = ?,
            dispatched_at = ?
        WHERE detection_id = ?
        """, (reason, now_str, detection_id))
        conn.commit()
        success = (cursor.rowcount > 0)
    except Exception as exc:
        print(f"[ERROR] database: Failed to disapprove defect {detection_id}: {exc}")
    finally:
        conn.close()

    return success


def get_pending_review_defects(db_path: str = None) -> pd.DataFrame:
    """
    Retrieve all defect records currently awaiting user review / approval.
    """
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM defect_records WHERE grievance_status IN ('Unreported', 'Pending Review') ORDER BY id DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def get_all_defects(db_path: str = None) -> pd.DataFrame:
    """
    Retrieve all recorded defect entries as a pandas DataFrame.
    """
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM defect_records ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def get_geotagged_defects(db_path: str = None) -> pd.DataFrame:
    """
    Retrieve only records that have verified latitude and longitude values.
    """
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM defect_records WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY id DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    return df
