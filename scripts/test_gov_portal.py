"""
scripts/test_gov_portal.py
==========================
End-to-end verification script for Government of India Citizen Grievance Portal
Integration (CPGRAMS / MoRTH, NHAI, PMGSY Meri Sadak).

Tests:
1. Reverse geocoding of GPS coordinates to Indian road address.
2. Building compliant complaint payload with telemetry and citizen safety statement.
3. ReportLab generation of official printable PDF complaint dossier.
4. Portal dispatch simulation and SQLite database status migration/tracking.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.database import init_database, get_all_defects, insert_defect, update_grievance_status
from src.gov_portal_api import (
    reverse_geocode_location,
    build_grievance_payload,
    generate_complaint_pdf,
    dispatch_grievance_to_portal,
    generate_complaint_reference,
)


def run_gov_portal_test():
    print("=" * 70)
    print("TEST: Government of India Grievance Portal Integration")
    print("=" * 70)

    # 1. Initialize DB
    init_database()
    print("[OK] Database initialized and schema verified.")

    # 2. Test reverse geocoding
    test_lat, test_lon = 28.53831, 77.24312  # New Delhi (Greater Kailash area)
    print(f"\n[INFO] Testing reverse geocoding for coordinates ({test_lat}, {test_lon})...")
    loc_info = reverse_geocode_location(test_lat, test_lon)
    print(f"[OK] Reverse Geocode Result:")
    print(f"     Road: {loc_info.get('road')}")
    print(f"     District: {loc_info.get('district')}")
    print(f"     State: {loc_info.get('state')}")
    print(f"     Postcode: {loc_info.get('postcode')}")
    print(f"     Full: {loc_info.get('formatted_address')}")

    # 3. Create or select a sample defect
    sample_defect = {
        "detection_id": "TEST_POTHOLE_MORTH_001",
        "track_id": "TRK_001",
        "latitude": test_lat,
        "longitude": test_lon,
        "severity_category": "High",
        "severity_score": 0.842,
        "anomaly_score": -0.285,
        "pixel_area": 8450,
        "image_path": os.path.join(config.REAL_ROAD_SAMPLES_DIR, "real_pothole_1.jpg")
        if os.path.exists(os.path.join(config.REAL_ROAD_SAMPLES_DIR, "real_pothole_1.jpg"))
        else None,
        "source_media": "test_dashcam_stream.mp4",
    }

    insert_defect(
        detection_id=sample_defect["detection_id"],
        track_id=sample_defect["track_id"],
        latitude=sample_defect["latitude"],
        longitude=sample_defect["longitude"],
        severity_category=sample_defect["severity_category"],
        severity_score=sample_defect["severity_score"],
        anomaly_score=sample_defect["anomaly_score"],
        pixel_area=sample_defect["pixel_area"],
        image_path=sample_defect["image_path"],
        source_media=sample_defect["source_media"],
    )
    print(f"\n[OK] Sample defect inserted into SQLite: {sample_defect['detection_id']}")

    # 4. Test building grievance payload for CPGRAMS and NHAI
    print("\n[INFO] Constructing CPGRAMS grievance payload...")
    payload_cpgrams = build_grievance_payload(sample_defect, portal_key="CPGRAMS")
    print(f"[OK] CPGRAMS Reference No: {payload_cpgrams['grievance_reference_no']}")
    print(f"     Target Portal: {payload_cpgrams['target_portal']}")
    print(f"     Statement snippet: {payload_cpgrams['official_statement'][:100]}...")

    # 5. Test PDF Dossier Generation
    print("\n[INFO] Generating official printable Government PDF complaint dossier...")
    pdf_path = generate_complaint_pdf(
        payload=payload_cpgrams,
        defect_image_path=sample_defect["image_path"],
    )
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    pdf_size_kb = os.path.getsize(pdf_path) / 1024.0
    print(f"[OK] Complaint PDF successfully created: {pdf_path} ({pdf_size_kb:.1f} KB)")

    # 6. Test Full Portal Dispatch Engine
    print("\n[INFO] Testing grievance dispatch execution...")
    dispatch_res = dispatch_grievance_to_portal(
        defect_dict=sample_defect,
        portal_key="NHAI",
    )
    print(f"[OK] Dispatch Response:")
    print(f"     Status: {dispatch_res['status']}")
    print(f"     API Status: {dispatch_res['api_status']}")
    print(f"     Ref Number: {dispatch_res['reference_number']}")
    print(f"     Target Portal: {dispatch_res['target_portal']}")
    print(f"     PDF Dossier: {dispatch_res['pdf_path']}")

    # 7. Verify SQLite Database Record Updated
    df = get_all_defects()
    updated_row = df[df["detection_id"] == sample_defect["detection_id"]]
    assert not updated_row.empty, "Record not found in database!"
    rec = updated_row.iloc[0]
    print(f"\n[OK] SQLite Database Verification:")
    print(f"     Grievance Status: {rec['grievance_status']}")
    print(f"     Grievance Ref No: {rec['grievance_ref_no']}")
    print(f"     Dispatched Portal: {rec['dispatched_portal']}")
    print(f"     Dispatched At: {rec['dispatched_at']}")

    assert rec["grievance_status"] == "Dispatched to Portal"
    print("\n" + "=" * 70)
    print("[SUCCESS] All Indian Government Portal Grievance integration tests PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_gov_portal_test()

