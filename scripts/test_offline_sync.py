"""
scripts/test_offline_sync.py
============================
Verification Test Suite for Offline Store-and-Forward Sync Queue (src/offline_sync.py).

Tests:
1. Network Watchdog (is_online reachability).
2. Enqueuing multiple complaints during simulated offline driving.
3. Queue summary and persistence in SQLite.
4. Auto-flushing queue when network connectivity restores.
5. Verification of official reference numbers and zero data loss.
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.offline_sync import OfflineSyncManager

_TEST_DB = os.path.join(_PROJECT_ROOT, "data", "test_offline_sync.db")


def setup():
    """Reset test database."""
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)


def test_offline_sync():
    print("\n" + "=" * 70)
    print("  OFFLINE STORE-AND-FORWARD QUEUE VERIFICATION TEST")
    print("=" * 70)

    setup()
    mgr = OfflineSyncManager(db_path=_TEST_DB)

    # 1. Test Network Watchdog
    online_status = mgr.is_online(timeout=1.0)
    print(f"[OK] Network Watchdog: Internet Reachability = {online_status}")

    # 2. Simulate Driving Through a Highway Tunnel / Dead-Zone (3 Potholes Detected)
    simulated_potholes = [
        {
            "detection_id": "OFFLINE_POTHOLE_001",
            "track_id": "TRK_TUNNEL_1",
            "latitude": 28.53834,
            "longitude": 77.24315,
            "severity_category": "High",
            "severity_score": 0.85,
            "max_depth_cm": 6.8,
            "cavity_volume_liters": 4.5,
            "asphalt_repair_mass_kg": 10.8,
            "irc_category": "IRC:Pothole",
            "road_name": "NH-48 Mountain Pass Tunnel",
            "district": "Gurugram",
            "state": "Haryana",
            "postcode": "122001",
        },
        {
            "detection_id": "OFFLINE_POTHOLE_002",
            "track_id": "TRK_TUNNEL_2",
            "latitude": 28.53950,
            "longitude": 77.24420,
            "severity_category": "Medium",
            "severity_score": 0.65,
            "max_depth_cm": 4.5,
            "cavity_volume_liters": 3.0,
            "asphalt_repair_mass_kg": 7.2,
            "irc_category": "IRC:Pothole",
            "road_name": "NH-48 Ghat Section Km 128",
            "district": "Gurugram",
            "state": "Haryana",
            "postcode": "122001",
        },
        {
            "detection_id": "OFFLINE_POTHOLE_003",
            "track_id": "TRK_TUNNEL_3",
            "latitude": 28.54100,
            "longitude": 77.24580,
            "severity_category": "High",
            "severity_score": 0.90,
            "max_depth_cm": 7.2,
            "cavity_volume_liters": 5.2,
            "asphalt_repair_mass_kg": 12.5,
            "irc_category": "IRC:Pothole",
            "road_name": "NH-48 Ghat Section Km 130",
            "district": "Gurugram",
            "state": "Haryana",
            "postcode": "122001",
        },
    ]

    print("\n[SIMULATION: OFFLINE DRIVING IN TUNNEL / ZERO 4G COVERAGE]")
    for p in simulated_potholes:
        res = mgr.enqueue_complaint(
            defect_dict=p,
            target_portal="NHAI",
            evidence_image_path="outputs/evidence/sample_evidence.jpg",
            reason="CELLULAR_DATA_OFFLINE_TUNNEL",
        )
        print(f"  --> Enqueued: {res['detection_id']} | Status: {res['status']} | Queue ID: {res['queue_id']}")
        assert res["status"] == "QUEUED_OFFLINE"
        assert res["success"] is True

    # Check Queue Summary While Offline
    summary_offline = mgr.get_queue_summary()
    print(f"\n[QUEUE AUDIT - WHILE OFFLINE]")
    print(f"  Total Buffered : {summary_offline['total_items']}")
    print(f"  Pending Sync   : {summary_offline['pending_count']}")
    print(f"  Dispatched     : {summary_offline['dispatched_count']}")
    assert summary_offline["total_items"] == 3
    assert summary_offline["pending_count"] == 3
    assert summary_offline["dispatched_count"] == 0
    assert summary_offline["is_clear"] is False

    # 3. Simulate Vehicle Exiting Tunnel & Reconnecting to 4G/5G / WiFi
    print("\n[SIMULATION: VEHICLE EXITS TUNNEL -> 4G NETWORK RESTORED]")
    print("[INFO] Initiating automated batch flush to Government Portals...")
    flush_res = mgr.flush_queue(max_batch=10)

    print(f"\n[FLUSH RESULTS]")
    print(f"  Successfully Synced : {flush_res['synced_count']}")
    print(f"  Failed              : {flush_res['failed_count']}")
    print(f"  Remaining Pending   : {flush_res['remaining_pending']}")

    for item in flush_res["synced_items"]:
        print(f"  --> Flushed: {item['detection_id']} -> Assigned Docket Ref: {item['reference_no']}")
        assert item["reference_no"].startswith("NHAI")

    assert flush_res["synced_count"] == 3
    assert flush_res["failed_count"] == 0
    assert flush_res["remaining_pending"] == 0

    # Final Queue Verification
    summary_final = mgr.get_queue_summary()
    assert summary_final["pending_count"] == 0
    assert summary_final["dispatched_count"] == 3
    assert summary_final["is_clear"] is True
    print("\n[OK] Final Queue State: 100% CLEAR, Zero Data Loss!")

    # Cleanup test db
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)

    print("\n  --> ALL OFFLINE SYNC TESTS PASSED (100%)!")


def main():
    test_offline_sync()
    print("\n" + "=" * 70)
    print("  *** 100% SUCCESS: RESILIENT OFFLINE SYNC ENGINE FULLY VERIFIED ***")
    print("=" * 70)


if __name__ == "__main__":
    main()
