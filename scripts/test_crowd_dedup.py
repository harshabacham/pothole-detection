"""
scripts/test_crowd_dedup.py
===========================
Verification Test for Crowdsourced Spatial Deduplication & Citizen Corroboration Engine.

Tests:
1. First vehicle detects a pothole -> NEW complaint dispatched.
2. Second vehicle detects the SAME pothole (within 15m) -> DUPLICATE SUPPRESSED, citizen count = 2.
3. Third vehicle -> count = 3.
4. After 5 citizens -> escalation to COMMUNITY_ALERT.
5. After 15 citizens -> escalation to HIGH_PRIORITY.
6. After 50 citizens -> escalation to CRITICAL_MASS.
7. A pothole 200m away -> treated as a DIFFERENT pothole, new complaint filed.
8. Registry summary correctly counts duplicates prevented.
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.crowd_dedup import CrowdsourcedSpatialDeduplicator

# Use isolated test database
_TEST_DB = os.path.join(_PROJECT_ROOT, "data", "test_crowd_dedup.db")


def setup():
    """Clean test database."""
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)


def test_crowd_dedup():
    print("\n" + "=" * 70)
    print("  CROWDSOURCED SPATIAL DEDUPLICATION & CITIZEN CORROBORATION TEST")
    print("=" * 70)

    setup()
    dedup = CrowdsourcedSpatialDeduplicator(db_path=_TEST_DB, dedup_radius_m=15.0)

    # Pothole A: Greater Kailash II, Delhi
    pothole_lat = 28.53834
    pothole_lon = 77.24315

    # --- Vehicle 1: First to detect this pothole ---
    res1 = dedup.check_and_decide(
        lat=pothole_lat,
        lon=pothole_lon,
        detection_id="VEH1_DETECT_001",
        severity_score=0.65,
        max_depth_cm=5.2,
        cavity_volume_liters=3.8,
        irc_category="IRC:Pothole",
        evidence_image_path="outputs/evidence/veh1_evidence.jpg",
    )
    print(f"\n[Vehicle 1] Action: {res1['action']} | Should Dispatch: {res1['should_dispatch']}")
    assert res1["should_dispatch"] is True, "First vehicle must dispatch!"
    assert res1["action"] == "NEW_GRIEVANCE_PENDING"

    # Simulate actual dispatch and registration
    reg1 = dedup.register_new_grievance(
        lat=pothole_lat,
        lon=pothole_lon,
        detection_id="VEH1_DETECT_001",
        reference_no="NHAI/E/2026/10001",
        target_portal="NHAI",
        severity_category="Medium",
        max_depth_cm=5.2,
        cavity_volume_liters=3.8,
        irc_category="IRC:Pothole",
        road_name="Greater Kailash II",
        district="South Delhi",
        state="Delhi",
        evidence_image_path="outputs/evidence/veh1_evidence.jpg",
    )
    print(f"[Vehicle 1] Registered: {reg1['registry_id']} | Ref: {reg1['reference_no']} | Citizens: {reg1['citizen_count']}")
    assert reg1["citizen_count"] == 1

    # --- Vehicle 2: Same pothole, 8 meters away ---
    res2 = dedup.check_and_decide(
        lat=pothole_lat + 0.00005,  # ~5.5m north
        lon=pothole_lon + 0.00003,  # ~2.6m east
        detection_id="VEH2_DETECT_001",
        severity_score=0.70,
        max_depth_cm=5.5,
        evidence_image_path="outputs/evidence/veh2_evidence.jpg",
    )
    print(f"\n[Vehicle 2] Action: {res2['action']} | Should Dispatch: {res2['should_dispatch']} | Citizens: {res2['citizen_count']}")
    assert res2["should_dispatch"] is False, "Second vehicle should NOT dispatch a duplicate!"
    assert res2["action"] == "DUPLICATE_SUPPRESSED"
    assert res2["citizen_count"] == 2
    assert res2["reference_no"] == "NHAI/E/2026/10001"  # Same original reference!

    # --- Vehicles 3-5: Build to COMMUNITY_ALERT threshold ---
    for veh_num in range(3, 6):
        res = dedup.check_and_decide(
            lat=pothole_lat + 0.00002 * (veh_num - 2),
            lon=pothole_lon - 0.00001 * (veh_num - 2),
            detection_id=f"VEH{veh_num}_DETECT_001",
            severity_score=0.60,
            max_depth_cm=5.0 + veh_num * 0.1,
        )
        print(f"[Vehicle {veh_num}] Action: {res['action']} | Citizens: {res['citizen_count']} | Tier: {res['escalation_tier']}")
        assert res["should_dispatch"] is False

    assert res["citizen_count"] == 5
    assert res["escalation_tier"] == "COMMUNITY_ALERT"
    print("[OK] 5 citizens -> COMMUNITY_ALERT escalation tier reached!")

    # --- Vehicles 6-15: Build to HIGH_PRIORITY ---
    for veh_num in range(6, 16):
        res = dedup.check_and_decide(
            lat=pothole_lat + 0.00001 * (veh_num % 5),
            lon=pothole_lon,
            detection_id=f"VEH{veh_num}_DETECT_001",
            severity_score=0.55,
            max_depth_cm=5.3,
        )
    print(f"[Vehicle 15] Citizens: {res['citizen_count']} | Tier: {res['escalation_tier']}")
    assert res["citizen_count"] == 15
    assert res["escalation_tier"] == "HIGH_PRIORITY"
    print("[OK] 15 citizens -> HIGH_PRIORITY escalation tier reached!")

    # --- Vehicles 16-50: Build to CRITICAL_MASS ---
    for veh_num in range(16, 51):
        res = dedup.check_and_decide(
            lat=pothole_lat,
            lon=pothole_lon + 0.00001 * (veh_num % 3),
            detection_id=f"VEH{veh_num}_DETECT_001",
            severity_score=0.60,
            max_depth_cm=5.5,
        )
    print(f"[Vehicle 50] Citizens: {res['citizen_count']} | Tier: {res['escalation_tier']}")
    assert res["citizen_count"] == 50
    assert res["escalation_tier"] == "CRITICAL_MASS"
    print("[OK] 50 citizens -> CRITICAL_MASS escalation tier reached!")

    # --- Different pothole 200m away: should be treated as NEW ---
    res_new = dedup.check_and_decide(
        lat=pothole_lat + 0.0018,  # ~200m north
        lon=pothole_lon,
        detection_id="VEH_FAR_DETECT_001",
        severity_score=0.80,
        max_depth_cm=8.0,
    )
    print(f"\n[Far Vehicle] Pothole 200m away -> Action: {res_new['action']} | Should Dispatch: {res_new['should_dispatch']}")
    assert res_new["should_dispatch"] is True, "Pothole 200m away must be treated as new!"

    # --- Registry summary ---
    summary = dedup.get_registry_summary()
    print(f"\n[REGISTRY SUMMARY]")
    print(f"  Unique Potholes Registered : {summary['total_unique_potholes']}")
    print(f"  Active Grievances          : {summary['active_grievances']}")
    print(f"  Total Citizen Reports       : {summary['total_citizen_reports']}")
    print(f"  Duplicate Submissions Prevented: {summary['duplicate_submissions_prevented']}")
    print(f"  CRITICAL_MASS Escalations  : {summary['critical_mass_escalations']}")

    assert summary["total_unique_potholes"] == 1  # Only Pothole A registered (far one not registered yet)
    assert summary["duplicate_submissions_prevented"] == 49  # 50 citizens - 1 original = 49 prevented
    assert summary["critical_mass_escalations"] == 1

    print("\n  --> ALL CROWDSOURCED DEDUPLICATION TESTS PASSED!")

    # Cleanup
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)


def main():
    test_crowd_dedup()
    print("\n" + "=" * 70)
    print("  *** 100% SUCCESS: CROWDSOURCED SPATIAL DEDUPLICATION VERIFIED ***")
    print("=" * 70)


if __name__ == "__main__":
    main()

