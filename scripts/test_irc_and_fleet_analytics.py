"""
scripts/test_irc_and_fleet_analytics.py
=======================================
Verification suite for:
1. IRC (Indian Roads Congress) Pavement Distress Taxonomy (IRC:82-2015, IRC:SP:16).
2. ASTM D6433 Pavement Condition Index (PCI, scale 0-100).
3. Multi-Drive Pothole Growth Velocity and Degradation Modeling.
"""

import os
import sys
import numpy as np
import cv2

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.distress_taxonomy import PavementDistressClassifier
from src.fleet_analytics import PotholeDegradationTracker, haversine_distance_meters
from src.dashcam_agent import VehicleDashcamAutoReporter


def run_tests():
    print("=" * 75)
    print("  TEST: IRC DISTRESS TAXONOMY, PCI & FLEET DEGRADATION TRACKER")
    print("=" * 75)

    # 1. Test Haversine Distance
    print("\n[STEP 1] Testing Haversine Distance Calculation...")
    lat1, lon1 = 28.53831, 77.24312
    # Small offset (~10 meters north)
    lat2, lon2 = 28.53840, 77.24312
    dist_m = haversine_distance_meters(lat1, lon1, lat2, lon2)
    print(f"  [OK] Computed GPS Distance: {dist_m:.2f} meters")
    assert 5.0 < dist_m < 15.0, "Haversine should compute approx 10 meters!"

    # 2. Test IRC Distress Taxonomy Classification
    print("\n[STEP 2] Testing IRC Pavement Distress Taxonomy Classifier...")
    classifier = PavementDistressClassifier()

    # Case A: Pothole cavity (deep depression)
    pothole_patch = np.full((64, 64, 3), 120, dtype=np.uint8)
    cv2.circle(pothole_patch, (32, 32), 20, (30, 30, 30), -1)
    cand_pothole = {"x": 100, "y": 200, "width": 64, "height": 64, "severity_score": 0.8}
    depth_pothole = {"max_depth_cm": 6.8, "cavity_volume_liters": 4.5}

    res_pothole = classifier.classify_distress(pothole_patch, cand_pothole, depth_pothole)
    print(f"  [OK] Deep cavity classified as : {res_pothole['irc_category']} ({res_pothole['irc_code']})")
    print(f"       Action: {res_pothole['maintenance_action']}")
    print(f"       PCI Deduct Points: {res_pothole['pci_deduct_points']}")
    assert res_pothole["irc_category"] == "IRC:Pothole"

    # Case B: Linear Crack (high aspect ratio)
    linear_patch = np.full((120, 30, 3), 130, dtype=np.uint8)
    cv2.line(linear_patch, (15, 0), (15, 120), (40, 40, 40), 2)
    cand_linear = {"x": 200, "y": 150, "width": 30, "height": 120, "severity_score": 0.45}
    depth_linear = {"max_depth_cm": 1.5, "cavity_volume_liters": 0.5}

    res_linear = classifier.classify_distress(linear_patch, cand_linear, depth_linear)
    print(f"  [OK] Elongated crack classified: {res_linear['irc_category']} ({res_linear['irc_code']})")
    assert res_linear["irc_category"] == "IRC:Linear_Crack"

    # Case C: Alligator / Fatigue Cracking (high edge density)
    alligator_patch = np.full((64, 64, 3), 125, dtype=np.uint8)
    for i in range(10, 60, 10):
        cv2.line(alligator_patch, (0, i), (64, i), (30, 30, 30), 1)
        cv2.line(alligator_patch, (i, 0), (i, 64), (30, 30, 30), 1)
    cand_alli = {"x": 300, "y": 250, "width": 64, "height": 64, "severity_score": 0.6}
    depth_alli = {"max_depth_cm": 2.2, "cavity_volume_liters": 1.2}

    res_alli = classifier.classify_distress(alligator_patch, cand_alli, depth_alli)
    print(f"  [OK] Grid cracking classified  : {res_alli['irc_category']} ({res_alli['irc_code']})")
    assert res_alli["irc_category"] == "IRC:Alligator_Cracking"

    # 3. Test ASTM D6433 Pavement Condition Index (PCI)
    print("\n[STEP 3] Testing ASTM D6433 / IRC Pavement Condition Index (PCI)...")
    survey_defects = [res_pothole, res_linear, res_alli]
    pci_res = classifier.compute_pci_rating(survey_defects, road_length_km=1.0)
    print(f"  [OK] Pavement Condition Index : {pci_res['pci_score']} / 100")
    print(f"  [OK] Quality Rating           : {pci_res['rating']}")
    print(f"  [OK] Recommended PWD Strategy : {pci_res['pwd_strategy']}")
    assert 0 <= pci_res["pci_score"] <= 100

    # 4. Test Multi-Drive Fleet Degradation Tracker & Growth Velocity
    print("\n[STEP 4] Testing Fleet Degradation Tracker & Growth Velocity...")
    test_db = "data/test_fleet.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    tracker = PotholeDegradationTracker(db_path=test_db, cluster_radius_m=12.0)

    # Drive 1: Day 0 - Initial sighting of pothole
    sighting1 = tracker.record_sighting(
        lat=28.53831,
        lon=77.24312,
        depth_cm=3.5,
        volume_l=2.0,
        pixel_area=3000,
        road_name="Greater Kailash Road",
        irc_category="IRC:Pothole",
        timestamp="2026-10-01T10:00:00",
    )
    print(f"  [OK] Drive 1 Registered: Cluster={sighting1['cluster_id']} (Status: {sighting1['degradation_status']})")
    assert sighting1["is_new_cluster"] is True

    # Drive 2: Day 14 - Revisit same location (7 meters away, defect has grown)
    sighting2 = tracker.record_sighting(
        lat=28.53835,  # ~5 meters offset
        lon=77.24315,
        depth_cm=6.2,  # deepened
        volume_l=4.8,  # expanded volume
        pixel_area=4600,  # expanded area (+53.3%)
        road_name="Greater Kailash Road",
        irc_category="IRC:Pothole",
        timestamp="2026-10-15T10:00:00",
    )
    print(f"  [OK] Drive 2 Revisit Matched: Cluster={sighting2['cluster_id']}")
    print(f"       Revisit Count      : {sighting2['revisit_count']}")
    print(f"       Area Growth (%)    : +{sighting2['growth_percent']}%")
    print(f"       Volume Growth Rate : +{sighting2['growth_rate_l_week']} L/week")
    print(f"       Degradation Status : {sighting2['degradation_status']}")
    assert sighting2["is_new_cluster"] is False
    assert sighting2["revisit_count"] == 2
    assert sighting2["growth_percent"] > 30.0
    assert sighting2["degradation_status"] == "RAPID_DETERIORATION"

    # Drive 3: Day 25 - Severe monsoon collapse
    sighting3 = tracker.record_sighting(
        lat=28.53833,
        lon=77.24313,
        depth_cm=9.5,  # > 8.0 cm critical collapse threshold!
        volume_l=16.0,
        pixel_area=7200,
        road_name="Greater Kailash Road",
        irc_category="IRC:Pothole",
        timestamp="2026-10-26T10:00:00",
    )
    print(f"  [OK] Drive 3 Severe Collapse Flagged: Status = {sighting3['degradation_status']}")
    assert sighting3["degradation_status"] == "CRITICAL_COLLAPSE_RISK"

    # Fleet Summary
    summary = tracker.get_fleet_summary()
    print(f"  [OK] Fleet Analytics Summary: {summary['total_clusters']} clusters, {summary['critical_collapse_count']} critical collapse risk(s).")
    assert summary["critical_collapse_count"] >= 1

    # 5. Test Live Dashcam Reporter with IRC & Degradation
    print("\n[STEP 5] Testing Vehicle Dashcam Auto-Reporter with IRC & Degradation Integration...")
    sample_vid = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
    if os.path.exists(sample_vid):
        reporter = VehicleDashcamAutoReporter(
            target_portal="NHAI",
            min_severity="Medium",
            confirmation_frames=2,
            auto_dispatch=True,
            sample_interval=5,
        )

        cap = cv2.VideoCapture(sample_vid)
        frame_idx = 0
        dispatched_count = 0

        while cap.isOpened() and frame_idx < 25:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 5 != 0:
                continue

            resized = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            ann_frame, events, telem = reporter.process_frame(
                resized,
                frame_idx,
                timestamp_sec=frame_idx / 25.0,
                source_name="fleet_test_drive",
            )
            dispatched_count += len(events)

        cap.release()
        print(f"  [OK] Dashcam processed {frame_idx} frames, dispatched {dispatched_count} defect(s).")
        if reporter.dispatched_events:
            ev0 = reporter.dispatched_events[0]
            print(f"  [OK] Event Metadata Verified:")
            print(f"       IRC Category   : {ev0.get('irc_category')}")
            print(f"       IRC Code       : {ev0.get('irc_code')}")
            print(f"       PWD Action     : {ev0.get('maintenance_action')}")
            print(f"       Growth Status  : {ev0.get('degradation_status')}")
            print(f"       Complaint PDF  : {ev0.get('pdf_dossier')}")
            assert ev0.get("irc_category") is not None
            assert os.path.exists(ev0.get("pdf_dossier"))

    print("\n" + "=" * 75)
    print("  [SUCCESS] ALL IRC TAXONOMY, PCI & FLEET DEGRADATION TESTS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
