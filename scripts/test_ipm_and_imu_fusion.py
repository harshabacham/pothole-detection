"""
scripts/test_ipm_and_imu_fusion.py
==================================
Comprehensive Verification Test for:
1. Inverse Perspective Mapping (IPM) & Bird's Eye View (BEV) Transformer (src/ipm_transform.py)
2. Optical-Inertial Shock Correlator & Accelerometer Fusion (src/imu_fusion.py)
"""

import os
import sys
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ipm_transform import InversePerspectiveMapper
from src.imu_fusion import OpticalInertialShockCorrelator


def test_ipm_transformation():
    print("\n" + "=" * 70)
    print("  [TEST 1] VERIFYING INVERSE PERSPECTIVE MAPPING (IPM) ENGINE")
    print("=" * 70)

    ipm = InversePerspectiveMapper()
    img_w, img_h = 640, 480
    dummy_frame = np.ones((img_h, img_w, 3), dtype=np.uint8) * 60

    # 1. Warp test
    bev = ipm.warp_to_bev(dummy_frame)
    assert bev.shape == (ipm.bev_h, ipm.bev_w, 3), f"Unexpected BEV shape: {bev.shape}"
    print(f"[OK] Perspective Warp: Successfully warped {img_w}x{img_h} -> {ipm.bev_w}x{ipm.bev_h} BEV plane.")

    # 2. Camera-to-BEV coordinate mapping
    # Bottom center (close ahead)
    x_bev, y_bev, xm_center, ym_near = ipm.camera_to_bev_coords(320, 460, img_w, img_h)
    print(f"[OK] Near Center Point (320, 460) -> BEV: ({x_bev}, {y_bev}) | Metric: ({xm_center}m, {ym_near}m)")
    assert abs(xm_center) < 0.5, f"Expected center near 0m, got {xm_center}"
    assert ym_near < 8.0, f"Expected near distance < 8m, got {ym_near}"

    # Far center (20m ahead)
    _, _, _, ym_far = ipm.camera_to_bev_coords(320, 250, img_w, img_h)
    print(f"[OK] Far Center Point (320, 250) -> Distance ahead: {ym_far}m")
    assert ym_far > ym_near, "Far distance must be greater than near distance"

    # 3. Lateral Wheel Track Classification
    left_track = ipm.classify_wheel_track(-0.85)
    center_track = ipm.classify_wheel_track(0.05)
    right_track = ipm.classify_wheel_track(0.90)

    print(f"[OK] Lateral -0.85m -> {left_track['wheel_hazard']} (Multiplier: {left_track['severity_multiplier']}x)")
    print(f"[OK] Lateral +0.05m -> {center_track['wheel_hazard']} (Multiplier: {center_track['severity_multiplier']}x)")
    print(f"[OK] Lateral +0.90m -> {right_track['wheel_hazard']} (Multiplier: {right_track['severity_multiplier']}x)")

    assert left_track["wheel_hazard"] == "LEFT_WHEEL_PATH"
    assert center_track["wheel_hazard"] == "LANE_CENTER"
    assert right_track["wheel_hazard"] == "RIGHT_WHEEL_PATH"

    # 4. Metric Surface Area (cm²)
    test_bbox = (280, 400, 80, 50)
    metric_area = ipm.compute_metric_surface_area(test_bbox, img_w, img_h)
    print(f"[OK] Bounding Box {test_bbox} -> Perspective-Invariant Area: {metric_area} cm²")
    assert metric_area > 100.0, "Area should be physically positive"

    # 5. Radar Widget Rendering
    radar_hud = ipm.render_bev_radar_widget(
        defects_list=[{"bbox": test_bbox}],
        img_w=img_w,
        img_h=img_h,
    )
    assert radar_hud.shape == (240, 160, 3)
    print(f"[OK] BEV Mini-Radar HUD: Rendered {radar_hud.shape[1]}x{radar_hud.shape[0]} ADAS display widget.")
    print("  --> ALL IPM TESTS PASSED!")


def test_optical_inertial_fusion():
    print("\n" + "=" * 70)
    print("  [TEST 2] VERIFYING OPTICAL-INERTIAL SHOCK FUSION & CORRELATOR")
    print("=" * 70)

    correlator = OpticalInertialShockCorrelator()

    # Scenario: Vehicle driving at 45 km/h (12.5 m/s)
    # Detects pothole A 12.5 meters ahead at t=1.0s (in LEFT_WHEEL_PATH)
    # Expected arrival at t = 1.0 + (12.5 / 12.5) = 2.0s
    t_start = 1.0
    correlator.register_optical_candidate(
        track_id="POTHOLE_TRACK_01",
        detection_time_sec=t_start,
        distance_ahead_m=12.5,
        speed_kmh=45.0,
        wheel_track_hazard="LEFT_WHEEL_PATH",
        visual_severity_score=75.0,
        cavity_volume_liters=4.2,
    )

    # Detects pothole B 12.5 meters ahead at t=1.0s (in LANE_CENTER)
    # Expected arrival at t = 2.0s
    correlator.register_optical_candidate(
        track_id="POTHOLE_TRACK_02",
        detection_time_sec=t_start,
        distance_ahead_m=12.5,
        speed_kmh=45.0,
        wheel_track_hazard="LANE_CENTER",
        visual_severity_score=50.0,
        cavity_volume_liters=2.0,
    )

    # Stream accelerometer telemetry from t=1.0s to t=3.0s
    for t_step in np.linspace(1.0, 3.0, 40):
        # Simulate severe physical pothole shock spike at t=2.05s on the left tire!
        is_hit = (2.00 <= t_step <= 2.10)
        correlator.simulate_realistic_telemetry(
            timestamp_sec=t_step,
            speed_kmh=45.0,
            active_impact=is_hit,
            impact_severity="High",
        )

    # Evaluate correlation at t=2.6s (past arrival window)
    resolved = correlator.evaluate_correlations(current_time_sec=2.6)
    print(f"[OK] Correlator processed {len(resolved)} defect events.")
    assert len(resolved) == 2, f"Expected 2 resolved events, got {len(resolved)}"

    res_map = {r["track_id"]: r for r in resolved}

    # Verify Track 01 (Left Wheel Path -> Hit with shock)
    hit_res = res_map["POTHOLE_TRACK_01"]
    print(f"[OK] Pothole 01 (Left Wheel): Status={hit_res['fusion_status']} | Shock={hit_res['max_gz_shock']}g | Threat Index={hit_res['threat_index']}/100")
    assert hit_res["fusion_status"] == "CONFIRMED_PHYSICAL_IMPACT"
    assert hit_res["confidence"] > 0.99
    assert hit_res["max_gz_shock"] > 1.5

    # Verify Track 02 (Lane Center -> Straddled between wheels, no rim shock)
    center_res = res_map["POTHOLE_TRACK_02"]
    print(f"[OK] Pothole 02 (Center): Status={center_res['fusion_status']} | Impact={center_res['impact_label']} | Threat Index={center_res['threat_index']}/100")
    assert center_res["fusion_status"] == "STRADDLED_CENTER"
    assert center_res["confidence"] >= 0.90

    print("  --> ALL OPTICAL-INERTIAL FUSION TESTS PASSED!")


def main():
    test_ipm_transformation()
    test_optical_inertial_fusion()
    print("\n" + "=" * 70)
    print("  *** 100% SUCCESS: IPM & IMU SENSOR FUSION ENGINES FULLY VERIFIED ***")
    print("=" * 70)


if __name__ == "__main__":
    main()
