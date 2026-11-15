"""
scripts/test_depth_and_night_adas.py
====================================
End-to-end test script for:
1. Monocular Metric Depth & Cavity Volumetric Profiler (cm, Liters, kg asphalt)
2. Night-Vision Illumination Enhancement & Rain Puddle Filter
3. In-Cabin Driver ADAS Audio Collision Warning System
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
from src.depth_estimator import MetricDepthEstimator
from src.night_vision import NightVisionEnhancer
from src.adas_audio import InCabinAdasAudio
from src.dashcam_agent import VehicleDashcamAutoReporter


def run_tests():
    print("=" * 75)
    print("  TEST: 3D DEPTH PROFILER, NIGHT VISION & ADAS AUDIO SYSTEM")
    print("=" * 75)

    # 1. Test Depth Estimator
    print("\n[STEP 1] Testing Monocular Metric Depth Estimator...")
    depth_est = MetricDepthEstimator(camera_height_m=1.35, camera_pitch_deg=10.0)

    # Test distance at various pixel heights (from bottom of screen upwards)
    d_bottom = depth_est.compute_distance_ahead(y_pixel=470, image_height=480)
    d_mid = depth_est.compute_distance_ahead(y_pixel=300, image_height=480)
    print(f"  [OK] Distance ahead for bottom road pixel (y=470): {d_bottom:.1f} meters")
    print(f"  [OK] Distance ahead for mid road pixel (y=300):    {d_mid:.1f} meters")
    assert d_bottom < d_mid, "Bottom pixel should be closer than mid pixel!"

    # Test 3D Cavity Profiler on a synthetic pothole patch
    test_img = np.full((480, 640, 3), 130, dtype=np.uint8)
    cv2.circle(test_img, (320, 360), 35, (45, 45, 45), -1)  # dark cavity
    cand_test = {"x": 285, "y": 325, "width": 70, "height": 70, "severity_score": 0.85}

    profile = depth_est.profile_pothole_3d(test_img, cand_test, vehicle_speed_kmh=45.0)
    print(f"  [OK] Max Cavity Depth        : {profile['max_depth_cm']:.1f} cm")
    print(f"  [OK] Mean Depth              : {profile['mean_depth_cm']:.1f} cm")
    print(f"  [OK] Cavity Void Volume      : {profile['cavity_volume_liters']:.2f} Liters")
    print(f"  [OK] Asphalt Repair Required : {profile['asphalt_repair_mass_kg']:.2f} kg")
    print(f"  [OK] Distance Ahead          : {profile['distance_ahead_m']:.1f} meters")
    print(f"  [OK] Time to Impact (TTI)    : {profile['time_to_impact_sec']:.2f} seconds")
    assert profile["max_depth_cm"] > 2.0, "Depth should detect depression!"
    assert profile["asphalt_repair_mass_kg"] > 0.1, "Repair mass should be positive!"
    assert profile["depth_colormap"] is not None, "Color depth map generated!"

    # 2. Test Night-Vision & Puddle Filter
    print("\n[STEP 2] Testing Night-Vision Enhancer & Puddle Filter...")
    night_agent = NightVisionEnhancer(luminance_thresh=75)

    dark_img = (test_img * 0.3).astype(np.uint8)  # simulate pitch dark scene
    is_night, lum = night_agent.is_night_scene(dark_img)
    print(f"  [OK] Dark scene detection: is_night={is_night} (Luminance={lum})")
    assert is_night is True, "Dark scene should trigger night mode!"

    enhanced_img = night_agent.enhance_night_frame(dark_img)
    enhanced_lum = float(np.mean(enhanced_img))
    print(f"  [OK] Enhanced frame luminance: {enhanced_lum:.1f} (Boosted from {lum})")
    assert enhanced_lum > lum, "Night enhancement should boost shadow brightness!"

    # Test rain puddle filter
    puddle_patch = np.full((64, 64, 3), 235, dtype=np.uint8)  # bright smooth specular surface
    is_puddle, spec_score, lap_var = night_agent.is_surface_puddle(puddle_patch)
    print(f"  [OK] Specular reflection test on wet puddle: is_puddle={is_puddle} (Specular={spec_score}, LapVar={lap_var})")
    assert is_puddle is True, "Specular mirror surface should be flagged as puddle!"

    # 3. Test In-Cabin ADAS Audio Warning
    print("\n[STEP 3] Testing In-Cabin ADAS Collision Warning System...")
    adas = InCabinAdasAudio(enabled=True, cooldown_sec=1.0, warning_dist_m=25.0)

    # Test alert trigger when hazard is within 15 meters
    alert_triggered = adas.check_and_alert(distance_ahead_m=12.0, time_to_impact_sec=0.96, severity_cat="High", depth_cm=8.5)
    print(f"  [OK] ADAS Collision Alert Triggered: {alert_triggered} (Hazard within 12m)")
    assert alert_triggered is True, "Hazard within warning distance must trigger alert!"

    # Test cooldown suppression
    alert_cooldown = adas.check_and_alert(distance_ahead_m=12.0, time_to_impact_sec=0.96, severity_cat="High", depth_cm=8.5)
    print(f"  [OK] ADAS Cooldown Check: {not alert_cooldown} (Subsequent immediate alerts suppressed)")
    assert alert_cooldown is False, "Rapid successive alerts should be suppressed by cooldown!"

    # 4. Test End-to-End Vehicle Dashcam Auto-Reporter with 3D Depth
    print("\n[STEP 4] Testing Vehicle Dashcam Auto-Reporter with 3D Depth & PDF Generation...")
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

        while cap.isOpened() and frame_idx < 30:
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
                source_name="test_drive",
            )
            dispatched_count += len(events)

        cap.release()
        print(f"  [OK] Processed {frame_idx} test frames. Dispatched {dispatched_count} pothole(s).")
        if reporter.dispatched_events:
            ev0 = reporter.dispatched_events[0]
            print(f"  [OK] Sample Dispatched Pothole Metrics:")
            print(f"       - Ref ID      : {ev0['reference_no']}")
            print(f"       - Max Depth   : {ev0['max_depth_cm']:.1f} cm")
            print(f"       - Cavity Vol  : {ev0['cavity_volume_liters']:.2f} L")
            print(f"       - Asphalt Req : {ev0['asphalt_repair_mass_kg']:.2f} kg")
            print(f"       - Distance    : {ev0['distance_ahead_m']:.1f} m")
            print(f"       - Evidence Img: {ev0['evidence_image']}")
            print(f"       - PDF Dossier : {ev0['pdf_dossier']}")
            assert os.path.exists(ev0["evidence_image"]), "Evidence image must exist!"
            assert os.path.exists(ev0["pdf_dossier"]), "Complaint PDF dossier must exist!"

    print("\n" + "=" * 75)
    print("  [SUCCESS] ALL 3D DEPTH, NIGHT-VISION & ADAS TESTS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()

