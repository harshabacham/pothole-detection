"""
scripts/test_real_life_modules.py
=================================
Verification test suite for real-world automotive edge case modules:
1. Speed Breaker & Rumble Strip vs Pothole Discriminator (src/speed_breaker_classifier.py)
2. Windshield Wiper & Specular Sun Glare Filter (src/weather_lens_filter.py)
3. Connected Vehicle (V2X) Early Hazard Radar & Braking Engine (src/v2x_hazard_broadcaster.py)
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.speed_breaker_classifier import SpeedBreakerDiscriminator
from src.weather_lens_filter import WeatherLensFilter
from src.v2x_hazard_broadcaster import V2XHazardRadar, haversine_distance_m, calculate_bearing_deg
from src.database import get_db_connection, init_database
from src.crowd_dedup import CrowdsourcedSpatialDeduplicator


class TestRealLifeModules(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_db = os.path.join(self.tmp_dir.name, "test_real_life.db")
        init_database(self.test_db)
        CrowdsourcedSpatialDeduplicator(db_path=self.test_db)

        self.speed_breaker_clf = SpeedBreakerDiscriminator()
        self.weather_filter = WeatherLensFilter(min_wiper_length=80)
        self.v2x_radar = V2XHazardRadar(db_path=self.test_db, enable_audio=False)

    def tearDown(self):
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 1. SPEED BREAKER & RUMBLE STRIP DISCRIMINATOR TESTS
    # -----------------------------------------------------------------------
    def test_convex_speed_breaker_classification(self):
        """Verify that a wide convex bump is classified as SPEED_BREAKER_CONVEX and suppressed."""
        # Create a synthetic speed breaker image (transverse bright crest in middle)
        h, w = 60, 320
        bump_img = np.ones((h, w, 3), dtype=np.uint8) * 90
        # Add bright illuminated crest in middle rows
        bump_img[20:38, :] = 190
        # Trailing shadow
        bump_img[38:50, :] = 50

        bbox = (100, 300, 320, 60)  # Aspect ratio 5.33 (lane spanning)
        imu_bump_wave = [1.0, 1.35, 1.55, 0.85, 0.95, 1.0]  # Compression then rebound

        res = self.speed_breaker_clf.classify_defect_patch(
            patch_bgr=bump_img,
            bbox=bbox,
            frame_width=640,
            imu_shock_profile=imu_bump_wave,
        )

        self.assertEqual(res["classification"], "SPEED_BREAKER_CONVEX")
        self.assertFalse(res["is_hazard"])
        self.assertTrue(res["suppress_grievance"])
        self.assertTrue(res["aspect_ratio"] > 3.0)

    def test_concave_pothole_classification(self):
        """Verify that a circular dark depression is classified as POTHOLE and not suppressed."""
        h, w = 80, 90
        pothole_img = np.ones((h, w, 3), dtype=np.uint8) * 140
        # Draw dark inner depression cavity
        cv2.circle(pothole_img, (45, 40), 28, (35, 35, 35), -1)

        bbox = (260, 340, 90, 80)  # Aspect ratio 1.12
        imu_pothole_wave = [1.0, 0.70, 0.60, 1.85, 1.20, 1.0]  # Drop then impact

        res = self.speed_breaker_clf.classify_defect_patch(
            patch_bgr=pothole_img,
            bbox=bbox,
            frame_width=640,
            imu_shock_profile=imu_pothole_wave,
        )

        self.assertEqual(res["classification"], "POTHOLE")
        self.assertTrue(res["is_hazard"])
        self.assertFalse(res["suppress_grievance"])

    def test_rumble_strip_periodicity(self):
        """Verify that multiple parallel transverse bands are classified as RUMBLE_STRIP_PERIODIC."""
        h, w = 120, 280
        rumble_img = np.ones((h, w, 3), dtype=np.uint8) * 90
        # Draw 4 bright parallel horizontal bars
        for r_top in [15, 40, 65, 90]:
            rumble_img[r_top : r_top + 10, :] = 210

        bbox = (150, 300, 280, 120)
        res = self.speed_breaker_clf.classify_defect_patch(
            patch_bgr=rumble_img,
            bbox=bbox,
            frame_width=640,
        )

        self.assertEqual(res["classification"], "RUMBLE_STRIP_PERIODIC")
        self.assertTrue(res["suppress_grievance"])
        self.assertTrue(res["rumble_peak_count"] >= 3)

    # -----------------------------------------------------------------------
    # 2. WINDSHIELD WIPER & SUN GLARE TESTS
    # -----------------------------------------------------------------------
    def test_wiper_blade_detection_and_suppression(self):
        """Verify that a sweeping diagonal windshield wiper is detected and occluded defects are rejected."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 110
        # Draw a dark diagonal wiper blade across windshield (angle ~45 degrees)
        cv2.line(frame, (80, 440), (380, 160), (20, 20, 20), thickness=16)

        is_present, wiper_mask, lines = self.weather_filter.detect_wiper_blade(frame)
        self.assertTrue(is_present)
        self.assertIsNotNone(wiper_mask)
        self.assertTrue(len(lines) > 0)

        # Defect box directly over the wiper blade
        defect_on_wiper = (200, 270, 60, 60)
        is_occluded = self.weather_filter.is_defect_occluded_by_wiper(defect_on_wiper, wiper_mask)
        self.assertTrue(is_occluded)

        # Defect box away from the wiper blade
        defect_away = (520, 400, 50, 50)
        is_occluded_away = self.weather_filter.is_defect_occluded_by_wiper(defect_away, wiper_mask)
        self.assertFalse(is_occluded_away)

    def test_sun_glare_neutralization(self):
        """Verify detection and attenuation of saturated specular solar glare."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        # Add saturated glare spot (pure white highlights)
        frame[100:220, 200:380] = 255

        enhanced_frame, glare_detected, glare_pct = self.weather_filter.neutralize_sun_glare(frame)
        self.assertTrue(glare_detected)
        self.assertTrue(glare_pct > 1.5)
        # Saturated glare should be attenuated below 255 in enhanced frame
        self.assertTrue(np.mean(enhanced_frame[120:180, 220:340]) < 255)

    # -----------------------------------------------------------------------
    # 3. V2X EARLY HAZARD RADAR & STOPPING DISTANCE TESTS
    # -----------------------------------------------------------------------
    def test_physics_stopping_distance(self):
        """Verify physics-based stopping distance calculation at various speeds."""
        dry_60 = self.v2x_radar.calculate_stopping_distance_m(speed_kmh=60.0, is_wet=False)
        wet_60 = self.v2x_radar.calculate_stopping_distance_m(speed_kmh=60.0, is_wet=True)

        self.assertAlmostEqual(dry_60["reaction_distance_m"], 20.0, delta=1.0)
        self.assertTrue(wet_60["total_stopping_distance_m"] > dry_60["total_stopping_distance_m"])
        self.assertTrue(dry_60["safe_warning_horizon_m"] >= 100.0)

    def test_v2x_geofence_scanning_and_alert(self):
        """Verify scanning ahead discovers registered road hazards within warning horizon."""
        # Insert a verified severe pothole 180m ahead of the car into grievance_registry
        conn = get_db_connection(self.test_db)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO grievance_registry (
                registry_id, latitude, longitude, road_name, severity_category,
                max_depth_cm, cavity_volume_liters, citizen_corroboration_count,
                original_reference_no, is_resolved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            "REG_NH48_HAZARD_01",
            28.5400, 77.2431,  # ~188 meters north
            "NH-48 Expressway",
            "High", 7.5, 18.0, 12,
            "NHAI/E/2026/99881",
        ))
        conn.commit()
        conn.close()

        # Car is at (28.5383, 77.2431) moving North (Heading = 0.0 deg) at 70 km/h
        car_lat, car_lon = 28.5383, 77.2431
        alerts = self.v2x_radar.scan_ahead_for_hazards(
            current_lat=car_lat,
            current_lon=car_lon,
            current_speed_kmh=70.0,
            current_heading_deg=0.0,  # North
        )

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert["reference_no"], "NHAI/E/2026/99881")
        self.assertEqual(alert["severity_category"], "High")
        self.assertTrue(150.0 < alert["distance_m"] < 250.0)
        self.assertEqual(alert["recommended_speed_kmh"], 30.0)


if __name__ == "__main__":
    unittest.main()
