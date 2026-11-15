"""
scripts/test_power_and_manhole.py
=================================
Verification test suite for:
1. Thermal, Battery & Traffic-Aware Adaptive Compute Engine (src/edge_power_manager.py)
2. Civil Infrastructure & Sunken Manhole Classifier (src/manhole_detector.py)
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.edge_power_manager import EdgePowerThermalManager
from src.manhole_detector import InfrastructureAssetClassifier


class TestPowerAndManhole(unittest.TestCase):
    def setUp(self):
        self.power_mgr = EdgePowerThermalManager(simulated_temp_c=50.0, simulated_battery_pct=90.0)
        self.asset_clf = InfrastructureAssetClassifier()

    # -----------------------------------------------------------------------
    # 1. EDGE POWER & THERMAL MANAGER TESTS
    # -----------------------------------------------------------------------
    def test_thermal_compute_tiers(self):
        """Verify dynamic scaling across thermal regimes (High Performance, Balanced, ECO, Survival)."""
        # Normal temp: 50C -> Deep tier
        self.power_mgr.set_simulated_environment(temp_c=50.0)
        t_deep = self.power_mgr.determine_compute_tier()
        self.assertEqual(t_deep["tier"], EdgePowerThermalManager.TIER_DEEP)
        self.assertEqual(t_deep["feature_mode"], "deep")

        # Warm temp: 68C -> Balanced tier
        self.power_mgr.set_simulated_environment(temp_c=68.0)
        t_bal = self.power_mgr.determine_compute_tier()
        self.assertEqual(t_bal["tier"], EdgePowerThermalManager.TIER_BALANCED)

        # High summer heat: 77C -> ECO OpenCV texture tier (disables heavy PyTorch)
        self.power_mgr.set_simulated_environment(temp_c=77.0)
        t_eco = self.power_mgr.determine_compute_tier()
        self.assertEqual(t_eco["tier"], EdgePowerThermalManager.TIER_ECO)
        self.assertEqual(t_eco["feature_mode"], "texture_only")

        # Dangerous heat: 85C -> Critical survival tier (IMU shock only)
        self.power_mgr.set_simulated_environment(temp_c=85.0)
        t_surv = self.power_mgr.determine_compute_tier()
        self.assertEqual(t_surv["tier"], EdgePowerThermalManager.TIER_SURVIVAL)
        self.assertFalse(t_surv["allow_continuous_vision"])

    def test_traffic_standstill_throttling(self):
        """Verify that standing in a 0 km/h traffic jam throttles compute to save power."""
        self.power_mgr.set_simulated_environment(temp_c=55.0)

        # Frame 1 at 0 km/h: processes keyframe
        proc1, reason1, _ = self.power_mgr.should_process_frame(vehicle_speed_kmh=0.0, current_time_sec=10.0)
        self.assertTrue(proc1)

        # Frame 2 arriving 0.5s later at 0 km/h: should skip (asphalt has not changed)
        proc2, reason2, _ = self.power_mgr.should_process_frame(vehicle_speed_kmh=0.0, current_time_sec=10.5)
        self.assertFalse(proc2)
        self.assertEqual(reason2, "TRAFFIC_STANDSTILL_THROTTLE")

        # Frame 3 arriving 4.0s later: allowed
        proc3, reason3, _ = self.power_mgr.should_process_frame(vehicle_speed_kmh=0.0, current_time_sec=14.5)
        self.assertTrue(proc3)

    # -----------------------------------------------------------------------
    # 2. CIVIL INFRASTRUCTURE & SUNKEN MANHOLE TESTS
    # -----------------------------------------------------------------------
    def test_flush_manhole_cover_suppression(self):
        """Verify that a flush legal circular manhole (depth < 2.5cm) is suppressed from PWD complaint."""
        img = np.ones((120, 120, 3), dtype=np.uint8) * 110
        # Draw circular cast-iron manhole cover
        cv2.circle(img, (60, 60), 45, (60, 60, 60), -1)
        cv2.circle(img, (60, 60), 45, (20, 20, 20), 3)  # rim

        res = self.asset_clf.evaluate_asset(img, measured_depth_cm=0.8)
        self.assertTrue(res["is_manhole"])
        self.assertFalse(res["is_sunken_hazard"])
        self.assertTrue(res["suppress_grievance"])
        self.assertEqual(res["classification"], "MANHOLE_FLUSH_LEGAL")

    def test_sunken_manhole_hazard_escalation(self):
        """Verify that a sunken manhole (depth >= 2.5cm) is flagged as an IRC hazard targeted to Water Board."""
        img = np.ones((120, 120, 3), dtype=np.uint8) * 110
        cv2.circle(img, (60, 60), 45, (40, 40, 40), -1)
        cv2.circle(img, (60, 60), 45, (10, 10, 10), 4)

        # Measured depth is 4.8 cm (severe drop)
        res = self.asset_clf.evaluate_asset(img, measured_depth_cm=4.8)
        self.assertTrue(res["is_manhole"])
        self.assertTrue(res["is_sunken_hazard"])
        self.assertFalse(res["suppress_grievance"])
        self.assertEqual(res["classification"], "IRC:SUNKEN_MANHOLE_HAZARD")
        self.assertIn("Water Supply and Sewerage", res["target_authority"])

    def test_utility_trench_excavation(self):
        """Verify rectangular utility trench detection."""
        # Long narrow trench cut (w=340, h=60)
        img = np.ones((60, 340, 3), dtype=np.uint8) * 90
        res = self.asset_clf.evaluate_asset(img, measured_depth_cm=3.2)
        self.assertEqual(res["classification"], "IRC:UTILITY_TRENCH_EXCAVATION")
        self.assertFalse(res["suppress_grievance"])


if __name__ == "__main__":
    unittest.main()
