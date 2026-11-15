"""
scripts/test_citizen_notifier.py
================================
Test suite for Citizen WhatsApp & SMS Grievance Tracker & Notification Engine.
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.citizen_notifier import CitizenNotifier


class TestCitizenNotifier(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_db = os.path.join(self.tmp_dir.name, "test_pothole.db")
        self.notifier = CitizenNotifier(db_path=self.test_db, default_phone="+919876543210")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_phone_cleaning(self):
        """Verify phone numbers are cleaned into WhatsApp-compatible format."""
        self.assertEqual(self.notifier.clean_phone_number("+91 98765-43210"), "919876543210")
        self.assertEqual(self.notifier.clean_phone_number("9876543210"), "9876543210")
        self.assertEqual(self.notifier.clean_phone_number(""), "")

    def test_google_maps_url(self):
        """Verify Google Maps pin link generation."""
        url = self.notifier.generate_google_maps_url(28.6139, 77.2090)
        self.assertIn("https://maps.google.com/?q=28.613900,77.209000", url)

    def test_grievance_filed_notification(self):
        """Verify message formatting and click-to-chat link for new grievance."""
        event = {
            "detection_id": "DET_TEST_001",
            "reference_no": "MoRTH/E/2026/99887",
            "target_portal": "MoRTH CPGRAMS",
            "road_name": "Outer Ring Road, Block B",
            "district": "South Delhi",
            "state": "Delhi",
            "latitude": 28.5492,
            "longitude": 77.2185,
            "max_depth_cm": 6.4,
            "cavity_volume_liters": 15.2,
            "severity_category": "High",
            "irc_category": "IRC:Pothole",
            "threat_index": 82.5,
        }

        res = self.notifier.notify(event, notification_type="GRIEVANCE_FILED")

        self.assertEqual(res["notification_type"], "GRIEVANCE_FILED")
        self.assertEqual(res["reference_no"], "MoRTH/E/2026/99887")
        self.assertIn("MoRTH/E/2026/99887", res["whatsapp_text"])
        self.assertIn("api.whatsapp.com/send", res["whatsapp_url"])
        self.assertIn("Outer Ring Road", res["sms_text"])
        self.assertTrue(len(res["sms_text"]) <= 200)

        # Check DB persistence
        history = self.notifier.get_notification_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["reference_no"], "MoRTH/E/2026/99887")
        self.assertEqual(history[0]["notification_type"], "GRIEVANCE_FILED")

    def test_corroboration_notification(self):
        """Verify message formatting for citizen corroboration."""
        event = {
            "detection_id": "DET_TEST_002",
            "reference_no": "NHAI/E/2026/11223",
            "road_name": "NH-48 Jaipur Highway",
            "latitude": 27.9123,
            "longitude": 76.5432,
            "max_depth_cm": 5.1,
            "citizen_count": 6,
            "escalation_tier": "COMMUNITY_ALERT",
        }

        res = self.notifier.notify(event, notification_type="CORROBORATION_CONFIRMED")

        self.assertEqual(res["notification_type"], "CORROBORATION_CONFIRMED")
        self.assertIn("6 independent drivers", res["whatsapp_text"])
        self.assertIn("COMMUNITY ALERT", res["whatsapp_text"])
        self.assertIn("NHAI/E/2026/11223", res["sms_text"])

        # Check DB count
        history = self.notifier.get_notification_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["notification_type"], "CORROBORATION_CONFIRMED")

    def test_offline_buffered_notification(self):
        """Verify message formatting for offline buffered drive."""
        event = {
            "detection_id": "DET_OFFLINE_003",
            "reference_no": "OFFLINE_QUEUED",
            "road_name": "Khardung La Pass Dead-Zone",
            "latitude": 34.2789,
            "longitude": 77.6045,
        }

        res = self.notifier.notify(event, notification_type="AUTO")
        self.assertEqual(res["notification_type"], "OFFLINE_QUEUED")
        self.assertIn("OFFLINE DRIVE", res["whatsapp_text"])
        self.assertIn("Zero data loss", res["whatsapp_text"])


if __name__ == "__main__":
    unittest.main()
