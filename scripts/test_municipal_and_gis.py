"""
scripts/test_municipal_and_gis.py
=================================
Verification test suite for Municipal PWD Work Orders, SLA Enforcement,
Repair Verification, and 3D Google Earth KML / Smart City Feeds.
"""

import os
import sys
import tempfile
import unittest
import pandas as pd
import xml.etree.ElementTree as ET

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.municipal_workflow import MunicipalWorkOrderManager
from src.gis_export import (
    export_defects_to_geojson,
    geojson_to_bytes,
    export_defects_to_kml,
    kml_to_bytes,
    export_smart_city_wms_feed,
)


class TestMunicipalAndGIS(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_db = os.path.join(self.tmp_dir.name, "test_potholes.db")
        self.mgr = MunicipalWorkOrderManager(db_path=self.test_db)

        # Sample defect dataframe
        self.sample_df = pd.DataFrame([
            {
                "detection_id": "DET_KML_001",
                "track_id": "TRK_001",
                "latitude": 28.5383,
                "longitude": 77.2431,
                "severity_category": "High",
                "severity_score": 0.88,
                "anomaly_score": -0.35,
                "max_depth_cm": 7.4,
                "cavity_volume_liters": 18.2,
                "grievance_ref_no": "NHAI/E/2026/88771",
                "timestamp": "2026-11-14T09:00:00",
            },
            {
                "detection_id": "DET_KML_002",
                "track_id": "TRK_002",
                "latitude": 28.5395,
                "longitude": 77.2450,
                "severity_category": "Medium",
                "severity_score": 0.54,
                "anomaly_score": -0.18,
                "max_depth_cm": 4.1,
                "cavity_volume_liters": 8.5,
                "grievance_ref_no": "NHAI/E/2026/88772",
                "timestamp": "2026-11-14T09:05:00",
            },
            {
                "detection_id": "DET_KML_003",
                "track_id": "TRK_003",
                "latitude": 28.5410,
                "longitude": 77.2480,
                "severity_category": "Low",
                "severity_score": 0.22,
                "anomaly_score": 0.05,
                "max_depth_cm": 2.2,
                "cavity_volume_liters": 3.0,
                "grievance_ref_no": "NHAI/E/2026/88773",
                "timestamp": "2026-11-14T09:10:00",
            },
        ])

    def tearDown(self):
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    def test_municipal_work_order_lifecycle(self):
        """Test full PWD work order issuance, BOQ costing, SLA tracking, and repair closure."""
        # 1. Issue Work Order
        wo = self.mgr.create_work_order(
            reference_no="NHAI/E/2026/88771",
            road_name="National Highway 48 (Sector 29)",
            district="Gurugram",
            state="Haryana",
            latitude=28.5383,
            longitude=77.2431,
            max_depth_cm=7.4,
            cavity_volume_liters=18.2,
            asphalt_repair_mass_kg=43.7,
        )

        self.assertIn("PWD/WO/2026/", wo["work_order_id"])
        self.assertEqual(wo["status"], "WORK_ORDER_ISSUED")
        self.assertTrue(wo["estimated_cost_inr"] > 4000.0)

        # 2. Check SLA Evaluation
        sla_info = self.mgr.check_sla(wo)
        self.assertEqual(sla_info["sla_status"], "WITHIN_SLA")
        self.assertTrue(sla_info["hours_remaining"] <= 72.0)

        # 3. Generate Printable Work Order PDF
        pdf_path = os.path.join(self.tmp_dir.name, "test_wo.pdf")
        gen_path = self.mgr.generate_work_order_pdf(wo["work_order_id"], output_path=pdf_path)
        self.assertTrue(os.path.exists(gen_path))
        self.assertTrue(os.path.getsize(gen_path) > 1000)

        # 4. Verify & Close Repair
        close_res = self.mgr.verify_and_close_repair(
            work_order_id=wo["work_order_id"],
            engineer_notes="Patched with hot mix bitumen and compacted with 2-ton vibratory roller.",
            notify_citizens=True,
        )

        self.assertEqual(close_res["status"], "REPAIRED_VERIFIED")
        self.assertIn("GRIEVANCE RESOLVED", close_res["citizen_notification"]["whatsapp_text"])

        # 5. Check List / DataFrame
        df = self.mgr.get_all_work_orders_df()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["status"], "REPAIRED_VERIFIED")

    def test_3d_google_earth_kml_export(self):
        """Verify 3D Google Earth KML generation with extruded polygonal cylinders."""
        kml_path = os.path.join(self.tmp_dir.name, "test_3d.kml")
        out_path = export_defects_to_kml(self.sample_df, output_path=kml_path, extrude_3d=True)
        self.assertTrue(os.path.exists(out_path))

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify XML structure and 3D KML tags
        self.assertIn("<?xml version=", content)
        self.assertIn("<kml xmlns=", content)
        self.assertIn("<extrude>1</extrude>", content)
        self.assertIn("<altitudeMode>relativeToGround</altitudeMode>", content)
        self.assertIn("potholeHigh", content)
        self.assertIn("DET_KML_001", content)

        # Verify in-memory bytes
        kml_bytes = kml_to_bytes(self.sample_df, extrude_3d=True)
        self.assertTrue(len(kml_bytes) > 500)

    def test_smart_city_wms_feed(self):
        """Verify Smart City ICCC protocol payload structure."""
        feed = export_smart_city_wms_feed(self.sample_df)
        self.assertEqual(feed["smart_city_system"], "National Highway & Urban Road Surface Monitoring Grid")
        self.assertEqual(feed["total_active_hazards"], 3)
        self.assertEqual(feed["critical_priority_hazards"], 1)
        self.assertEqual(len(feed["incidents"]), 3)
        self.assertEqual(feed["incidents"][0]["incident_id"], "DET_KML_001")


if __name__ == "__main__":
    unittest.main()
