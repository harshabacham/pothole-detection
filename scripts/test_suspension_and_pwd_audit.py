"""
scripts/test_suspension_and_pwd_audit.py
========================================
Comprehensive Verification Test for:
1. Vehicle Suspension Profiler & Dynamic Speed Calibration (src/suspension_calibrator.py)
2. PWD Contractor Warranty, DLP & Penalty Audit Engine (src/pwd_audit_report.py)
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.suspension_calibrator import VehicleSuspensionProfiler, VEHICLE_PROFILES
from src.pwd_audit_report import HighwayContractorAuditor


def test_suspension_calibrator():
    print("\n" + "=" * 70)
    print("  [TEST 1] VERIFYING VEHICLE SUSPENSION PROFILER & ADAPTIVE SAMPLER")
    print("=" * 70)

    profiler = VehicleSuspensionProfiler("HATCHBACK_SEDAN")
    spec = profiler.get_current_spec()
    print(f"[OK] Initialized profile: {spec['display_name']}")
    assert spec["profile_key"] == "HATCHBACK_SEDAN"
    assert spec["shock_threshold_g"] == 1.30

    # 1. Test Profile Switching and Persistence
    for p_key in VEHICLE_PROFILES.keys():
        profiler.save_profile(p_key)
        reloaded = VehicleSuspensionProfiler()
        assert reloaded.profile_key == p_key, f"Failed to persist {p_key}"
        print(f"[OK] Persisted & Reloaded: {p_key} ({reloaded.spec['icon']} {reloaded.spec['display_name']})")

    # 2. Test Speed-Responsive Adaptive Frame Sampling
    test_speeds = [
        (85.0, 2),   # High Highway speed -> Sample aggressively (N=2)
        (60.0, 3),   # Fast arterial -> N=3
        (35.0, 5),   # City traffic -> N=5
        (15.0, 8),   # Slow congested -> N=8
        (1.0, 20),   # Standstill / Red light -> N=20 (Conserves CPU)
    ]
    for speed, expected_rate in test_speeds:
        actual_rate = profiler.compute_adaptive_sample_rate(speed)
        print(f"[OK] Speed {speed:4.1f} km/h -> Sample Every {actual_rate} Frames (Expected: {expected_rate})")
        assert actual_rate == expected_rate, f"Speed {speed} failed: got {actual_rate}"

    # 3. Test Threat Index Calibration Across Vehicle Architectures
    # Scenario: 4.0cm pothole, moderate shock (1.4g)
    # A) Sedan
    profiler.save_profile("HATCHBACK_SEDAN")
    sedan_eval = profiler.calibrate_threat_index(raw_threat_index=60.0, depth_cm=4.0, shock_gz=1.4)
    print(f"[OK] Sedan 4.0cm Pothole Threat: {sedan_eval['threat_index']}/100 ({sedan_eval['risk_tier']})")

    # B) Two-Wheeler (Should elevate to CRITICAL hazard!)
    profiler.save_profile("TWO_WHEELER_SCOOTER")
    tw_eval = profiler.calibrate_threat_index(raw_threat_index=60.0, depth_cm=4.0, shock_gz=1.4)
    print(f"[OK] 2-Wheeler 4.0cm Pothole Threat: {tw_eval['threat_index']}/100 ({tw_eval['risk_tier']})")
    assert tw_eval["threat_index"] >= 85.0, "Two-wheeler must get elevated critical hazard score!"

    # Reset to Sedan for normal tests
    profiler.save_profile("HATCHBACK_SEDAN")
    print("  --> ALL VEHICLE SUSPENSION PROFILER TESTS PASSED!")


def test_pwd_contractor_audit():
    print("\n" + "=" * 70)
    print("  [TEST 2] VERIFYING PWD CONTRACTOR WARRANTY & PENALTY AUDIT ENGINE")
    print("=" * 70)

    auditor = HighwayContractorAuditor()
    projects = auditor.get_all_projects()
    print(f"[OK] Found {len(projects)} registered highway contractor projects.")
    assert len(projects) >= 3

    # Audit Project 1: NHAI-EPC-NH48-PKG4
    proj_id = "NHAI-EPC-NH48-PKG4"
    simulated_defects = [
        {"severity_category": "High", "max_depth_cm": 6.8, "cavity_volume_liters": 4.5, "asphalt_repair_mass_kg": 10.8},
        {"severity_category": "High", "max_depth_cm": 5.5, "cavity_volume_liters": 3.8, "asphalt_repair_mass_kg": 9.1},
        {"severity_category": "Medium", "max_depth_cm": 4.2, "cavity_volume_liters": 2.5, "asphalt_repair_mass_kg": 6.0},
        {"severity_category": "Medium", "max_depth_cm": 3.9, "cavity_volume_liters": 2.0, "asphalt_repair_mass_kg": 4.8},
        {"severity_category": "Medium", "max_depth_cm": 4.5, "cavity_volume_liters": 3.1, "asphalt_repair_mass_kg": 7.4},
        {"severity_category": "Low", "max_depth_cm": 2.1, "cavity_volume_liters": 1.0, "asphalt_repair_mass_kg": 2.4},
    ]

    audit_res = auditor.audit_project_stretch(proj_id, defects_override=simulated_defects)

    print(f"[OK] Project: {audit_res['project_id']} | Stretch: {audit_res['highway_name']}")
    print(f"     Contractor: {audit_res['contractor_name']}")
    print(f"     Warranty Status: {'ACTIVE (under DLP)' if audit_res['is_under_warranty'] else 'EXPIRED'}")
    print(f"     Pavement Condition Index (PCI): {audit_res['current_pci']}/100 (Drop: -{audit_res['pci_drop']})")
    print(f"     Total Defects: {audit_res['total_defects_found']} (Critical: {audit_res['critical_defects']})")
    print(f"     Asphalt Required: {audit_res['total_asphalt_needed_kg']} kg ({audit_res['total_void_liters']} L)")
    print(f"     Statutory Compliance: {audit_res['compliance_status']}")
    print(f"     Liquidated Penalty: ₹ {audit_res['liquidated_penalty_inr']:,} INR")

    assert audit_res["is_under_warranty"] is True
    assert audit_res["total_defects_found"] == 6
    assert audit_res["critical_defects"] == 2
    assert audit_res["liquidated_penalty_inr"] > 0
    assert audit_res["compliance_status"] in ["LIQUIDATED_DAMAGES_LEVIED", "BLACKLIST_RECOMMENDED"]

    # Generate Official Printable PDF Report
    pdf_path = auditor.generate_audit_pdf(audit_res)
    print(f"[OK] Generated Official PWD Audit PDF: {pdf_path}")
    assert os.path.exists(pdf_path), "PDF file was not created!"
    assert os.path.getsize(pdf_path) > 2000, "PDF file is suspiciously small!"

    print("  --> ALL PWD CONTRACTOR AUDIT TESTS PASSED!")


def main():
    test_suspension_calibrator()
    test_pwd_contractor_audit()
    print("\n" + "=" * 70)
    print("  *** 100% SUCCESS: SUSPENSION & PWD AUDIT ENGINES FULLY VERIFIED ***")
    print("=" * 70)


if __name__ == "__main__":
    main()
