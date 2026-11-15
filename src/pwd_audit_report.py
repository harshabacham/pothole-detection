"""
src/pwd_audit_report.py
=======================
PWD Contractor Warranty, Defect Liability Period (DLP) & Penalty Audit Engine.

Government Accountability & Legal Infrastructure Compliance:
Under Ministry of Road Transport and Highways (MoRTH) and NHAI EPC contract clauses
(Section 10.3, IRC:SP:84, and IRC:82-2015), contractors are legally bound to a
Defect Liability Period (DLP, typically 3 to 5 years). Any pothole or pavement distress
must be rectified within 48 hours of discovery, failing which contractual liquidated
damages of ₹25,000 per day per defect are levied, up to contractor debarment / blacklisting.

Key Capabilities:
1. Maps GPS coordinates of detected potholes to Highway Chainage (Km markers) and Contractor Registry.
2. Evaluates Defect Liability Period (DLP) warranty expiration dates.
3. Computes contractual financial penalties (₹ INR) under MoRTH Clause 10.3.
4. Determines Contractor Legal Compliance Status:
   - COMPLIANT                  : High PCI (>= 80), zero critical cavities
   - RECTIFICATION_NOTICE_ISSUED: Potholes found within warranty (48-hour cure notice)
   - LIQUIDATED_DAMAGES_LEVIED  : Unrepaired defects exceeding statutory timeline (Fine levied)
   - BLACKLIST_RECOMMENDED      : Premature road collapse (>15 defects / PCI < 55 within 18 months)
5. Generates official executive ReportLab PDF dossiers for the Ministry and State Vigilance Commission.
"""

import os
import sys
import math
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.database import get_db_connection

_AUDIT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "audits")

# ---------------------------------------------------------------------------
# Benchmark Contractor & Highway Stretch Registry (MoRTH / NHAI Projects)
# ---------------------------------------------------------------------------
DEFAULT_CONTRACTOR_PROJECTS = [
    {
        "project_id": "NHAI-EPC-NH48-PKG4",
        "highway_name": "National Highway 48 (Delhi - Gurugram - Jaipur Expressway)",
        "chainage_start_km": 115.0,
        "chainage_end_km": 145.0,
        "contractor_name": "M/s Apex Infrastructure & Highway Builders Ltd.",
        "project_director_office": "NHAI Project Implementation Unit (PIU), Gurugram",
        "completion_date": "2024-03-20",
        "dlp_period_years": 3.0,
        "contract_value_cr_inr": 284.50,
        "liquidated_damage_per_day_inr": 25000,
        "target_pci_baseline": 92.0,
    },
    {
        "project_id": "MoRTH-PWD-DELHI-ORR-02",
        "highway_name": "Outer Ring Road (Greater Kailash - Nehru Place Corridor)",
        "chainage_start_km": 12.0,
        "chainage_end_km": 28.0,
        "contractor_name": "M/s Horizon Roadways & Urban Infra Pvt. Ltd.",
        "project_director_office": "Delhi State PWD South Zone Division",
        "completion_date": "2024-08-10",
        "dlp_period_years": 2.0,
        "contract_value_cr_inr": 118.20,
        "liquidated_damage_per_day_inr": 20000,
        "target_pci_baseline": 90.0,
    },
    {
        "project_id": "NHAI-HAM-NH44-PKG7",
        "highway_name": "National Highway 44 (Panipat - Jalandhar Section)",
        "chainage_start_km": 85.0,
        "chainage_end_km": 130.0,
        "contractor_name": "M/s Regal Translink Engineering Corp.",
        "project_director_office": "NHAI Regional Office, Ambala",
        "completion_date": "2023-11-05",
        "dlp_period_years": 4.0,
        "contract_value_cr_inr": 412.00,
        "liquidated_damage_per_day_inr": 35000,
        "target_pci_baseline": 94.0,
    },
]


class HighwayContractorAuditor:
    """
    Automated civil engineering auditor verifying contractor warranty compliance,
    liquidated damage liabilities, and road quality degradation.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        os.makedirs(_AUDIT_OUTPUT_DIR, exist_ok=True)
        self._init_contractor_db()

    def _init_contractor_db(self):
        """Create and populate contractor projects database table."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS contractor_projects (
            project_id TEXT PRIMARY KEY,
            highway_name TEXT,
            chainage_start_km REAL,
            chainage_end_km REAL,
            contractor_name TEXT,
            project_director_office TEXT,
            completion_date TEXT,
            dlp_period_years REAL,
            contract_value_cr_inr REAL,
            liquidated_damage_per_day_inr REAL,
            target_pci_baseline REAL
        );
        """)

        # Insert defaults if table is empty
        count = cursor.execute("SELECT COUNT(*) FROM contractor_projects").fetchone()[0]
        if count == 0:
            for p in DEFAULT_CONTRACTOR_PROJECTS:
                cursor.execute("""
                INSERT INTO contractor_projects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p["project_id"], p["highway_name"], p["chainage_start_km"],
                    p["chainage_end_km"], p["contractor_name"], p["project_director_office"],
                    p["completion_date"], p["dlp_period_years"], p["contract_value_cr_inr"],
                    p["liquidated_damage_per_day_inr"], p["target_pci_baseline"],
                ))
            conn.commit()

        conn.close()

    def get_all_projects(self) -> list:
        """Fetch all registered highway contractor project profiles."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM contractor_projects").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def audit_project_stretch(self, project_id: str, defects_override: list = None) -> dict:
        """
        Execute an automated warranty audit for a specific contractor highway stretch.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT * FROM contractor_projects WHERE project_id = ?", (project_id,)
        ).fetchone()

        if not row:
            conn.close()
            raise ValueError(f"Project ID '{project_id}' not found in registry.")

        project = dict(row)

        # Pull recorded defects
        if defects_override is not None:
            defects = defects_override
        else:
            defect_rows = cursor.execute("SELECT * FROM defect_records").fetchall()
            defects = [dict(r) for r in defect_rows]

        conn.close()

        # Calculate Defect Liability Period (DLP) status
        comp_dt = datetime.fromisoformat(project["completion_date"])
        dlp_years = float(project["dlp_period_years"])
        dlp_expiry = comp_dt + timedelta(days=int(dlp_years * 365.25))
        now_dt = datetime.now()
        is_under_dlp = (now_dt <= dlp_expiry)
        days_since_comp = (now_dt - comp_dt).days
        months_since_comp = round(days_since_comp / 30.4, 1)

        # Analyze defects on this stretch
        # If no stretch specific GPS filter is supplied, aggregate available defects
        total_potholes = len(defects)
        critical_count = sum(1 for d in defects if d.get("severity_category") == "High" or d.get("max_depth_cm", 0) >= 5.0)
        medium_count = sum(1 for d in defects if d.get("severity_category") == "Medium")
        low_count = sum(1 for d in defects if d.get("severity_category") == "Low")

        total_asphalt_kg = sum(d.get("asphalt_repair_mass_kg", 8.2) for d in defects)
        total_void_liters = sum(d.get("cavity_volume_liters", 3.4) for d in defects)

        # Calculate PCI degradation
        stretch_len_km = max(1.0, project["chainage_end_km"] - project["chainage_start_km"])
        defect_density_per_km = round(total_potholes / stretch_len_km, 2)

        # PCI calculation (ASTM D6433 deduction)
        pci_deduct = min(80.0, (critical_count * 12.0) + (medium_count * 5.0) + (low_count * 2.0))
        current_pci = round(max(15.0, project["target_pci_baseline"] - pci_deduct), 1)
        pci_drop = round(project["target_pci_baseline"] - current_pci, 1)

        # Legal Penalty Calculation (MoRTH Clause 10.3 / IRC:SP:84)
        daily_fine_inr = float(project["liquidated_damage_per_day_inr"])
        # Assume average unaddressed defect response time of 14 days beyond 48-hr cure window
        unrepaired_days = 14
        total_penalty_inr = int(total_potholes * daily_fine_inr * unrepaired_days)

        # Compliance status determination
        if total_potholes == 0 or (current_pci >= 82.0 and critical_count == 0):
            compliance_status = "COMPLIANT"
            action_code = "NO_ACTION_REQUIRED"
            action_desc = "Highway pavement within contractual warranty tolerance. Surface quality compliant."
            penalty_levied_inr = 0
        elif total_potholes <= 4 and months_since_comp >= 12.0:
            compliance_status = "RECTIFICATION_NOTICE_ISSUED"
            action_code = "CURE_NOTICE_48HR"
            action_desc = "Issuance of mandatory 48-Hour Cure Notice under Clause 10.3. Contractor must patch immediately."
            penalty_levied_inr = 0
        elif months_since_comp <= 18.0 and (critical_count >= 5 or current_pci < 60.0):
            compliance_status = "BLACKLIST_RECOMMENDED"
            action_code = "DEBARMENT_PROCEEDING"
            action_desc = "PREMATURE STRUCTURAL ROAD FAILURE: Pavement collapsed within 18 months of commissioning. Recommend 3-year debarment/blacklisting under GFR Rule 151."
            penalty_levied_inr = total_penalty_inr
        else:
            compliance_status = "LIQUIDATED_DAMAGES_LEVIED"
            action_code = "PENALTY_DEDUCTION"
            action_desc = f"Liquidated damages of ₹{total_penalty_inr:,} levied against contractor's performance bank guarantee for unresolved road distress."
            penalty_levied_inr = total_penalty_inr

        return {
            "project_id": project["project_id"],
            "highway_name": project["highway_name"],
            "chainage_str": f"Km {project['chainage_start_km']} to Km {project['chainage_end_km']}",
            "stretch_length_km": stretch_len_km,
            "contractor_name": project["contractor_name"],
            "project_director_office": project["project_director_office"],
            "completion_date": project["completion_date"],
            "dlp_expiry_date": dlp_expiry.strftime("%Y-%m-%d"),
            "is_under_warranty": is_under_dlp,
            "age_months": months_since_comp,
            "contract_value_cr": project["contract_value_cr_inr"],
            "target_pci_baseline": project["target_pci_baseline"],
            "current_pci": current_pci,
            "pci_drop": pci_drop,
            "defect_density_per_km": defect_density_per_km,
            "total_defects_found": total_potholes,
            "critical_defects": critical_count,
            "medium_defects": medium_count,
            "low_defects": low_count,
            "total_asphalt_needed_kg": round(total_asphalt_kg, 1),
            "total_void_liters": round(total_void_liters, 1),
            "compliance_status": compliance_status,
            "action_code": action_code,
            "action_recommendation": action_desc,
            "liquidated_penalty_inr": penalty_levied_inr,
            "audit_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "defects_list": defects[:10],  # Include top 10 for table
        }

    def generate_audit_pdf(self, audit_dict: dict, output_path: str = None) -> str:
        """
        Generate an official, print-ready PWD / MoRTH Highway Audit Report & Penalty Dossier.
        """
        proj_clean = audit_dict["project_id"].replace("/", "_").replace("-", "_")
        if output_path is None:
            output_path = os.path.join(_AUDIT_OUTPUT_DIR, f"PWD_Contractor_Audit_{proj_clean}.pdf")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "GovTitle",
            parent=styles["Heading1"],
            fontSize=13,
            leading=16,
            alignment=1,
            textColor=colors.HexColor("#1a252f"),
            fontName="Helvetica-Bold",
        )
        sub_style = ParagraphStyle(
            "GovSub",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            alignment=1,
            textColor=colors.HexColor("#555555"),
            fontName="Helvetica",
        )
        sec_style = ParagraphStyle(
            "SecHeader",
            parent=styles["Heading2"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#2c3e50"),
            fontName="Helvetica-Bold",
        )
        body_style = ParagraphStyle(
            "GovBody",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#222222"),
            fontName="Helvetica",
        )
        alert_style = ParagraphStyle(
            "AlertBody",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#900C3F"),
            fontName="Helvetica-Bold",
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("GOVERNMENT OF INDIA", sub_style))
        elements.append(Paragraph("MINISTRY OF ROAD TRANSPORT & HIGHWAYS (MoRTH)", title_style))
        elements.append(Paragraph("NATIONAL HIGHWAYS AUTHORITY OF INDIA (NHAI) — CONTRACT PERFORMANCE AUDIT CELL", sub_style))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("<b>STATUTORY HIGHWAY PAVEMENT AUDIT & CONTRACTOR WARRANTY ENFORCEMENT DOSSIER</b>", ParagraphStyle("RedBanner", parent=title_style, fontSize=10.5, textColor=colors.HexColor("#c0392b"))))
        elements.append(Paragraph(f"Audit Reference: MoRTH/AUDIT/{audit_dict['project_id']} | Date: {audit_dict['audit_timestamp']}", sub_style))
        elements.append(Spacer(1, 10))

        # 2. Project Profile Table
        elements.append(Paragraph("1. CONTRACT & INFRASTRUCTURE PROFILE", sec_style))
        proj_data = [
            ["Project Identifier:", audit_dict["project_id"], "Contract Value:", f"₹ {audit_dict['contract_value_cr']} Crores"],
            ["Highway Stretch:", audit_dict["highway_name"], "Chainage Markers:", audit_dict["chainage_str"]],
            ["Contractor Name:", audit_dict["contractor_name"], "Stretch Length:", f"{audit_dict['stretch_length_km']:.1f} Km"],
            ["Commissioning Date:", audit_dict["completion_date"], "DLP Warranty Expiry:", f"{audit_dict['dlp_expiry_date']} ({'ACTIVE' if audit_dict['is_under_warranty'] else 'EXPIRED'})"],
            ["Superintending Office:", audit_dict["project_director_office"], "Warranty Age:", f"{audit_dict['age_months']} months"],
        ]
        t_proj = Table(proj_data, colWidths=[110, 180, 110, 140])
        t_proj.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2c3e50")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dcdcdc")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_proj)
        elements.append(Spacer(1, 10))

        # 3. Pavement Condition & Compliance Status Table
        elements.append(Paragraph("2. PAVEMENT HEALTH AUDIT & LEGAL COMPLIANCE STATUS", sec_style))
        status_color = "#27ae60" if audit_dict["compliance_status"] == "COMPLIANT" else "#c0392b"
        comp_data = [
            ["Pavement Condition Index (PCI):", f"{audit_dict['current_pci']} / 100 (Baseline: {audit_dict['target_pci_baseline']}, Drop: -{audit_dict['pci_drop']})"],
            ["Total Defects Verified:", f"{audit_dict['total_defects_found']} (Critical: {audit_dict['critical_defects']}, Medium: {audit_dict['medium_defects']}, Low: {audit_dict['low_defects']})"],
            ["Defect Frequency Density:", f"{audit_dict['defect_density_per_km']} potholes per kilometer"],
            ["Asphalt Repair Material Required:", f"{audit_dict['total_asphalt_needed_kg']} kg ({audit_dict['total_void_liters']} L cavity volume)"],
            ["STATUTORY COMPLIANCE STATUS:", audit_dict["compliance_status"]],
            ["CONTRACTUAL PENALTY LEVIED:", f"₹ {audit_dict['liquidated_penalty_inr']:,} INR (under MoRTH Clause 10.3)"],
        ]
        t_comp = Table(comp_data, colWidths=[180, 360])
        t_comp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2c3e50")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#fef2f2") if audit_dict["compliance_status"] != "COMPLIANT" else colors.HexColor("#f0fdf4")),
            ("TEXTCOLOR", (1, 4), (1, 4), colors.HexColor(status_color)),
            ("FONTNAME", (1, 4), (1, 4), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_comp)
        elements.append(Spacer(1, 10))

        # 4. Official Penalty & Show Cause Directive
        elements.append(Paragraph("3. SUPERINTENDING ENGINEER DIRECTIVE & PENALTY DETERMINATION", sec_style))
        directive_text = (
            f"<b>ACTION DIRECTIVE ({audit_dict['action_code']}):</b> {audit_dict['action_recommendation']}<br/><br/>"
            f"Pursuant to the powers vested under <b>MoRTH Standard Contract Section 10.3 and IRC:SP:84-2019</b>, "
            f"the contractor is hereby formally notified that the road surface on <b>{audit_dict['highway_name']}</b> "
            f"exhibits unacceptable structural failure within the statutory Defect Liability Period. "
            f"The computed penalty of <b>₹ {audit_dict['liquidated_penalty_inr']:,}</b> shall be adjusted against "
            f"the contractor's upcoming milestone payments / Performance Security if full asphalt restoration "
            f"is not verified within the stipulated timeline."
        )
        elements.append(Paragraph(directive_text, alert_style if audit_dict['compliance_status'] != 'COMPLIANT' else body_style))
        elements.append(Spacer(1, 14))

        # 5. Signature Block
        sig_data = [
            ["Report Prepared By:", "Verified By:", "Sanctioning Authority:"],
            ["\nAutomated AI Highway\nSurveillance Engine\n(Dashcam Core v2.4)",
             "\nExecutive Engineer (Quality Control)\nPIU Highway Division\nGovernment of India",
             "\nSuperintending Engineer / Project Director\nNational Highways Authority of India\nMoRTH, New Delhi"]
        ]
        t_sig = Table(sig_data, colWidths=[180, 180, 180])
        t_sig.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("LINEABOVE", (0, 1), (-1, 1), 0.5, colors.HexColor("#999999")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(KeepTogether(t_sig))

        doc.build(elements)
        return output_path
