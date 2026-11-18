"""
app.py
======
Autonomous Pavement Telemetry & Road Surface Distress Survey System.
Real-time automotive dashcam surveillance, unsupervised anomaly isolation,
3D metric cavity profiling, and statutory Government of India grievance dispatch.
Compliant with IRC:82-2015, MoRTH Section 10, and ASTM D6433 standards.
"""

import os
import sys
import io
import json as _json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.dashboard_helpers import (
    bgr_to_rgb,
    bgr_to_png_bytes,
    dataframe_to_csv_bytes,
    check_pipeline_readiness,
)
from src.inference_pipeline import load_all_artifacts, run_inference_on_image
from src.database import (
    init_database,
    get_all_defects,
    insert_defect,
    update_grievance_status,
    disapprove_defect,
    get_pending_review_defects,
)
from src.map_visualization import build_geographic_map
from src.video_processor import process_road_video
from src.gis_export import (
    geojson_to_bytes,
    export_defects_to_geojson,
    kml_to_bytes,
    export_smart_city_wms_feed,
)
from src.gov_portal_api import (
    reverse_geocode_location,
    build_grievance_payload,
    generate_complaint_pdf,
    dispatch_grievance_to_portal,
)
from src.dashcam_agent import VehicleDashcamAutoReporter
from src.camera_setup import (
    get_available_cameras,
    get_saved_camera_config,
    save_camera_config,
    get_active_dashcam_source,
)
from src.distress_taxonomy import PavementDistressClassifier
from src.fleet_analytics import PotholeDegradationTracker
from src.crowd_dedup import CrowdsourcedSpatialDeduplicator
from src.suspension_calibrator import VehicleSuspensionProfiler, VEHICLE_PROFILES
from src.pwd_audit_report import HighwayContractorAuditor
from src.offline_sync import OfflineSyncManager
from src.citizen_notifier import CitizenNotifier
from src.municipal_workflow import MunicipalWorkOrderManager
from streamlit_folium import st_folium


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PotholeAI — Road Surface Intelligence Platform",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design System — Clean, Modern, Professional
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ─── Global Reset ─────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #111827;
        background: #f9fafb;
    }

    /* hide default header padding */
    .block-container { padding-top: 1rem !important; }

    /* ─── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] hr { border-color: #1e293b !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #334155 !important;
        border-color: #475569 !important;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px 0;
        margin-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }
    .sidebar-logo-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }
    .sidebar-logo-text { font-weight: 800; font-size: 1.0rem; color: #f1f5f9 !important; }
    .sidebar-logo-sub  { font-size: 0.65rem; color: #64748b !important; letter-spacing: 0.5px; }
    .sidebar-section-header {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #475569 !important;
        padding: 10px 0 4px 0;
        margin-top: 4px;
    }
    .sidebar-metric {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .sidebar-metric-label { font-size: 0.65rem; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.5px; }
    .sidebar-metric-value { font-size: 1.1rem; font-weight: 700; color: #f1f5f9 !important; font-family: 'JetBrains Mono', monospace; }

    /* ─── Page Header ──────────────────────────────────────────────────── */
    .page-header {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .page-header-left { display: flex; align-items: center; gap: 16px; }
    .header-icon {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        flex-shrink: 0;
    }
    .header-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 2px;
        font-weight: 500;
    }
    .header-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 1px solid;
    }
    .badge-blue   { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
    .badge-green  { background: #f0fdf4; color: #15803d; border-color: #bbf7d0; }
    .badge-amber  { background: #fffbeb; color: #b45309; border-color: #fde68a; }
    .badge-slate  { background: #f8fafc; color: #475569; border-color: #e2e8f0; }

    /* ─── KPI Cards ────────────────────────────────────────────────────── */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }
    @media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 500px) { .kpi-row { grid-template-columns: 1fr; } }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }
    .kpi-card-accent {
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .acc-blue   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .acc-red    { background: linear-gradient(90deg, #ef4444, #f87171); }
    .acc-sky    { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
    .acc-green  { background: linear-gradient(90deg, #22c55e, #4ade80); }

    .kpi-icon {
        font-size: 1.3rem;
        margin-bottom: 6px;
        display: block;
    }
    .kpi-label {
        font-size: 0.67rem;
        font-weight: 700;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }
    .kpi-number {
        font-size: 1.9rem;
        font-weight: 800;
        color: #111827;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin: 2px 0;
    }
    .kpi-sub {
        font-size: 0.7rem;
        color: #9ca3af;
        font-weight: 500;
        margin-top: 2px;
    }

    /* ─── Tab Styling ──────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f3f4f6;
        border-radius: 12px;
        padding: 4px;
        border: none !important;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        color: #6b7280 !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #111827 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* ─── Section Headers ──────────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
    }
    .section-icon {
        width: 32px; height: 32px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .si-blue   { background: #eff6ff; }
    .si-green  { background: #f0fdf4; }
    .si-amber  { background: #fffbeb; }
    .si-red    { background: #fef2f2; }
    .si-purple { background: #faf5ff; }
    .section-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .section-desc {
        font-size: 0.74rem;
        color: #9ca3af;
        margin-top: 1px;
    }

    /* ─── Info/Empty State ─────────────────────────────────────────────── */
    .empty-state {
        background: #f9fafb;
        border: 2px dashed #e5e7eb;
        border-radius: 14px;
        padding: 40px 24px;
        text-align: center;
        color: #9ca3af;
    }
    .empty-state-icon { font-size: 2.5rem; margin-bottom: 10px; }
    .empty-state-title { font-size: 1.0rem; font-weight: 600; color: #374151; margin-bottom: 6px; }
    .empty-state-desc { font-size: 0.82rem; }

    /* ─── Incident Cards ───────────────────────────────────────────────── */
    .incident-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-left: 4px solid #e5e7eb;
        transition: box-shadow 0.15s;
    }
    .incident-card.critical { border-left-color: #ef4444; }
    .incident-card.elevated { border-left-color: #f59e0b; }
    .incident-card.monitored { border-left-color: #22c55e; }

    .incident-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
    }
    .severity-chip {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 9px;
        border-radius: 20px;
    }
    .chip-high   { background: #fef2f2; color: #b91c1c; }
    .chip-medium { background: #fffbeb; color: #92400e; }
    .chip-low    { background: #f0fdf4; color: #166534; }

    .defect-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #9ca3af;
        font-weight: 500;
    }

    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        background: #f9fafb;
        border: 1px solid #f3f4f6;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 10px 0;
    }
    .metric-cell { display: flex; flex-direction: column; }
    .metric-label {
        font-size: 0.62rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 0.84rem;
        font-weight: 700;
        color: #111827;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 1px;
    }

    /* ─── Ticket Card (live dashcam events) ────────────────────────────── */
    .ticket {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.82rem;
    }
    .ticket.t-high   { border-left-color: #ef4444; }
    .ticket.t-medium { border-left-color: #f59e0b; }
    .ticket.t-low    { border-left-color: #22c55e; }
    .ticket-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .ticket-road { font-weight: 700; color: #111827; }
    .ticket-ref  { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #9ca3af; }
    .ticket-stats { display: flex; gap: 10px; }
    .ticket-stat { color: #6b7280; }
    .ticket-stat span { font-weight: 700; color: #374151; }

    /* ─── Analysis Result Grid ─────────────────────────────────────────── */
    .result-metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
        margin: 14px 0;
    }
    @media (max-width: 768px) { .result-metric-grid { grid-template-columns: repeat(3, 1fr); } }
    .result-metric {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .result-metric-num {
        font-size: 1.5rem;
        font-weight: 800;
        color: #111827;
        font-family: 'JetBrains Mono', monospace;
    }
    .result-metric-lbl {
        font-size: 0.65rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* ─── Buttons ──────────────────────────────────────────────────────── */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 9px 20px !important;
        box-shadow: 0 2px 6px rgba(59,130,246,0.3) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        border: 1px solid #e5e7eb !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover { background: #f9fafb !important; }

    /* ─── Forms & Inputs ───────────────────────────────────────────────── */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: #e5e7eb !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }
    div[data-testid="stForm"] {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
    }

    /* ─── Dataframe ────────────────────────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ─── Expander ─────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* ─── Divider ──────────────────────────────────────────────────────── */
    .divider {
        height: 1px;
        background: #f3f4f6;
        margin: 16px 0;
    }

    /* ─── Progress Bar ─────────────────────────────────────────────────── */
    .stProgress > div > div { background: #3b82f6 !important; border-radius: 4px !important; }

    /* ─── Alert/Info boxes ─────────────────────────────────────────────── */
    .stAlert { border-radius: 10px !important; }

    /* ─── Approve / Disapprove buttons ────────────────────────────────── */
    .approve-btn > button {
        background: #22c55e !important;
        color: #fff !important;
        border: none !important;
    }
    .approve-btn > button:hover { background: #16a34a !important; }
    .disapprove-btn > button {
        background: #ef4444 !important;
        color: #fff !important;
        border: none !important;
    }
    .disapprove-btn > button:hover { background: #dc2626 !important; }

    /* ─── Status pill ──────────────────────────────────────────────────── */
    .status-dispatched { color: #15803d; font-weight: 700; font-size: 0.8rem; }
    .status-pending    { color: #b45309; font-weight: 700; font-size: 0.8rem; }
    .status-rejected   { color: #b91c1c; font-weight: 700; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cache & Init
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising ML pipeline…")
def get_cached_models():
    return load_all_artifacts()


def _sev_chip(cat: str) -> str:
    cat_l = str(cat).lower()
    cls = "chip-high" if cat_l == "high" else "chip-medium" if cat_l == "medium" else "chip-low"
    return f'<span class="severity-chip {cls}">{cat.upper()}</span>'


def _card_cls(cat: str) -> str:
    cat_l = str(cat).lower()
    return "critical" if cat_l == "high" else "elevated" if cat_l == "medium" else "monitored"


def render_citizen_queue_view(defects_df: pd.DataFrame, key_prefix: str = "cq"):
    """Render the Citizen Verification and Approval Queue."""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon si-green">✅</div>
        <div>
            <div class="section-title">Citizen Verification & Approval Queue</div>
            <div class="section-desc">Review defects recorded by camera surveys — approve to file official petitions with government authorities, or reject false alarms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if defects_df.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-title">No defects recorded yet</div>
            <div class="empty-state-desc">Use the Live Camera Survey or Inspection Studio to record road defects first.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    status_series    = defects_df["grievance_status"].fillna("Unreported")
    pending_mask     = status_series.isin(["Pending Review", "Unreported", "PORTAL_SYNC_QUEUED", "PENDING_USER_REVIEW"])
    dispatched_mask  = status_series.str.startswith("Dispatched", na=False)
    disapproved_mask = status_series.str.startswith("Disapproved", na=False)

    pending_df     = defects_df[pending_mask]
    dispatched_df  = defects_df[dispatched_mask]
    disapproved_df = defects_df[disapproved_mask]

    # Mini KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Recorded", len(defects_df))
    kc2.metric("⏳ Awaiting Review", len(pending_df))
    kc3.metric("✅ Dispatched", len(dispatched_df))
    kc4.metric("❌ Rejected", len(disapproved_df))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    filter_opts = [
        f"Pending ({len(pending_df)})",
        f"Dispatched ({len(dispatched_df)})",
        f"Rejected ({len(disapproved_df)})",
        f"All ({len(defects_df)})",
    ]
    filter_view = st.radio("Filter Queue", filter_opts, horizontal=True, key=f"{key_prefix}_filter_opts")

    if "Pending" in filter_view:
        view_df = pending_df
    elif "Dispatched" in filter_view:
        view_df = dispatched_df
    elif "Rejected" in filter_view:
        view_df = disapproved_df
    else:
        view_df = defects_df

    if view_df.empty:
        st.caption("No records in this category.")
        return

    for _, row in view_df.iterrows():
        det_id   = row["detection_id"]
        status   = row.get("grievance_status") or "Pending Review"
        cat      = row.get("severity_category", "Medium")
        cat_l    = str(cat).lower()
        card_cls = _card_cls(cat)
        chip     = _sev_chip(cat)

        has_bev  = pd.notnull(row.get("bev_area_sqcm"))
        d_cm     = float(row["bev_area_sqcm"] / 100.0) if has_bev else None
        vol_l    = round(d_cm * 0.65, 1) if d_cm is not None else None

        has_gps  = "latitude" in row.index and pd.notnull(row.get("latitude")) and pd.notnull(row.get("longitude"))
        lat_val  = float(row["latitude"])  if has_gps else None
        lon_val  = float(row["longitude"]) if has_gps else None
        gps_str  = f"{lat_val:.5f}, {lon_val:.5f}" if has_gps else "Ungeotagged"
        d_str    = f"{d_cm:.1f} cm" if d_cm is not None else "N/A"
        v_str    = f"{vol_l:.1f} L"  if vol_l is not None else "N/A"
        ts_str   = str(row.get("timestamp", "N/A"))[:16]
        img_path = row.get("image_path")

        col_img, col_info, col_act = st.columns([1, 2, 1.4])

        with col_img:
            if img_path and os.path.exists(str(img_path)):
                st.image(str(img_path), use_container_width=True)
            else:
                st.markdown(f"""
                <div style="background:#f9fafb; border:1px dashed #e5e7eb; border-radius:8px; 
                     padding:28px 10px; text-align:center; color:#9ca3af; font-size:0.72rem;">
                    📷 Photo Evidence<br/><code style="font-size:0.65rem">{det_id[:20]}</code>
                </div>
                """, unsafe_allow_html=True)

        with col_info:
            st.markdown(f"""
            <div class="incident-card {card_cls}">
                <div class="incident-header">
                    {chip}
                    <span class="defect-id">{det_id[:30]}</span>
                </div>
                <div class="metrics-row">
                    <div class="metric-cell">
                        <span class="metric-label">GPS</span>
                        <span class="metric-value" style="font-size:0.72rem;">{gps_str}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">Depth</span>
                        <span class="metric-value">{d_str}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">Volume</span>
                        <span class="metric-value">{v_str}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-label">Recorded</span>
                        <span class="metric-value" style="font-size:0.72rem;">{ts_str}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_act:
            if status in ["Pending Review", "Unreported", "PORTAL_SYNC_QUEUED", "PENDING_USER_REVIEW"]:
                target_p = st.selectbox(
                    "Authority",
                    list(config.GOV_PORTALS.keys()),
                    key=f"{key_prefix}_sel_{det_id}",
                    format_func=lambda k: f"{k}",
                    label_visibility="collapsed",
                )
                col_ap, col_dis = st.columns(2)
                with col_ap:
                    if st.button("✅ Approve", key=f"{key_prefix}_ap_{det_id}", type="primary", use_container_width=True):
                        defect_dict = {
                            "detection_id": det_id,
                            "track_id": row.get("track_id", "TRK_0"),
                            "latitude": lat_val,
                            "longitude": lon_val,
                            "severity_category": cat,
                            "severity_score": float(row.get("severity_score", 0.5)),
                            "anomaly_score": float(row.get("anomaly_score", -0.2)),
                            "pixel_area": int(row.get("pixel_area", 4000)),
                            "image_path": img_path,
                            "source_media": row.get("source_media", "camera_feed"),
                            "timestamp": row.get("timestamp"),
                        }
                        res = dispatch_grievance_to_portal(defect_dict, portal_key=target_p)
                        update_grievance_status(
                            detection_id=det_id,
                            status="Dispatched",
                            ref_no=res["reference_number"],
                            portal=res["target_portal"],
                        )
                        CitizenNotifier().notify(
                            event_dict={**defect_dict, "reference_no": res["reference_number"], "target_portal": res["target_portal"]},
                            notification_type="GRIEVANCE_FILED",
                        )
                        st.success(f"Filed to {res['target_portal']}")
                        st.rerun()
                with col_dis:
                    if st.button("❌ Reject", key=f"{key_prefix}_dis_{det_id}", use_container_width=True):
                        disapprove_defect(det_id, reason="Rejected by citizen — false positive")
                        st.warning("Marked as rejected.")
                        st.rerun()

            elif status.startswith("Dispatched"):
                st.markdown(f'<div class="status-dispatched">✅ Dispatched</div>', unsafe_allow_html=True)
                st.caption(f"Ref: {row.get('grievance_ref_no', 'N/A')}")
                st.caption(f"Portal: {row.get('dispatched_portal', '—')}")
            elif status.startswith("Disapproved"):
                st.markdown('<div class="status-rejected">❌ Rejected</div>', unsafe_allow_html=True)
                if st.button("↩ Revert to Pending", key=f"{key_prefix}_rev_{det_id}", use_container_width=True):
                    update_grievance_status(det_id, status="Pending Review", ref_no="PENDING_REVIEW")
                    st.rerun()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def main():
    init_database()

    # ── Global data ────────────────────────────────────────────────────────
    defects_df      = get_all_defects()
    total_records   = len(defects_df)
    high_count      = int((defects_df["severity_category"] == "High").sum()) if not defects_df.empty and "severity_category" in defects_df.columns else 0
    dedup_mgr       = CrowdsourcedSpatialDeduplicator()
    dedup_summary   = dedup_mgr.get_registry_summary()
    offline_mgr     = OfflineSyncManager()
    q_summary       = offline_mgr.get_queue_summary()
    is_net_online   = offline_mgr.is_online(timeout=0.25)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🛣️</div>
            <div>
                <div class="sidebar-logo-text">PotholeAI</div>
                <div class="sidebar-logo-sub">ROAD SURFACE INTELLIGENCE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-header">ML Engine</div>', unsafe_allow_html=True)
        feature_mode = st.selectbox(
            "Feature Extractor",
            ["Deep Features (ResNet-18)", "Fast OpenCV Texture (35+ FPS)", "Handcrafted (LBP+GLCM+Edge)"],
            label_visibility="collapsed",
            help="Deep = ResNet-18 embeddings. Fast = C++ gradient kernels.",
        )
        internal_feature_mode = "texture_only" if "Fast OpenCV" in feature_mode else feature_mode

        detector_mode = st.selectbox(
            "Anomaly Detector",
            [
                "YOLOv8 Deep Neural (Fast & Accurate)",
                "Standard Isolation Forest",
                "Extended Isolation Forest (EIF)",
                "Ensemble (IForest + EIF)",
            ],
            index=0,
            help="YOLOv8 = Deep neural pothole detector. IForest/EIF = Unsupervised anomaly detection.",
        )

        clustering_mode = st.selectbox(
            "Spatial Clustering",
            ["HDBSCAN (Adaptive Density)", "DBSCAN (Fixed Radius)"],
            index=0,
            help="HDBSCAN adapts automatically across varying defect densities without a fixed epsilon.",
        )

        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            use_pca_recon = st.checkbox("PCA Recon Error", value=True, help="Secondary unsupervised signal: projection MSE from normal manifold.")
        with col_adv2:
            auto_night_vision = st.checkbox("Night-Vision Boost", value=True, help="Adaptive illumination for dark/night road scenes.")

        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Active Stack</div>
            <div class="sidebar-metric-value" style="font-size:0.75rem; line-height:1.2;">
                {feature_mode.split("(")[0].strip()}<br>
                <span style="color:#38bdf8;">{detector_mode.split("(")[0].strip()}</span> · 
                <span style="color:#a78bfa;">{clustering_mode.split("(")[0].strip()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🎓 Unsupervised ML Viva Guide", expanded=False):
            st.markdown(r"""
            **Concepts used in this system:**
            1. **PCA (16-D)**: Unsupervised dimensionality reduction removing covariance. Residual reconstruction error ($MSE$) acts as an outlier metric.
            2. **Isolation Forest**: Isolates anomalies in fewer random tree splits ($\bar{h}(x) \ll c(n)$).
            3. **Extended IForest (EIF)**: Random hyperplane cuts avoiding axis-aligned bias.
            4. **HDBSCAN**: Density-based hierarchical clustering; assigns label `-1` to candidate road distress noise points without manual $\epsilon$.
            5. **Texture Analytics**: LBP uniform patterns & GLCM co-occurrence matrix for surface roughness.
            """)

        st.markdown('<div class="sidebar-section-header">Pipeline Stats</div>', unsafe_allow_html=True)
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="sidebar-metric-label">Records</div>
                <div class="sidebar-metric-value">{total_records}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sb2:
            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="sidebar-metric-label">Critical</div>
                <div class="sidebar-metric-value" style="color:#f87171 !important">{high_count}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-header">Connectivity</div>', unsafe_allow_html=True)
        net_color  = "#4ade80" if is_net_online else "#f87171"
        net_label  = "Online" if is_net_online else "Offline"
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Network</div>
            <div class="sidebar-metric-value" style="color:{net_color} !important; font-size:0.9rem;">{net_label} · {q_summary['pending_count']} queued</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Flush Offline Queue", use_container_width=True):
            flushed = offline_mgr.flush_queue()
            st.success(f"Flushed {flushed} queued complaints.")
            st.rerun()

        st.markdown('<div class="sidebar-section-header">Model Configuration</div>', unsafe_allow_html=True)
        st.caption(f"Patch: {config.DEFAULT_PATCH_SIZE}×{config.DEFAULT_PATCH_SIZE} px")
        st.caption(f"Contamination: {config.IFOREST_CONTAMINATION*100:.0f}%")
        st.caption("Unsupervised: PCA (16-D) + Isolation Forest")

        st.markdown("---")
        # Pipeline readiness check
        is_ready, missing_models = check_pipeline_readiness()
        if not is_ready:
            with st.spinner("Calibrating models on first launch…"):
                from auto_setup import ensure_models_and_data_ready
                ensure_models_and_data_ready(verbose=False)
                st.cache_resource.clear()
            is_ready, _ = check_pipeline_readiness()

        status_icon = "✅" if is_ready else "⚠️"
        st.caption(f"{status_icon} Pipeline {'Ready' if is_ready else 'Needs Setup'}")

    # ── Load ML artifacts ──────────────────────────────────────────────────
    artifacts = get_cached_models()

    # ── Page Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div class="page-header-left">
            <div class="header-icon">🛣️</div>
            <div>
                <h1 class="header-title">PotholeAI — Road Surface Intelligence</h1>
                <div class="header-subtitle">Unsupervised Edge Detection · 3D Cavity Profiling · MoRTH Section 10 Governance · IRC:82-2015 Compliant</div>
            </div>
        </div>
        <div class="header-badges">
            <span class="badge badge-blue">PCA + Isolation Forest</span>
            <span class="badge badge-green">DBSCAN Clustering</span>
            <span class="badge badge-amber">MoRTH Section 10 SLA</span>
            <span class="badge badge-slate">IRC:82 · ASTM D6433</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────────────────
    grievances_filed = dedup_summary.get("total_citizen_reports", 0)
    corroborated     = dedup_summary.get("duplicate_submissions_prevented", 0)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-card-accent acc-blue"></div>
            <span class="kpi-icon">📋</span>
            <div class="kpi-label">Recorded Defects</div>
            <div class="kpi-number">{total_records}</div>
            <div class="kpi-sub">Logged in SQLite registry</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-accent acc-red"></div>
            <span class="kpi-icon">🚨</span>
            <div class="kpi-label">Critical Hazards</div>
            <div class="kpi-number" style="color:#ef4444">{high_count}</div>
            <div class="kpi-sub">High severity · depth ≥ 5 cm</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-accent acc-sky"></div>
            <span class="kpi-icon">🔁</span>
            <div class="kpi-label">Corroborated</div>
            <div class="kpi-number" style="color:#0ea5e9">{corroborated}</div>
            <div class="kpi-sub">Multi-vehicle deduplication</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-accent acc-green"></div>
            <span class="kpi-icon">📨</span>
            <div class="kpi-label">Grievances Filed</div>
            <div class="kpi-number" style="color:#22c55e">{grievances_filed}</div>
            <div class="kpi-sub">Dispatched to CPGRAMS / NHAI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    status_series = defects_df["grievance_status"].fillna("Unreported") if not defects_df.empty and "grievance_status" in defects_df.columns else pd.Series(dtype=object)
    pending_count = int(status_series.isin(["Pending Review", "Unreported", "PORTAL_SYNC_QUEUED", "PENDING_USER_REVIEW"]).sum())

    # ── Main Tabs ──────────────────────────────────────────────────────────
    tab_camera, tab_queue, tab_inspect, tab_gis, tab_gov = st.tabs([
        "📹  Live Camera & Survey",
        f"✅  Citizen Review Queue ({pending_count})",
        "🔬  Inspection Studio",
        "🗺️  GIS Map & Database",
        "🏛️  Municipal Governance & SLA",
    ])

    # =====================================================================
    # TAB 1 — LIVE CAMERA & ROAD SURVEY
    # =====================================================================
    with tab_camera:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon si-blue">📹</div>
            <div>
                <div class="section-title">Live Camera & Road Survey</div>
                <div class="section-desc">Turn on your camera to detect road distress in real-time. Confirmed defects are measured and saved for your approval before filing to government portals.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        cam_mode = st.radio(
            "Survey Detection Source",
            [
                "📱  Live Camera (Phone / Browser WebCam)",
                "🎬  Continuous Video Survey (Benchmark / Dashcam File)",
            ],
            horizontal=True,
        )

        if "Live Camera" in cam_mode:
            st.info("📱 **Live Camera Detection**: Turn on your camera below and point it at the road surface. Tap **Capture** to take photos of potholes or pavement defects. The AI detects the pothole in real-time, estimates severity & 3D depth, and automatically saves it for citizen approval before filing.")

            cam_shot = st.camera_input("Point camera at road defect / surface", label_visibility="visible")

            if cam_shot is not None:
                file_bytes = np.asarray(bytearray(cam_shot.read()), dtype=np.uint8)
                shot_bgr   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if shot_bgr is not None:
                    with st.spinner("Analyzing road surface with unsupervised ML engine…"):
                        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_filename = f"camera_snap_{timestamp_now}.jpg"
                        res = run_inference_on_image(
                            shot_bgr, artifacts,
                            image_name=img_filename,
                            feature_mode=internal_feature_mode,
                            detector_mode=detector_mode,
                            enable_night_vision=auto_night_vision,
                            use_pca_recon_error=use_pca_recon,
                        )

                    annotated = res.get("severity_bgr") if res.get("severity_bgr") is not None else shot_bgr
                    cand_df = res.get("candidates_df")
                    n_cands = len(cand_df) if cand_df is not None and not cand_df.empty else 0

                    os.makedirs("outputs/evidence", exist_ok=True)
                    ev_path = os.path.join("outputs/evidence", img_filename)
                    cv2.imwrite(ev_path, annotated)

                    # Persist detections into database as Pending Review
                    if n_cands > 0:
                        for idx, r in cand_df.iterrows():
                            d_id = f"CAM_{timestamp_now}_{r.get('candidate_id', idx)}"
                            insert_defect(
                                detection_id=d_id,
                                track_id=str(r.get("candidate_id", idx)),
                                latitude=None, longitude=None,
                                severity_category=r.get("severity_category", "Medium"),
                                severity_score=float(r.get("severity_score", 0.5)),
                                anomaly_score=float(r.get("avg_anomaly_score", -0.2)),
                                pixel_area=int(r.get("pixel_area", 4000)),
                                image_path=ev_path,
                                source_media="Live Browser Camera",
                            )

                    # View detection
                    c_view1, c_view2 = st.columns([1.3, 1])
                    with c_view1:
                        st.image(bgr_to_rgb(annotated), caption=f"Detection View ({n_cands} defects found)", use_container_width=True)
                    with c_view2:
                        if n_cands > 0:
                            st.success(f"🚨 **{n_cands} Road Defect(s) Detected & Saved!**")
                            st.markdown(f"""
                            - **Status**: ⏳ Saved to **Citizen Review Queue** (`Pending Review`)
                            - **Severity**: {res.get('high_severity_count', 0)} High · {res.get('medium_severity_count', 0)} Medium · {res.get('low_severity_count', 0)} Low
                            - **Action**: Approve below to file directly to government authorities, or reject if false alarm.
                            """)
                            st.markdown("---")
                            auth_choice = st.selectbox(
                                "Target Authority",
                                list(config.GOV_PORTALS.keys()),
                                key=f"quick_portal_{timestamp_now}",
                                format_func=lambda k: f"{k}: {config.GOV_PORTALS[k]['name']}",
                            )
                            b_app, b_rej = st.columns(2)
                            with b_app:
                                if st.button("✅ Approve & File", type="primary", use_container_width=True, key=f"btn_cam_ap_{timestamp_now}"):
                                    top_r = cand_df.iloc[0]
                                    top_id = f"CAM_{timestamp_now}_{top_r.get('candidate_id', 0)}"
                                    d_payload = {
                                        "detection_id": top_id,
                                        "latitude": None, "longitude": None,
                                        "severity_category": top_r.get("severity_category", "Medium"),
                                        "severity_score": float(top_r.get("severity_score", 0.5)),
                                        "anomaly_score": float(top_r.get("avg_anomaly_score", -0.2)),
                                        "pixel_area": int(top_r.get("pixel_area", 4000)),
                                        "image_path": ev_path,
                                        "source_media": "Live Browser Camera",
                                    }
                                    d_res = dispatch_grievance_to_portal(d_payload, portal_key=auth_choice)
                                    update_grievance_status(
                                        detection_id=top_id,
                                        status="Dispatched",
                                        ref_no=d_res["reference_number"],
                                        portal=d_res["target_portal"],
                                    )
                                    CitizenNotifier().notify(
                                        event_dict={**d_payload, "reference_no": d_res["reference_number"], "target_portal": d_res["target_portal"]},
                                        notification_type="GRIEVANCE_FILED",
                                    )
                                    st.success(f"✅ Filed to {d_res['target_portal']}! Ref: {d_res['reference_number']}")
                                    st.rerun()
                            with b_rej:
                                if st.button("❌ Reject", use_container_width=True, key=f"btn_cam_rej_{timestamp_now}"):
                                    for idx, r in cand_df.iterrows():
                                        d_id = f"CAM_{timestamp_now}_{r.get('candidate_id', idx)}"
                                        disapprove_defect(d_id, reason="Rejected by citizen camera review")
                                    st.warning("Marked as rejected.")
                                    st.rerun()
                        else:
                            st.info("✅ Pavement surface is clear. No structural cavities detected.")

        else:
            saved_cam_cfg      = get_saved_camera_config()
            current_cam_idx    = saved_cam_cfg.get("camera_index", 0)
            current_src_type   = saved_cam_cfg.get("source_type", "usb_camera")
            current_stream_url = saved_cam_cfg.get("stream_url", "http://192.168.1.100:8080/video")

            # ── Camera Setup Expander ──────────────────────────────────────────
            with st.expander("⚙️  Camera & Vehicle Setup", expanded=False):
                hw_tab1, hw_tab2, hw_tab3 = st.tabs(["USB Dashcam", "Smartphone / RTSP", "Vehicle Profile"])

                with hw_tab1:
                    c_scan, c_status = st.columns([1, 2])
                    with c_scan:
                        scan_clicked = st.button("🔍  Scan Cameras", use_container_width=True)
                    with c_status:
                        if current_src_type == "usb_camera":
                            st.success(f"Active: Camera {current_cam_idx}")
                        else:
                            st.info(f"Mode: {current_src_type}")

                    if scan_clicked or "scanned_cams" in st.session_state:
                        if scan_clicked:
                            with st.spinner("Probing camera hardware…"):
                                st.session_state["scanned_cams"] = get_available_cameras(max_probe=4)
                        detected = st.session_state.get("scanned_cams", [])
                        if not detected:
                            st.warning("No USB cameras found. Connect a dashcam or use smartphone stream.")
                        else:
                            cam_cols = st.columns(min(len(detected), 3))
                            for i, cam in enumerate(detected):
                                with cam_cols[i % len(cam_cols)]:
                                    is_sel = cam["index"] == current_cam_idx and current_src_type == "usb_camera"
                                    with st.container(border=True):
                                        st.markdown(f"**Camera {cam['index']}** · {cam['width']}×{cam['height']}")
                                        st.image(bgr_to_rgb(cam["preview_bgr"]), use_container_width=True)
                                        if st.button(
                                            f"{'✅ ' if is_sel else ''}Set as Dashcam",
                                            key=f"set_cam_{cam['index']}",
                                            use_container_width=True,
                                            type="primary" if is_sel else "secondary",
                                        ):
                                            save_camera_config(camera_index=cam["index"], source_type="usb_camera")
                                            st.success(f"Saved Camera {cam['index']}")
                                            st.rerun()

                with hw_tab2:
                    st.caption("Mount your smartphone on the windshield and stream via IP Webcam app:")
                    p_url = st.text_input("Stream URL (RTSP / HTTP MJPEG):", value=current_stream_url)
                    if st.button("Save Stream URL", use_container_width=True):
                        save_camera_config(stream_url=p_url, source_type="phone_stream")
                        st.success(f"Saved: {p_url}")
                        st.rerun()

                with hw_tab3:
                    v_prof    = VehicleSuspensionProfiler()
                    curr_spec = v_prof.get_current_spec()
                    v_keys    = list(VEHICLE_PROFILES.keys())
                    col_vp1, col_vp2 = st.columns([2, 1])
                    with col_vp1:
                        v_choice = st.selectbox(
                            "Vehicle Class:",
                            v_keys,
                            index=v_keys.index(curr_spec["profile_key"]) if curr_spec["profile_key"] in v_keys else 0,
                            format_func=lambda k: VEHICLE_PROFILES[k]["display_name"],
                        )
                        chosen_spec = VEHICLE_PROFILES[v_choice]
                        if st.button("Save Vehicle Profile", use_container_width=True):
                            v_prof.save_profile(v_choice)
                            st.success(f"Saved: {chosen_spec['display_name']}")
                            st.rerun()
                    with col_vp2:
                        st.metric("Ground Clearance", f"{chosen_spec['ground_clearance_mm']} mm")
                        st.metric("G-Shock Threshold", f"{chosen_spec['shock_threshold_g']:.2f} g")

            # ── Survey Configuration ───────────────────────────────────────────
            with st.container(border=True):
                st.markdown("**Survey Configuration**")
                cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
                with cfg_col1:
                    feed_opts = [
                        "Benchmark Video (sample_road_test.mp4)",
                        "Upload Video / Record on Mobile (MP4 / Camera)",
                        f"USB Dashcam ({'Cam ' + str(current_cam_idx) if current_src_type == 'usb_camera' else 'Hardware Device'})",
                        "Wireless RTSP / HTTP Stream",
                    ]
                    dashcam_src = st.selectbox("Video Source", feed_opts, index=0, label_visibility="visible")
                with cfg_col2:
                    portal_keys  = list(config.GOV_PORTALS.keys())
                    target_portal = st.selectbox(
                        "Target Authority",
                        portal_keys,
                        format_func=lambda k: f"{k}",
                    )
                with cfg_col3:
                    min_sev = st.selectbox("Min Severity", ["Medium", "High", "Low"])
                with cfg_col4:
                    sample_rate = st.selectbox("Frame Stride", [2, 5, 10], index=1, format_func=lambda s: f"1 in {s} frames")

            dispatch_policy = st.radio(
                "Submission Policy",
                [
                    "📋  Record & Review — Save all detections; approve or reject before submitting",
                    "⚡  Auto-Dispatch — Immediately submit confirmed hazards to government portal",
                ],
                horizontal=True,
            )
            auto_dispatch = "Auto-Dispatch" in dispatch_policy

            # ── Source resolution ──────────────────────────────────────────────
            active_src     = None
            custom_vid_path = None
            if "Benchmark Video" in dashcam_src:
                active_src = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
            elif "Upload" in dashcam_src or "Record" in dashcam_src:
                up_vid = st.file_uploader("Upload or Record Road Video (MP4 / AVI)", type=["mp4", "avi", "mov"])
                if up_vid:
                    os.makedirs("temp", exist_ok=True)
                    custom_vid_path = os.path.join("temp", up_vid.name)
                    with open(custom_vid_path, "wb") as fv:
                        fv.write(up_vid.read())
                active_src = custom_vid_path
            elif "RTSP" in dashcam_src or "Wireless" in dashcam_src:
                active_src = current_stream_url
            elif "USB" in dashcam_src or dashcam_src.startswith("Auto"):
                active_src = current_cam_idx
            else:
                active_src = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")

            # ── Start button ───────────────────────────────────────────────────
            col_start, col_hint = st.columns([1, 3])
            with col_start:
                start_btn = st.button("▶  Start Road Survey", type="primary", use_container_width=True)
            with col_hint:
                st.caption("Analysis begins immediately on click. The video feed shows detected defects in real-time with bounding boxes and severity annotations.")

            if start_btn:
                # Fall back to sample video if camera or selected file is inaccessible
                if active_src is None or (isinstance(active_src, str) and not str(active_src).startswith("http") and not os.path.exists(str(active_src))):
                    fallback = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
                    if os.path.exists(fallback):
                        active_src = fallback
                    else:
                        st.error("Invalid video source. Upload a video file or connect a camera first.")

                if active_src is not None:
                    reporter = VehicleDashcamAutoReporter(
                        artifacts=artifacts,
                        target_portal=target_portal,
                        min_severity=min_sev,
                        confirmation_frames=2,
                        auto_dispatch=auto_dispatch,
                        sample_interval=sample_rate,
                        feature_mode=internal_feature_mode,
                        detector_mode=detector_mode,
                    )

                    cap = cv2.VideoCapture(active_src)
                    if not cap.isOpened():
                        fallback = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
                        if os.path.exists(fallback):
                            st.warning(f"Video source unavailable. Falling back to benchmark test video: {os.path.basename(fallback)}")
                            active_src = fallback
                            cap = cv2.VideoCapture(active_src)

                    if not cap.isOpened():
                        st.error(f"Cannot open video source: {active_src}. Please upload an MP4 video or check camera connections.")
                    else:
                        if str(active_src).isdigit() or str(active_src).startswith("http"):
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        frame_num   = 0
                        sampled_num = 0

                        col_vid, col_evts = st.columns([1.4, 1.0])
                        with col_vid:
                            st.markdown("**Detection Feed**")
                            vid_ph   = st.empty()
                            prog_bar = st.progress(0.0)
                        with col_evts:
                            st.markdown("**Detected Incidents**")
                            evts_ph = st.empty()
                            evts_ph.caption("Scanning pavement… confirmed defects appear here.")

                        all_dispatched = []

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret or frame is None or frame.size == 0:
                                break
                            frame_num += 1
                            if frame_num % sample_rate != 0:
                                continue
                            sampled_num += 1
                            t_sec = frame_num / fps

                            try:
                                frame_r = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
                                ann_frame, new_events, telem = reporter.process_frame(
                                    frame_bgr=frame_r,
                                    frame_idx=frame_num,
                                    timestamp_sec=t_sec,
                                    source_name="car_dashcam",
                                )
                            except Exception as frame_err:
                                print(f"[DASHCAM SURVEY EXCEPTION]: {frame_err}")
                                continue

                            vid_ph.image(bgr_to_rgb(ann_frame), use_container_width=True)
                            prog_bar.progress(min(1.0, sampled_num / 60.0))

                            if new_events:
                                all_dispatched.extend(new_events)
                                with evts_ph.container():
                                    for ev in reversed(all_dispatched[-4:]):
                                        cat    = ev.get("severity_category", "Medium")
                                        cat_l  = cat.lower()
                                        t_cls  = f"t-{cat_l}"
                                        d_cm   = ev.get("max_depth_cm", 0)
                                        vol_l  = ev.get("cavity_volume_liters", 0)
                                        mass   = ev.get("asphalt_repair_mass_kg", 0)
                                        road   = ev.get("road_name", "Survey Route")
                                        ref    = ev.get("reference_no", "PENDING")
                                        st.markdown(f"""
                                        <div class="ticket {t_cls}">
                                            <div class="ticket-header">
                                                <span class="ticket-road">{road}</span>
                                                <span class="ticket-ref">{ref}</span>
                                            </div>
                                            {_sev_chip(cat)}
                                            <div class="ticket-stats" style="margin-top:6px;">
                                                <span class="ticket-stat">Depth <span>{d_cm:.1f} cm</span></span>
                                                <span class="ticket-stat">Vol <span>{vol_l:.1f} L</span></span>
                                                <span class="ticket-stat">Fill <span>{mass:.1f} kg</span></span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        if ev.get("evidence_image") and os.path.exists(ev["evidence_image"]):
                                            st.image(ev["evidence_image"], use_container_width=True, caption="Photo Evidence")

                                        links_html = []
                                        if ev.get("whatsapp_url"):
                                            links_html.append(f'<a href="{ev["whatsapp_url"]}" target="_blank" style="display:inline-block; padding:3px 7px; background:#16a34a; color:#fff; border-radius:4px; text-decoration:none; font-size:0.75rem; margin-right:6px;">📲 WhatsApp</a>')
                                        if ev.get("google_maps_url"):
                                            links_html.append(f'<a href="{ev["google_maps_url"]}" target="_blank" style="display:inline-block; padding:3px 7px; background:#2563eb; color:#fff; border-radius:4px; text-decoration:none; font-size:0.75rem;">📍 Maps</a>')
                                        if links_html:
                                            st.markdown(f'<div style="margin:4px 0 8px 0;">{" ".join(links_html)}</div>', unsafe_allow_html=True)

                            if sampled_num >= 60:
                                break

                        cap.release()
                        st.success(f"Survey complete — {sampled_num} frames analysed · {len(all_dispatched)} defects logged")

                        if all_dispatched:
                            with st.expander(f"📋 Verified Defect Action Center ({len(all_dispatched)} incidents)", expanded=True):
                                for idx, ev in enumerate(all_dispatched):
                                    c_d1, c_d2 = st.columns([2, 1])
                                    with c_d1:
                                        st.markdown(f"**{ev.get('road_name', 'Highway')}** — Ref: `{ev.get('reference_no', 'N/A')}` · {ev.get('severity_category', 'Medium')} Severity")
                                        st.caption(f"Depth: {ev.get('max_depth_cm', 0):.1f} cm | Vol: {ev.get('cavity_volume_liters', 0):.1f} L | Fill: {ev.get('asphalt_repair_mass_kg', 0):.1f} kg")
                                    with c_d2:
                                        if ev.get("pdf_dossier") and os.path.exists(ev["pdf_dossier"]):
                                            with open(ev["pdf_dossier"], "rb") as fpdf:
                                                st.download_button(
                                                    "⬇ PDF Dossier",
                                                    fpdf.read(),
                                                    os.path.basename(ev["pdf_dossier"]),
                                                    "application/pdf",
                                                    key=f"post_dl_{idx}_{ev['detection_id']}",
                                                    use_container_width=True,
                                                )


    # =====================================================================
    # TAB 2 — CITIZEN REVIEW QUEUE
    # =====================================================================
    with tab_queue:
        render_citizen_queue_view(get_all_defects())

    # =====================================================================
    # TAB 2 — INSPECTION STUDIO
    # =====================================================================
    with tab_inspect:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon si-purple">🔬</div>
            <div>
                <div class="section-title">Inspection Studio</div>
                <div class="section-desc">Analyse individual road images or batch-process video files using the full ML pipeline</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        insp_mode = st.radio("Mode", ["📸  Single Image Analysis", "🎞️  Video File Processing"], horizontal=True)

        # ── Image Analysis ─────────────────────────────────────────────────
        if "Image" in insp_mode:
            left_col, right_col = st.columns([2, 1])
            with left_col:
                src_choice = st.selectbox(
                    "Image Source",
                    [
                        "📱 Mobile Camera (Live Browser Snap)",
                        "📁 Upload Image (JPG / PNG)",
                        "🛣️ Realistic Asphalt Benchmark",
                        "🔬 Synthetic Dataset Sample",
                    ],
                )

                image_bgr  = None
                image_name = "road_sample.jpg"
                run_cam_btn = False

                if "Mobile Camera" in src_choice:
                    st.info("📱 Point your mobile camera at the pavement defect and tap **Take Photo** below:")
                    cam_img = st.camera_input("Capture Road Surface", label_visibility="collapsed")
                    if cam_img:
                        image_name = "mobile_snap.jpg"
                        file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        run_cam_btn = st.button("🔍  Analyse Captured Surface", type="primary", use_container_width=True, key="btn_mobile_analyse")

                elif "Upload Image" in src_choice:
                    up_file = st.file_uploader("Choose a road surface image", type=["jpg", "jpeg", "png"])
                    if up_file:
                        image_name = up_file.name
                        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                elif "Benchmark" in src_choice:
                    real_dir = "data/real_road_samples"
                    if os.path.exists(real_dir):
                        real_samples = [f for f in sorted(os.listdir(real_dir)) if f.lower().endswith((".jpg", ".png"))]
                        if real_samples:
                            sel = st.selectbox("Select Benchmark Image", real_samples)
                            image_name = sel
                            image_bgr  = cv2.imread(os.path.join(real_dir, sel))
                        else:
                            st.caption("No benchmark images found.")
                    else:
                        st.warning("Run `python scripts/fetch_real_dataset.py` to download benchmark asphalt images.")

                elif "Synthetic" in src_choice:
                    s_dir = os.path.join(config.OUTPUT_DIR, "original")
                    if not os.path.exists(s_dir):
                        s_dir = config.INPUT_DIR
                    if os.path.exists(s_dir):
                        samples = [f for f in sorted(os.listdir(s_dir)) if any(f.lower().endswith(e) for e in config.SUPPORTED_EXTENSIONS)]
                        if samples:
                            sel = st.selectbox("Select Synthetic Sample", samples)
                            image_name = sel
                            image_bgr  = cv2.imread(os.path.join(s_dir, sel))

            with right_col:
                if image_bgr is not None:
                    h, w = image_bgr.shape[:2]
                    st.image(bgr_to_rgb(image_bgr), caption=f"{image_name}  ({w}×{h})", use_container_width=True)
                    run_img_btn = st.button("🔍  Analyse Surface", type="primary", use_container_width=True, key="btn_preview_analyse")
                    run_img_btn = run_img_btn or run_cam_btn
                else:
                    st.markdown("""
                    <div class="empty-state" style="padding:30px 20px;">
                        <div class="empty-state-icon">🖼️</div>
                        <div class="empty-state-title">No image loaded</div>
                        <div class="empty-state-desc">Select an image source on the left</div>
                    </div>
                    """, unsafe_allow_html=True)
                    run_img_btn = False

            if image_bgr is not None and run_img_btn:
                with st.spinner(f"Running {feature_mode} ({detector_mode}) pipeline…"):
                    results = run_inference_on_image(
                        image_bgr, artifacts,
                        image_name=image_name,
                        feature_mode=internal_feature_mode,
                        detector_mode=detector_mode,
                        enable_night_vision=auto_night_vision,
                        use_pca_recon_error=use_pca_recon,
                    )

                # Persist detections
                os.makedirs("outputs/evidence", exist_ok=True)
                for _, r in results["candidates_df"].iterrows():
                    det_id = f"{image_name}_{r['candidate_id']}"
                    saved_path = os.path.join("outputs/evidence", f"{det_id}.jpg")
                    cv2.imwrite(saved_path, results["severity_bgr"] if results.get("severity_bgr") is not None else image_bgr)

                    insert_defect(
                        detection_id=det_id,
                        track_id=r["candidate_id"],
                        latitude=None, longitude=None,
                        severity_category=r["severity_category"],
                        severity_score=r["severity_score"],
                        anomaly_score=r["avg_anomaly_score"],
                        pixel_area=r["pixel_area"],
                        image_path=saved_path,
                        source_media=image_name,
                    )

                # Metric summary
                r = results
                st.markdown(f"""
                <div class="result-metric-grid">
                    <div class="result-metric">
                        <div class="result-metric-num">{r['total_patches']}</div>
                        <div class="result-metric-lbl">Patches</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-num">{r['anomalous_patches']}</div>
                        <div class="result-metric-lbl">Anomalies</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-num">{r['candidate_count']}</div>
                        <div class="result-metric-lbl">Candidates</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-num" style="color:#22c55e">{r['low_severity_count']}</div>
                        <div class="result-metric-lbl">Low Sev.</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-num" style="color:#f59e0b">{r['medium_severity_count']}</div>
                        <div class="result-metric-lbl">Medium Sev.</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-num" style="color:#ef4444">{r['high_severity_count']}</div>
                        <div class="result-metric-lbl">High Sev.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Telemetry badges for unsupervised signals
                night_badge = ""
                if r.get("is_night_scene"):
                    night_badge = f'<span class="badge" style="background:rgba(234,179,8,0.15); color:#facc15; border:1px solid rgba(234,179,8,0.3); padding:4px 8px; border-radius:6px; font-size:0.75rem; margin-right:6px;">🌙 Night-Vision Enhanced (Lum: {r.get("night_luminance")})</span>'

                recon_badge = ""
                if r.get("mean_pca_recon_error", 0.0) > 0:
                    recon_badge = f'<span class="badge" style="background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:4px 8px; border-radius:6px; font-size:0.75rem; margin-right:6px;">📐 Mean PCA Recon MSE: {r["mean_pca_recon_error"]:.4f}</span>'

                detector_badge = f'<span class="badge" style="background:rgba(167,139,250,0.15); color:#a78bfa; border:1px solid rgba(167,139,250,0.3); padding:4px 8px; border-radius:6px; font-size:0.75rem;">⚡ {r.get("detector_mode_used", detector_mode)}</span>'

                st.markdown(f'<div style="margin: 10px 0 16px 0;">{night_badge}{recon_badge}{detector_badge}</div>', unsafe_allow_html=True)

                if results["candidate_count"] > 0:
                    st.success(f"✅ Detected {results['candidate_count']} candidate defects — logged to database.")
                else:
                    st.info("No significant defects detected. Pavement appears serviceable.")

                # Visualisation tabs
                vt1, vt2, vt3, vt4 = st.tabs([
                    "Severity Bounding Boxes",
                    "Spatial Heatmap",
                    "Anomaly Overlay",
                    "CLAHE Pre-processed",
                ])
                with vt1:
                    st.image(bgr_to_rgb(results["severity_bgr"]), use_container_width=True)
                with vt2:
                    st.image(bgr_to_rgb(results["heatmap_bgr"]), use_container_width=True)
                with vt3:
                    st.image(bgr_to_rgb(results["overlay_bgr"]), use_container_width=True)
                with vt4:
                    st.image(bgr_to_rgb(results["processed_bgr"]), use_container_width=True)

                cand_df = results["candidates_df"]
                if not cand_df.empty:
                    with st.expander("📊 Detailed Candidate Data"):
                        st.dataframe(cand_df, use_container_width=True)
                        dl1, dl2 = st.columns(2)
                        with dl1:
                            st.download_button(
                                "⬇ Download Annotated Image",
                                bgr_to_png_bytes(results["severity_bgr"]),
                                f"{image_name}_annotated.png", "image/png",
                                use_container_width=True,
                            )
                        with dl2:
                            st.download_button(
                                "⬇ Download Metrics CSV",
                                dataframe_to_csv_bytes(cand_df),
                                f"{image_name}_metrics.csv", "text/csv",
                                use_container_width=True,
                            )

        # ── Video Processing ───────────────────────────────────────────────
        else:
            vid_col1, vid_col2 = st.columns([3, 2])
            with vid_col1:
                video_file   = st.file_uploader("Upload Road Survey Video (MP4, AVI)", type=["mp4", "avi"])
                gps_csv_file = st.file_uploader("Optional: GPS Log CSV (timestamp_sec, lat, lon)", type=["csv"])
                samp_vid_path = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
                use_sample = False
                if not video_file and os.path.exists(samp_vid_path):
                    use_sample = st.checkbox("Use benchmark test video (sample_road_test.mp4)", value=True)
            with vid_col2:
                v_stride    = st.slider("Analysis Stride (frames)", min_value=5, max_value=30, value=10)
                st.caption(f"Processes every {v_stride}th frame — lower = more thorough, slower.")
                run_vid_btn = st.button("▶  Process Video", type="primary", use_container_width=True)

            target_video  = None
            temp_gps_path = None
            if video_file:
                os.makedirs("temp", exist_ok=True)
                target_video = os.path.join("temp", video_file.name)
                with open(target_video, "wb") as fv:
                    fv.write(video_file.read())
            elif use_sample:
                target_video = samp_vid_path

            if gps_csv_file:
                temp_gps_path = os.path.join("temp", gps_csv_file.name)
                with open(temp_gps_path, "wb") as fg:
                    fg.write(gps_csv_file.read())

            if run_vid_btn and target_video:
                with st.spinner("Processing video — tracking spatial defects across frames…"):
                    vid_res = process_road_video(
                        video_path=target_video,
                        gps_log_path=temp_gps_path,
                        sample_interval=v_stride,
                        artifacts=artifacts,
                    )
                st.success("Video processing complete.")
                vr1, vr2, vr3 = st.columns(3)
                vr1.metric("Frames Sampled", vid_res["total_frames_sampled"])
                vr2.metric("Unique Defects", vid_res["unique_defects_tracked"])
                vr3.metric("DB Entries", vid_res["total_detections_logged"])

                if vid_res["tracks"]:
                    st.dataframe(pd.DataFrame(vid_res["tracks"]), use_container_width=True)
                if os.path.exists(vid_res["output_video_path"]):
                    st.video(vid_res["output_video_path"])
            elif run_vid_btn:
                st.warning("No video file selected.")

    # =====================================================================
    # TAB 3 — GIS MAP & DATABASE
    # =====================================================================
    with tab_gis:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon si-green">🗺️</div>
            <div>
                <div class="section-title">GIS Map & Defect Database</div>
                <div class="section-desc">Interactive spatial map of all recorded defects with OGC GeoJSON, KML, and Smart City export</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        defects_df   = get_all_defects()
        geotagged_df = defects_df.dropna(subset=["latitude", "longitude"]) if (not defects_df.empty and "latitude" in defects_df.columns) else pd.DataFrame()

        # Filter bar
        filt_col1, filt_col2, filt_col3, filt_col4 = st.columns([1, 1, 1, 3])
        with filt_col1:
            show_low  = st.checkbox("🟢 Low",    value=True)
        with filt_col2:
            show_med  = st.checkbox("🟡 Medium", value=True)
        with filt_col3:
            show_high = st.checkbox("🔴 High",   value=True)
        with filt_col4:
            st.caption(f"**{len(defects_df)}** total records · **{len(geotagged_df)}** geotagged")

        active_filters = []
        if show_low:  active_filters.append("Low")
        if show_med:  active_filters.append("Medium")
        if show_high: active_filters.append("High")

        map_col, tools_col = st.columns([3, 1])
        with map_col:
            filtered = geotagged_df[geotagged_df["severity_category"].isin(active_filters)] if not geotagged_df.empty else pd.DataFrame()
            folium_map = build_geographic_map(filtered)
            st_folium(folium_map, width="100%", height=520)

        with tools_col:
            st.markdown("""
            <div class="section-header" style="margin-bottom:10px;">
                <div class="section-icon si-blue">📤</div>
                <div>
                    <div class="section-title" style="font-size:0.9rem;">GIS Export</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            g_bytes = geojson_to_bytes(defects_df)
            st.download_button(
                "⬇ GeoJSON (QGIS / ArcGIS)",
                g_bytes, "pothole_survey.geojson", "application/geo+json",
                use_container_width=True,
            )

            kml_bytes = kml_to_bytes(defects_df)
            st.download_button(
                "⬇ 3D KML (Google Earth)",
                kml_bytes, "potholes_3d.kml", "application/vnd.google-earth.kml+xml",
                use_container_width=True,
            )

            smart_feed = export_smart_city_wms_feed(defects_df)
            st.download_button(
                "⬇ Smart City API Feed (JSON)",
                _json.dumps(smart_feed, indent=2).encode("utf-8"),
                "smart_city_feed.json", "application/json",
                use_container_width=True,
            )

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            if not geotagged_df.empty:
                last_row = geotagged_df.iloc[0]
                loc_txt  = reverse_geocode_location(last_row["latitude"], last_row["longitude"])
                st.caption(f"**Latest GPS Fix**")
                st.caption(f"{loc_txt.get('road', 'Road')}")
                st.caption(f"{last_row['latitude']:.5f}, {last_row['longitude']:.5f}")
            else:
                st.caption("No GPS coordinates recorded yet.")

        # Database table
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon si-blue">🗄️</div>
            <div>
                <div class="section-title">SQLite Defect Registry</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not defects_df.empty:
            st.dataframe(defects_df, use_container_width=True, height=320)
            d_csv = dataframe_to_csv_bytes(defects_df)
            st.download_button(
                "⬇ Export Registry CSV",
                d_csv, "pothole_records.csv", "text/csv",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🗄️</div>
                <div class="empty-state-title">Database is empty</div>
                <div class="empty-state-desc">Run a Live Dashcam Survey or analyse images to populate records.</div>
            </div>
            """, unsafe_allow_html=True)

    # =====================================================================
    # TAB 4 — MUNICIPAL GOVERNANCE
    # =====================================================================
    with tab_gov:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon si-amber">🏛️</div>
            <div>
                <div class="section-title">Municipal Governance & SLA</div>
                <div class="section-desc">Citizen verification queue, grievance dispatch, work orders, contractor audits, and PCI analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        defects_df = get_all_defects()

        gov_tab1, gov_tab2, gov_tab3, gov_tab4 = st.tabs([
            "📄  Grievance Portal Dossiers",
            "🔧  Work Orders",
            "⚖️  Contractor Audit",
            "📊  PCI Rating",
        ])

        # ── 4a: Grievance Portal Dossiers ──────────────────────────────────
        with gov_tab1:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon si-blue">📄</div>
                <div>
                    <div class="section-title">Official Grievance Portal</div>
                    <div class="section-desc">Direct integration with CPGRAMS, NHAI Rajmargyatra, Meri Sadak, and Municipal PWD</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if defects_df.empty:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">📄</div>
                    <div class="empty-state-title">No defect records found</div>
                    <div class="empty-state-desc">Log defects first from the Dashcam Survey or Inspection Studio.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                defect_opts = []
                for _, r in defects_df.iterrows():
                    stat    = r.get("grievance_status") or "Unreported"
                    lat_s   = f"{r['latitude']:.4f}" if pd.notnull(r.get("latitude")) else "No GPS"
                    lon_s   = f"{r['longitude']:.4f}" if pd.notnull(r.get("longitude")) else ""
                    gps_lbl = f"({lat_s}, {lon_s})" if lon_s else f"({lat_s})"
                    defect_opts.append(f"[{r['id']}] {r['detection_id'][:30]} | {r['severity_category']} | {gps_lbl} | {stat}")

                sel_idx = st.selectbox("Select Defect Record", range(len(defect_opts)), format_func=lambda i: defect_opts[i])
                row     = defects_df.iloc[sel_idx]

                selected = {
                    "detection_id"      : row["detection_id"],
                    "track_id"          : row.get("track_id", "TRK_0"),
                    "latitude"          : float(row["latitude"])  if pd.notnull(row.get("latitude"))  else 28.6139,
                    "longitude"         : float(row["longitude"]) if pd.notnull(row.get("longitude")) else 77.2090,
                    "severity_category" : row.get("severity_category", "Medium"),
                    "severity_score"    : float(row.get("severity_score", 0.5)),
                    "anomaly_score"     : float(row.get("anomaly_score", -0.2)),
                    "pixel_area"        : int(row.get("pixel_area", 4000)),
                    "image_path"        : row.get("image_path") if pd.notnull(row.get("image_path")) and os.path.exists(str(row.get("image_path", ""))) else None,
                    "source_media"      : row.get("source_media", "camera_feed"),
                    "timestamp"         : row.get("timestamp"),
                    "current_status"    : row.get("grievance_status", "Unreported"),
                    "current_ref_no"    : row.get("grievance_ref_no"),
                }

                col_gv1, col_gv2 = st.columns(2)
                with col_gv1:
                    portal_keys      = list(config.GOV_PORTALS.keys())
                    chosen_portal    = st.selectbox("Target Authority", portal_keys, format_func=lambda k: f"{k}: {config.GOV_PORTALS[k]['name']}")
                    geo_info         = reverse_geocode_location(selected["latitude"], selected["longitude"])
                    st.caption(f"Jurisdiction: {geo_info.get('formatted_address', 'N/A')}")

                with col_gv2:
                    payload  = build_grievance_payload(selected, portal_key=chosen_portal)
                    pdf_path = generate_complaint_pdf(payload, defect_image_path=selected.get("image_path"))
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("📨  Dispatch Petition", type="primary", use_container_width=True):
                            res = dispatch_grievance_to_portal(selected, portal_key=chosen_portal)
                            CitizenNotifier().notify(
                                event_dict={**selected, "reference_no": res["reference_number"], "target_portal": res["target_portal"]},
                                notification_type="GRIEVANCE_FILED",
                            )
                            st.success(f"Filed to {res['target_portal']} · Ref: {res['reference_number']}")
                    with btn_col2:
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as fp:
                                st.download_button(
                                    "⬇ Download PDF Dossier",
                                    fp.read(), os.path.basename(pdf_path), "application/pdf",
                                    use_container_width=True,
                                )

        # ── 4b: Work Orders ────────────────────────────────────────────────
        with gov_tab2:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon si-amber">🔧</div>
                <div>
                    <div class="section-title">Municipal PWD Work Orders</div>
                    <div class="section-desc">Issue repair work orders, generate BOQ, and enforce 72-hour SLA under MoRTH Section 10</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            wo_mgr = MunicipalWorkOrderManager()
            wo_df  = wo_mgr.get_all_work_orders_df()

            wo_col1, wo_col2 = st.columns([2, 1])
            with wo_col1:
                if not wo_df.empty:
                    st.dataframe(wo_df, use_container_width=True, height=380)
                else:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-state-icon">🔧</div>
                        <div class="empty-state-title">No work orders yet</div>
                        <div class="empty-state-desc">Issue a new repair work order using the form →</div>
                    </div>
                    """, unsafe_allow_html=True)

            with wo_col2:
                with st.form("create_wo_form"):
                    st.markdown("**Issue New Work Order**")
                    f_ref        = st.text_input("Grievance Reference", value="NHAI/E/2026/87373")
                    f_road       = st.text_input("Road Location", value="NH-48 Corridor Km 42")
                    f_contractor = st.selectbox("Contractor", ["L&T Infrastructure", "Dilip Buildcon", "IRB Infrastructure", "Municipal PWD Division"])
                    f_depth      = st.number_input("Measured Depth (cm)", value=5.2, min_value=0.0)
                    f_vol        = st.number_input("Void Volume (L)", value=3.4, min_value=0.0)
                    f_repair     = st.number_input("Asphalt Required (kg)", value=8.2, min_value=0.0)
                    wo_submit    = st.form_submit_button("📋  Generate Work Order & BOQ", type="primary", use_container_width=True)

                if wo_submit:
                    new_wo = wo_mgr.create_work_order(
                        registry_id=f_ref,
                        reference_no=f_ref,
                        road_name=f_road,
                        assigned_contractor=f_contractor,
                        measured_depth_cm=f_depth,
                        cavity_volume_liters=f_vol,
                        asphalt_repair_mass_kg=f_repair,
                    )
                    st.success(f"Work Order **{new_wo['work_order_id']}** issued — ₹{new_wo['total_cost_inr']:,}")
                    st.rerun()

        # ── 4c: Contractor Audit ───────────────────────────────────────────
        with gov_tab3:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon si-red">⚖️</div>
                <div>
                    <div class="section-title">Contractor Warranty & DLP Audit</div>
                    <div class="section-desc">Defect Liability Period compliance · MoRTH Section 10.3 · ₹25,000/day liquidated damages</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            auditor  = HighwayContractorAuditor()
            projects = auditor.get_all_projects()

            if projects:
                proj_names  = [
                    f"{p.get('project_id', 'PRJ')}: {p.get('highway_name', p.get('project_name', 'Highway Stretch'))} — {p.get('contractor_name', 'Contractor')}"
                    for p in projects
                ]
                sel_proj    = st.selectbox("Select EPC Contract", range(len(projects)), format_func=lambda i: proj_names[i])
                chosen_proj = projects[sel_proj]
                audit_res   = auditor.audit_project_stretch(chosen_proj.get("project_id", ""))

                defects_found = audit_res.get("total_defects_found", 0)
                penalty_inr   = audit_res.get("liquidated_penalty_inr", 0)
                daily_fine    = int(chosen_proj.get("liquidated_damage_per_day_inr", 25000))
                status_lbl    = audit_res.get("compliance_status", "COMPLIANT").replace("_", " ")

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Defects on Stretch", defects_found)
                a2.metric("Contractor Liability", f"₹{penalty_inr:,}")
                a3.metric("Liquidated Damages", f"₹{daily_fine:,}/day")
                a4.metric("Compliance Status", status_lbl)

                pdf_audit = auditor.generate_audit_pdf(audit_res)
                if os.path.exists(pdf_audit):
                    with open(pdf_audit, "rb") as fa:
                        st.download_button(
                            "⬇ Download Statutory Audit Dossier (PDF)",
                            fa.read(), os.path.basename(pdf_audit), "application/pdf",
                            use_container_width=True,
                        )
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">⚖️</div>
                    <div class="empty-state-title">No EPC projects registered</div>
                    <div class="empty-state-desc">Register highway contracts to begin DLP penalty auditing.</div>
                </div>
                """, unsafe_allow_html=True)

        # ── 4d: PCI Rating ─────────────────────────────────────────────────
        with gov_tab4:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon si-purple">📊</div>
                <div>
                    <div class="section-title">Pavement Condition Index (PCI)</div>
                    <div class="section-desc">Standardised 0–100 rating per IRC:82-2015 and ASTM D6433 with maintenance strategy recommendations</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            distress_clf = PavementDistressClassifier()
            defects_list = defects_df.to_dict(orient="records") if not defects_df.empty else []
            pci_res      = distress_clf.compute_pci_rating(defects_list)

            pci_score  = pci_res["pci_score"]
            pci_rating = pci_res["rating"]

            # PCI colour
            if pci_score >= config.PCI_EXCELLENT_THRESH:
                pci_color = "#22c55e"
            elif pci_score >= config.PCI_POOR_THRESH:
                pci_color = "#f59e0b"
            else:
                pci_color = "#ef4444"

            pci_col1, pci_col2 = st.columns([1, 2])
            with pci_col1:
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:28px; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                    <div style="font-size:0.68rem; font-weight:700; color:#9ca3af; text-transform:uppercase; letter-spacing:0.7px; margin-bottom:8px;">
                        Pavement Condition Index
                    </div>
                    <div style="font-size:4rem; font-weight:900; color:{pci_color}; font-family:'JetBrains Mono',monospace; line-height:1;">
                        {pci_score}
                    </div>
                    <div style="font-size:0.75rem; color:#6b7280; margin-top:4px;">out of 100</div>
                    <div style="margin-top:12px;">
                        <span style="background:{pci_color}20; color:{pci_color}; font-weight:700; font-size:0.82rem; padding:4px 12px; border-radius:20px;">
                            {pci_rating}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                st.info(f"**Recommended Strategy:** {pci_res['pwd_strategy']}")

            with pci_col2:
                if pci_res["breakdown"]:
                    b_df = pd.DataFrame(list(pci_res["breakdown"].items()), columns=["Distress Category", "Count"])
                    b_df = b_df[b_df["Count"] > 0].sort_values("Count", ascending=True)
                    st.bar_chart(b_df.set_index("Distress Category"), horizontal=True)
                else:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-state-icon">📊</div>
                        <div class="empty-state-title">No distress data available</div>
                        <div class="empty-state-desc">Log defects to compute the Pavement Condition Index breakdown.</div>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
