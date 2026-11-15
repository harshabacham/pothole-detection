"""
app.py
======
Pavement Management & Road Surface Defect Detection System.
Autonomous automotive dashcam surveillance, 3D metric cavity profiling,
unsupervised anomaly detection, and Government of India grievance dispatch.
Compliant with IRC:82-2015, MoRTH Section 10, and ASTM D6433 standards.
"""

import os
import sys
import io
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
# Page Configuration & Clean Enterprise Stylesheet
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pavement Management & Road Distress Survey System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Enterprise Application Header */
    .app-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 14px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .app-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .app-subtitle {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: 3px;
    }
    .status-badge-group {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 500;
        background: #f8fafc;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    .status-badge.online {
        background: #f0fdf4;
        color: #166534;
        border-color: #bbf7d0;
    }

    /* KPI Summary Row */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        font-size: 0.70rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-number {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        margin: 2px 0 1px 0;
        font-family: 'Roboto Mono', monospace;
    }
    .metric-caption {
        font-size: 0.70rem;
        color: #94a3b8;
    }

    /* Municipal Ticket / Defect Cards */
    .ticket-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .ticket-card.high {
        border-left: 4px solid #dc2626;
    }
    .ticket-card.medium {
        border-left: 4px solid #d97706;
    }
    .ticket-card.low {
        border-left: 4px solid #16a34a;
    }
    .ticket-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .badge-high {
        background: #fee2e2;
        color: #b91c1c;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .badge-medium {
        background: #fef3c7;
        color: #b45309;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .badge-low {
        background: #dcfce7;
        color: #15803d;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .ticket-docket {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.74rem;
        color: #64748b;
        font-weight: 500;
    }
    .ticket-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .telemetry-table {
        width: 100%;
        font-size: 0.76rem;
        border-collapse: collapse;
        margin: 8px 0;
        background: #f8fafc;
        border-radius: 6px;
        overflow: hidden;
    }
    .telemetry-table td {
        padding: 4px 8px;
        border-bottom: 1px solid #f1f5f9;
    }
    .telemetry-table .label {
        color: #64748b;
        font-weight: 500;
        width: 50%;
    }
    .telemetry-table .val {
        color: #0f172a;
        font-weight: 600;
        font-family: 'Roboto Mono', monospace;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pre-fitted model artifacts...")
def get_cached_models():
    """Load and cache pre-fitted scaler, PCA, Isolation Forest, and deep feature models."""
    return load_all_artifacts()


def main():
    init_database()

    defects_df = get_all_defects()
    total_records = len(defects_df)
    high_threat_count = int((defects_df["severity_category"] == "High").sum()) if not defects_df.empty and "severity_category" in defects_df.columns else 0

    dedup_mgr = CrowdsourcedSpatialDeduplicator()
    dedup_summary = dedup_mgr.get_registry_summary()

    # Clean Enterprise Header Bar
    st.markdown("""
    <div class="app-header">
        <div>
            <h1 class="app-title">Road Surface Defect Detection & Grievance Reporting System</h1>
            <div class="app-subtitle">Autonomous Dashcam Pothole Surveillance, 3D Metric Depth Profiling & Citizen Grievance Portal (IRC:82 / MoRTH Compliant)</div>
        </div>
        <div class="status-badge-group">
            <span class="status-badge online">● System Operational (28-30 FPS)</span>
            <span class="status-badge">GPS Telemetry: Locked</span>
            <span class="status-badge">Portals: CPGRAMS / NHAI Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Restrained KPI Metrics Strip
    st.markdown(f"""
    <div class="kpi-row">
        <div class="metric-box">
            <div class="metric-title">Recorded Road Defects</div>
            <div class="metric-number">{total_records}</div>
            <div class="metric-caption">Stored in local SQLite database</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">High-Severity Hazards</div>
            <div class="metric-number" style="color:#b91c1c;">{high_threat_count}</div>
            <div class="metric-caption">Depth ≥ 5.0 cm (Wheel strike risk)</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Duplicate Filings Suppressed</div>
            <div class="metric-number" style="color:#0369a1;">{dedup_summary['duplicate_submissions_prevented']}</div>
            <div class="metric-caption">Crowdsourced corroborations logged</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Grievances Filed to Authorities</div>
            <div class="metric-number" style="color:#15803d;">{dedup_summary['total_citizen_reports']}</div>
            <div class="metric-caption">MoRTH 72-hour SLA tracking active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Sidebar: Clean Configuration & Diagnostics Panel
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### System Configuration")
    
    # Feature backbone selection
    feature_mode = st.sidebar.selectbox(
        "Feature Extraction Backbone:",
        [
            "Deep Features (ResNet-18)",
            "Fast OpenCV Texture (35+ FPS)",
            "Handcrafted (LBP+GLCM+Edge)",
        ],
        help="Deep Features uses PyTorch ResNet-18 transfer embeddings. Fast OpenCV uses pure C++ texture gradients for edge real-time performance.",
    )
    internal_feature_mode = "texture_only" if "Fast OpenCV" in feature_mode else feature_mode

    # Offline queue status
    offline_mgr = OfflineSyncManager()
    q_summary = offline_mgr.get_queue_summary()
    is_net_online = offline_mgr.is_online(timeout=0.25)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Connectivity & Queue")
    st.sidebar.caption(f"Network Status: {'Online' if is_net_online else 'Offline'}")
    st.sidebar.caption(f"Buffered Offline Petitions: {q_summary['pending_count']}")
    if not q_summary["is_clear"]:
        if st.sidebar.button("Flush Offline Queue", use_container_width=True):
            f_res = offline_mgr.flush_queue()
            st.sidebar.success(f"Synced {f_res['synced_count']} petitions!")
            st.rerun()

    # Hardware verification
    is_ready, missing_models = check_pipeline_readiness()
    if not is_ready:
        with st.sidebar:
            with st.spinner("Calibrating ML models on first launch..."):
                from auto_setup import ensure_models_and_data_ready
                ensure_models_and_data_ready(verbose=False)
                st.cache_resource.clear()
        is_ready, _ = check_pipeline_readiness()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Patch Size: {config.DEFAULT_PATCH_SIZE}×{config.DEFAULT_PATCH_SIZE} px")
    st.sidebar.caption(f"Contamination: {config.IFOREST_CONTAMINATION * 100:.0f}%")
    st.sidebar.caption("Method: Unsupervised Isolation Forest")

    artifacts = get_cached_models()

    # -----------------------------------------------------------------------
    # 4 Core Functional Workspaces (Replaces 9 Scattered Radio Options)
    # -----------------------------------------------------------------------
    tab_dashcam, tab_inspect, tab_gis, tab_gov = st.tabs([
        "Live Dashcam Survey",
        "Inspection Studio",
        "GIS Map & Database",
        "Municipal Governance & SLA",
    ])

    # =======================================================================
    # WORKSPACE 1: LIVE DASHCAM SURVEY
    # =======================================================================
    with tab_dashcam:
        saved_cam_cfg = get_saved_camera_config()
        current_cam_idx = saved_cam_cfg.get("camera_index", 0)
        current_src_type = saved_cam_cfg.get("source_type", "usb_camera")
        current_stream_url = saved_cam_cfg.get("stream_url", "http://192.168.1.100:8080/video")

        with st.expander("Camera Hardware Setup & Input Configuration", expanded=False):
            tab_hw_usb, tab_hw_rtsp, tab_hw_suspension = st.tabs([
                "USB Dashcam",
                "Smartphone / RTSP Stream",
                "Vehicle Suspension Profile",
            ])

            with tab_hw_usb:
                c_scan, c_status = st.columns([1, 2])
                with c_scan:
                    scan_cameras_clicked = st.button("Scan Connected Cameras", use_container_width=True)
                with c_status:
                    if current_src_type == "usb_camera":
                        st.success(f"Active Device: Camera {current_cam_idx}")
                    else:
                        st.info(f"Active Mode: {current_src_type}")

                if scan_cameras_clicked or "scanned_cams" in st.session_state:
                    if scan_cameras_clicked:
                        with st.spinner("Probing connected camera hardware..."):
                            st.session_state["scanned_cams"] = get_available_cameras(max_probe=4)

                    detected_cams = st.session_state.get("scanned_cams", [])
                    if not detected_cams:
                        st.warning("No physical USB cameras detected. Plug in your dashcam or use smartphone stream.")
                    else:
                        cam_cols = st.columns(min(len(detected_cams), 3))
                        for idx_c, cam_info in enumerate(detected_cams):
                            with cam_cols[idx_c % len(cam_cols)]:
                                is_selected = (cam_info["index"] == current_cam_idx and current_src_type == "usb_camera")
                                with st.container(border=True):
                                    st.markdown(f"**Camera {cam_info['index']}**")
                                    st.caption(f"Resolution: {cam_info['width']} × {cam_info['height']}")
                                    st.image(bgr_to_rgb(cam_info["preview_bgr"]), use_container_width=True)
                                    if st.button(
                                        f"Set Camera {cam_info['index']} as Road Dashcam",
                                        key=f"btn_set_cam_{cam_info['index']}",
                                        use_container_width=True,
                                        type="primary" if is_selected else "secondary",
                                    ):
                                        save_camera_config(camera_index=cam_info["index"], source_type="usb_camera")
                                        st.success(f"Saved Camera {cam_info['index']} as road dashcam.")
                                        st.rerun()

            with tab_hw_rtsp:
                st.caption("Mount a smartphone on the windshield and stream RTSP or HTTP MJPEG:")
                p_url = st.text_input("Stream RTSP / HTTP URL:", value=current_stream_url)
                if st.button("Save Smartphone Stream URL", use_container_width=True):
                    save_camera_config(stream_url=p_url, source_type="phone_stream")
                    st.success(f"Saved stream URL: {p_url}")
                    st.rerun()

            with tab_hw_suspension:
                v_prof = VehicleSuspensionProfiler()
                curr_spec = v_prof.get_current_spec()
                col_vp1, col_vp2 = st.columns([1.5, 1])
                with col_vp1:
                    v_keys = list(VEHICLE_PROFILES.keys())
                    v_choice = st.selectbox(
                        "Active Vehicle Class:",
                        v_keys,
                        index=v_keys.index(curr_spec["profile_key"]) if curr_spec["profile_key"] in v_keys else 0,
                        format_func=lambda k: f"{VEHICLE_PROFILES[k]['display_name']}",
                    )
                    chosen_spec = VEHICLE_PROFILES[v_choice]
                    if st.button("Save Suspension Profile", use_container_width=True):
                        v_prof.save_profile(v_choice)
                        st.success(f"Saved profile: {chosen_spec['display_name']}")
                        st.rerun()
                with col_vp2:
                    st.caption(f"Clearance: {chosen_spec['ground_clearance_mm']} mm | G-Shock Threshold: {chosen_spec['shock_threshold_g']:.2f}g | Multiplier: {chosen_spec['threat_multiplier']}x")

        # Dashcam Control Toolbar
        col_ctl1, col_ctl2, col_ctl3, col_ctl4 = st.columns([1.8, 1.4, 1.2, 1.2])
        with col_ctl1:
            feed_options = [
                f"Auto-Detected Dashcam ({'Camera ' + str(current_cam_idx) if current_src_type == 'usb_camera' else 'Phone Stream'})",
                "Benchmark Driving Footage (sample_road_test.mp4)",
                "Custom Video File Upload",
                "Wireless RTSP / HTTP Stream",
            ]
            dashcam_src_mode = st.selectbox("Video Source:", feed_options)
        with col_ctl2:
            portal_options = list(config.GOV_PORTALS.keys())
            target_portal_sel = st.selectbox(
                "Target Authority:",
                portal_options,
                format_func=lambda k: f"{k} ({config.GOV_PORTALS[k]['ministry_code']})",
            )
        with col_ctl3:
            min_sev_sel = st.selectbox("Dispatch Threshold:", ["Medium", "High", "Low"])
        with col_ctl4:
            sample_rate = st.selectbox("Sampling Stride:", [2, 5, 10], index=1, format_func=lambda s: f"1 in {s} frames")

        # Submission Policy: User Review vs Autonomous Auto-Dispatch
        dispatch_policy = st.radio(
            "Submission Policy:",
            [
                "Record & Review Mode (Auto-record location & details, user manually approves or disapproves)",
                "Autonomous Auto-Dispatch (Instantly submit to government without confirmation)",
            ],
            index=0,
            horizontal=True,
            help="In Record & Review mode, all potholes, GPS coordinates, and photos are saved in the database, allowing you to review and approve/disapprove each one.",
        )
        auto_dispatch_val = "Autonomous Auto-Dispatch" in dispatch_policy

        custom_video_path = None
        if dashcam_src_mode.startswith("Auto-Detected Dashcam"):
            active_src = get_active_dashcam_source()
        elif dashcam_src_mode == "Wireless RTSP / HTTP Stream":
            active_src = current_stream_url
        elif dashcam_src_mode == "Custom Video File Upload":
            up_vid = st.file_uploader("Upload MP4 / AVI Video:", type=["mp4", "avi"])
            if up_vid:
                os.makedirs("temp", exist_ok=True)
                custom_video_path = os.path.join("temp", up_vid.name)
                with open(custom_video_path, "wb") as f:
                    f.write(up_vid.read())
            active_src = custom_video_path
        else:
            active_src = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")

        start_survey_btn = st.button("Start Road Survey", type="primary", use_container_width=True)

        if start_survey_btn:
            if active_src is None or (isinstance(active_src, str) and not active_src.startswith("http") and not os.path.exists(active_src)):
                st.error("Invalid video source or disconnected camera.")
            else:
                reporter = VehicleDashcamAutoReporter(
                    artifacts=artifacts,
                    target_portal=target_portal_sel,
                    min_severity=min_sev_sel,
                    confirmation_frames=2,
                    auto_dispatch=auto_dispatch_val,
                    sample_interval=sample_rate,
                    feature_mode=internal_feature_mode,
                )

                cap = cv2.VideoCapture(active_src)
                if not cap.isOpened():
                    fallback_vid = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
                    if os.path.exists(fallback_vid):
                        st.warning(f"Active camera unreachable. Falling back to test footage ({os.path.basename(fallback_vid)}).")
                        active_src = fallback_vid
                        cap = cv2.VideoCapture(active_src)

                if not cap.isOpened():
                    st.error(f"Could not open video stream: {active_src}")
                else:
                    if isinstance(active_src, int) or str(active_src).isdigit() or str(active_src).startswith("http"):
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    frame_num = 0
                    sampled_num = 0

                    c_video, c_events = st.columns([1.3, 1.0])
                    with c_video:
                        st.markdown("##### Video Input & Detection Overlay")
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0.0)
                        telemetry_ticker = st.empty()

                    with c_events:
                        st.markdown("##### Logged Defect Incidents")
                        events_placeholder = st.container()
                        status_caption = events_placeholder.caption("Monitoring road surface... Confirmed defects will appear here.")

                    all_dispatched = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_num += 1
                        if frame_num % sample_rate != 0:
                            continue

                        sampled_num += 1
                        t_sec = frame_num / fps

                        frame_resized = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
                        annotated_frame, new_events, telem = reporter.process_frame(
                            frame_bgr=frame_resized,
                            frame_idx=frame_num,
                            timestamp_sec=t_sec,
                            source_name="car_dashcam",
                        )

                        video_placeholder.image(bgr_to_rgb(annotated_frame), use_container_width=True)
                        progress_pct = min(1.0, float(sampled_num) / 60.0)
                        progress_bar.progress(progress_pct)

                        telemetry_ticker.caption(
                            f"GPS: {telem.get('latitude', 0.0):.5f}, {telem.get('longitude', 0.0):.5f} | "
                            f"Speed: {telem.get('speed_kmh', 45):.0f} km/h | "
                            f"Shock: {telem.get('accel_z', 1.02):.2f}g | "
                            f"Frame: {frame_num} (Sampled: {sampled_num})"
                        )

                        if new_events:
                            all_dispatched.extend(new_events)
                            status_caption.empty()
                            with events_placeholder:
                                for ev in reversed(all_dispatched[-3:]):
                                    cat = ev.get('severity_category', 'Medium')
                                    cat_lower = cat.lower()
                                    card_cls = f"ticket-card {cat_lower}"
                                    badge_cls = f"badge-{cat_lower}"
                                    d_cm = ev.get('max_depth_cm', 5.2)
                                    vol_l = ev.get('cavity_volume_liters', 3.4)
                                    mass_kg = ev.get('asphalt_repair_mass_kg', 8.2)
                                    wheel_hazard = ev.get('wheel_track_hazard', 'LEFT_WHEEL_PATH')

                                    ticket_html = f"""
                                    <div class="{card_cls}">
                                        <div class="ticket-header">
                                            <span class="{badge_cls}">{cat.upper()} SEVERITY</span>
                                            <span class="ticket-docket">Ref: {ev.get('reference_no', 'PENDING')}</span>
                                        </div>
                                        <div class="ticket-title">{ev.get('road_name', 'Sector Road')}, {ev.get('district', 'Jurisdiction')}</div>
                                        <table class="telemetry-table">
                                            <tr><td class="label">Estimated Depth:</td><td class="val">{d_cm:.1f} cm</td></tr>
                                            <tr><td class="label">Void Volume:</td><td class="val">{vol_l:.1f} L</td></tr>
                                            <tr><td class="label">Asphalt Required:</td><td class="val">{mass_kg:.1f} kg</td></tr>
                                            <tr><td class="label">Wheel Path:</td><td class="val">{wheel_hazard}</td></tr>
                                            <tr><td class="label">Assigned Authority:</td><td class="val">{ev.get('target_portal', 'NHAI')}</td></tr>
                                        </table>
                                    </div>
                                    """
                                    st.markdown(ticket_html, unsafe_allow_html=True)

                                    if ev.get("evidence_image") and os.path.exists(ev["evidence_image"]):
                                        st.image(ev["evidence_image"], caption=f"Photometric Evidence: {ev.get('track_id', 'DEFECT')}", use_container_width=True)

                                    col_act1, col_act2 = st.columns(2)
                                    with col_act1:
                                        if ev.get("whatsapp_url"):
                                            st.link_button("Share via WhatsApp", ev["whatsapp_url"], use_container_width=True)
                                    with col_act2:
                                        if ev.get("google_maps_url"):
                                            st.link_button("View on Google Maps", ev["google_maps_url"], use_container_width=True)

                                    if ev.get("pdf_dossier") and os.path.exists(ev["pdf_dossier"]):
                                        with open(ev["pdf_dossier"], "rb") as f_pdf:
                                            pdf_bytes = f_pdf.read()
                                        st.download_button(
                                            label="Download Grievance Dossier (PDF)",
                                            data=pdf_bytes,
                                            file_name=os.path.basename(ev["pdf_dossier"]),
                                            mime="application/pdf",
                                            key=f"dl_{ev['detection_id']}",
                                            use_container_width=True,
                                        )

                        if sampled_num >= 60:
                            break

                    cap.release()
                    st.success(f"Road survey complete. Evaluated {sampled_num} frames. Logged {len(all_dispatched)} confirmed defects.")

    # =======================================================================
    # WORKSPACE 2: INSPECTION STUDIO (IMAGES & VIDEOS)
    # =======================================================================
    with tab_inspect:
        studio_mode = st.radio("Inspection Mode:", ["Single Image Inspection", "Video File Processing"], horizontal=True)

        if studio_mode == "Single Image Inspection":
            col_inp1, col_inp2 = st.columns([2, 1])
            with col_inp1:
                input_mode = st.selectbox(
                    "Select Input Source:",
                    ["Upload Image File", "Realistic Asphalt Benchmark", "Synthetic Dataset Sample", "Live Dashcam Snapshot"],
                )

                image_bgr = None
                image_name = "road_sample.jpg"

                if input_mode == "Upload Image File":
                    uploaded_file = st.file_uploader("Upload road image (JPG, PNG)", type=["jpg", "jpeg", "png"])
                    if uploaded_file is not None:
                        image_name = uploaded_file.name
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                elif input_mode == "Realistic Asphalt Benchmark":
                    real_dir = "data/real_road_samples"
                    if os.path.exists(real_dir):
                        real_samples = [f for f in sorted(os.listdir(real_dir)) if f.lower().endswith((".jpg", ".png"))]
                        if real_samples:
                            selected_sample = st.selectbox("Select Benchmark Sample:", real_samples)
                            image_name = selected_sample
                            image_bgr = cv2.imread(os.path.join(real_dir, selected_sample))
                    else:
                        st.info("Run `python scripts/fetch_real_dataset.py` to populate realistic asphalt samples.")

                elif input_mode == "Synthetic Dataset Sample":
                    sample_dir = os.path.join(config.OUTPUT_DIR, "original")
                    if not os.path.exists(sample_dir):
                        sample_dir = config.INPUT_DIR
                    if os.path.exists(sample_dir):
                        samples = [f for f in sorted(os.listdir(sample_dir)) if any(f.lower().endswith(ext) for ext in config.SUPPORTED_EXTENSIONS)]
                        if samples:
                            selected_sample = st.selectbox("Select Synthetic Sample:", samples)
                            image_name = selected_sample
                            image_bgr = cv2.imread(os.path.join(sample_dir, selected_sample))

                elif input_mode == "Live Dashcam Snapshot":
                    cam_img = st.camera_input("Take Snapshot")
                    if cam_img is not None:
                        image_name = "dashcam_live.jpg"
                        file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with col_inp2:
                if image_bgr is not None:
                    h, w = image_bgr.shape[:2]
                    st.image(bgr_to_rgb(image_bgr), caption=f"{image_name} ({w}×{h} px)", use_container_width=True)
                    run_img_btn = st.button("Analyze Road Surface", type="primary", use_container_width=True)

            if image_bgr is not None and 'run_img_btn' in locals() and run_img_btn:
                with st.spinner(f"Evaluating surface with {feature_mode}..."):
                    results = run_inference_on_image(
                        image_bgr,
                        artifacts,
                        image_name=image_name,
                        feature_mode=internal_feature_mode,
                    )

                # Persist to SQLite
                for _, r in results["candidates_df"].iterrows():
                    insert_defect(
                        detection_id=f"{image_name}_{r['candidate_id']}",
                        track_id=r["candidate_id"],
                        latitude=None,
                        longitude=None,
                        severity_category=r["severity_category"],
                        severity_score=r["severity_score"],
                        anomaly_score=r["avg_anomaly_score"],
                        pixel_area=r["pixel_area"],
                        image_path=image_name,
                        source_media=image_name,
                    )

                st.success(f"Analysis Complete. Detected {results['candidate_count']} candidate defects.")

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Patches", results["total_patches"])
                m2.metric("Anomalies", results["anomalous_patches"])
                m3.metric("Defect Candidates", results["candidate_count"])
                m4.metric("Low Severity", results["low_severity_count"])
                m5.metric("Medium Severity", results["medium_severity_count"])
                m6.metric("High Severity", results["high_severity_count"])

                tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs([
                    "Severity Bounding Boxes",
                    "Spatial Density Heatmap",
                    "Anomaly Patch Overlay",
                    "CLAHE Preprocessed",
                ])
                with tab_t1:
                    st.image(bgr_to_rgb(results["severity_bgr"]), use_container_width=True)
                with tab_t2:
                    st.image(bgr_to_rgb(results["heatmap_bgr"]), use_container_width=True)
                with tab_t3:
                    st.image(bgr_to_rgb(results["overlay_bgr"]), use_container_width=True)
                with tab_t4:
                    st.image(bgr_to_rgb(results["processed_bgr"]), use_container_width=True)

                cand_df = results["candidates_df"]
                if not cand_df.empty:
                    st.dataframe(cand_df, use_container_width=True)
                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        st.download_button("Download Annotated Image (PNG)", bgr_to_png_bytes(results["severity_bgr"]), f"{image_name}_annotated.png", "image/png", use_container_width=True)
                    with c_dl2:
                        st.download_button("Download Defect Metrics (CSV)", dataframe_to_csv_bytes(cand_df), f"{image_name}_metrics.csv", "text/csv", use_container_width=True)

        else:
            # Video File Processing
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                video_file = st.file_uploader("Upload Road Survey Video (MP4, AVI)", type=["mp4", "avi"])
                gps_csv_file = st.file_uploader("Optional GPS Log (timestamp_sec, lat, lon)", type=["csv"])
                sample_video_path = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
                use_sample = False
                if not video_file and os.path.exists(sample_video_path):
                    use_sample = st.checkbox("Or use benchmark test video (sample_road_test.mp4)", value=True)

            with col_v2:
                v_stride = st.slider("Sampling Stride (frames):", min_value=5, max_value=30, value=10)
                run_vid_btn = st.button("Process Video Stream", type="primary", use_container_width=True)

            target_video = None
            if video_file:
                os.makedirs("temp", exist_ok=True)
                target_video = os.path.join("temp", video_file.name)
                with open(target_video, "wb") as f:
                    f.write(video_file.read())
            elif use_sample:
                target_video = sample_video_path

            temp_gps_path = None
            if gps_csv_file:
                temp_gps_path = os.path.join("temp", gps_csv_file.name)
                with open(temp_gps_path, "wb") as f:
                    f.write(gps_csv_file.read())

            if run_vid_btn and target_video:
                with st.spinner("Processing video frames and tracking spatial defects..."):
                    vid_res = process_road_video(
                        video_path=target_video,
                        gps_log_path=temp_gps_path,
                        sample_interval=v_stride,
                        artifacts=artifacts,
                    )

                st.success("Video processing complete.")
                vm1, vm2, vm3 = st.columns(3)
                vm1.metric("Sampled Frames", vid_res["total_frames_sampled"])
                vm2.metric("Unique Tracked Defects", vid_res["unique_defects_tracked"])
                vm3.metric("Database Entries Logged", vid_res["total_detections_logged"])

                if vid_res["tracks"]:
                    st.dataframe(pd.DataFrame(vid_res["tracks"]), use_container_width=True)

                if os.path.exists(vid_res["output_video_path"]):
                    st.video(vid_res["output_video_path"])

    # =======================================================================
    # WORKSPACE 3: GIS MAP & DATABASE REGISTRY
    # =======================================================================
    with tab_gis:
        defects_df = get_all_defects()
        geotagged_df = defects_df.dropna(subset=["latitude", "longitude"]) if not defects_df.empty else pd.DataFrame()

        col_gm1, col_gm2, col_gm3, col_gm4 = st.columns([1, 1, 1, 3])
        with col_gm1:
            show_low = st.checkbox("Low Severity", value=True)
        with col_gm2:
            show_med = st.checkbox("Medium Severity", value=True)
        with col_gm3:
            show_high = st.checkbox("High Severity", value=True)

        active_filters = []
        if show_low: active_filters.append("Low")
        if show_med: active_filters.append("Medium")
        if show_high: active_filters.append("High")

        with col_gm4:
            st.caption(f"Database Records: {len(defects_df)} | Geotagged Coordinates: {len(geotagged_df)}")

        col_map, col_gis_tools = st.columns([3, 1])
        with col_map:
            folium_map = build_geographic_map(defects_df, severity_filter=active_filters)
            st_folium(folium_map, width=950, height=480)

        with col_gis_tools:
            st.markdown("##### GIS & Engineering Exports")
            if not geotagged_df.empty:
                g_bytes = geojson_to_bytes(geotagged_df)
                st.download_button("Export GeoJSON (QGIS)", g_bytes, "pothole_survey.geojson", "application/geo+json", use_container_width=True)

                kml_data = kml_to_bytes(geotagged_df, extrude_3d=True)
                st.download_button("Export 3D KML (Google Earth)", kml_data, "potholes_3d.kml", "application/vnd.google-earth.kml+xml", use_container_width=True)

                import json as _json
                smart_feed = export_smart_city_wms_feed(geotagged_df)
                st.download_button("Export Smart City API Feed (JSON)", _json.dumps(smart_feed, indent=2).encode("utf-8"), "smart_city_potholes.json", "application/json", use_container_width=True)
            else:
                st.caption("No GPS points recorded yet.")

        st.markdown("---")
        st.markdown("##### Database Defect Records (SQLite)")
        if not defects_df.empty:
            st.dataframe(defects_df, use_container_width=True)
            d_csv = dataframe_to_csv_bytes(defects_df)
            st.download_button("Export Database Table (CSV)", d_csv, "pothole_records.csv", "text/csv", use_container_width=True)
        else:
            st.caption("Database is currently empty. Run a survey or inspect images to log records.")

    # =======================================================================
    # WORKSPACE 4: MUNICIPAL GOVERNANCE & SLA
    # =======================================================================
    with tab_gov:
        gov_mode = st.radio(
            "Governance Module:",
            [
                "Citizen Verification Queue (Approve / Disapprove)",
                "Official Grievance Dossier (CPGRAMS / NHAI)",
                "Municipal Work Orders & 72h SLA",
                "Contractor Warranty & DLP Audit",
                "Pavement Condition Index (PCI)",
            ],
            horizontal=True,
        )

        defects_df = get_all_defects()

        if "Citizen Verification Queue" in gov_mode:
            st.markdown("##### Citizen Verification & Manual Approval Queue")
            st.caption("Review automatically recorded road cavities before dispatching petitions to Government of India authorities. Reject false positives or approve verified hazards.")

            if defects_df.empty:
                st.info("No recorded potholes in the database yet. Run a Live Dashcam Survey or inspect sample images to populate records.")
            else:
                # Compute KPI statistics
                status_series = defects_df["grievance_status"].fillna("Unreported")
                pending_mask = status_series.isin(["Pending Review", "Unreported", "PORTAL_SYNC_QUEUED", "PENDING_USER_REVIEW"])
                dispatched_mask = status_series.str.startswith("Dispatched", na=False)
                disapproved_mask = status_series.str.startswith("Disapproved", na=False)

                pending_df = defects_df[pending_mask]
                dispatched_df = defects_df[dispatched_mask]
                disapproved_df = defects_df[disapproved_mask]

                c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
                c_kpi1.metric("Total Recorded", len(defects_df))
                c_kpi2.metric("Awaiting Approval", len(pending_df))
                c_kpi3.metric("Approved & Dispatched", len(dispatched_df))
                c_kpi4.metric("Disapproved (Rejected)", len(disapproved_df))

                col_flt1, col_flt2 = st.columns([2, 1])
                with col_flt1:
                    filter_choice = st.selectbox(
                        "Display Filter:",
                        [
                            f"Pending Review ({len(pending_df)})",
                            f"Approved & Dispatched ({len(dispatched_df)})",
                            f"Disapproved ({len(disapproved_df)})",
                            f"All Records ({len(defects_df)})",
                        ],
                    )

                if "Pending Review" in filter_choice:
                    active_view_df = pending_df
                elif "Approved" in filter_choice:
                    active_view_df = dispatched_df
                elif "Disapproved" in filter_choice:
                    active_view_df = disapproved_df
                else:
                    active_view_df = defects_df

                if active_view_df.empty:
                    st.caption("No records found in this view category.")
                else:
                    st.markdown("---")
                    for _, row in active_view_df.iterrows():
                        det_id = row["detection_id"]
                        status = row.get("grievance_status") or "Pending Review"
                        cat = row.get("severity_category", "Medium")
                        cat_lower = str(cat).lower()
                        d_cm = float(row.get("bev_area_sqcm", 400.0) / 100.0) if pd.notnull(row.get("bev_area_sqcm")) else 5.2
                        vol_l = round(d_cm * 0.65, 1)

                        lat = float(row["latitude"]) if pd.notnull(row["latitude"]) else 28.5383
                        lon = float(row["longitude"]) if pd.notnull(row["longitude"]) else 77.2431
                        img_path = row.get("image_path")

                        col_card_img, col_card_info, col_card_act = st.columns([1.1, 1.9, 1.4])

                        with col_card_img:
                            if img_path and os.path.exists(str(img_path)):
                                st.image(img_path, caption=f"Defect ID: {det_id}", use_container_width=True)
                            else:
                                st.markdown(f"""
                                <div style="background:#f1f5f9; border-radius:6px; padding:24px 12px; text-align:center; color:#64748b; font-size:0.75rem;">
                                    <b>{det_id}</b><br/>Photometric Frame
                                </div>
                                """, unsafe_allow_html=True)

                        with col_card_info:
                            st.markdown(f"**Pothole Record: `{det_id}`**")
                            st.caption(f"Recorded at {row.get('timestamp', 'N/A')} | Source: {row.get('source_media', 'Dashcam')}")
                            st.markdown(f"""
                            - **GPS Coordinates**: `{lat:.5f}, {lon:.5f}`
                            - **Severity**: `{cat}` (Score: `{float(row.get('severity_score', 0.5)):.2f}`)
                            - **Estimated Depth**: `{d_cm:.1f} cm` | **Volume**: `{vol_l:.1f} L`
                            - **Current Status**: `{status}`
                            """)

                        with col_card_act:
                            if status in ["Pending Review", "Unreported", "PORTAL_SYNC_QUEUED", "PENDING_USER_REVIEW"]:
                                st.markdown("**Citizen Action:**")
                                target_p = st.selectbox(
                                    "Target Ministry:",
                                    list(config.GOV_PORTALS.keys()),
                                    key=f"sel_p_{det_id}",
                                    format_func=lambda k: f"{k} ({config.GOV_PORTALS[k]['name'][:15]}...)",
                                )

                                col_sub_ap, col_sub_dis = st.columns(2)
                                with col_sub_ap:
                                    if st.button("✅ Approve & Submit", key=f"appr_btn_{det_id}", type="primary", use_container_width=True):
                                        defect_dict = {
                                            "detection_id": det_id,
                                            "track_id": row.get("track_id", "TRK_0"),
                                            "latitude": lat,
                                            "longitude": lon,
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
                                        st.success(f"Filed to {res['target_portal']} (Ref: {res['reference_number']})")
                                        st.rerun()

                                with col_sub_dis:
                                    if st.button("❌ Disapprove", key=f"disappr_btn_{det_id}", use_container_width=True):
                                        disapprove_defect(det_id, reason="Rejected by citizen (False positive / Not a defect)")
                                        st.warning("Pothole Disapproved. Will not be sent to authorities.")
                                        st.rerun()
                            elif status.startswith("Dispatched"):
                                st.success(f"Dispatched: {row.get('grievance_ref_no', 'N/A')}")
                                st.caption(f"Portal: {row.get('dispatched_portal', 'CPGRAMS')}")
                            elif status.startswith("Disapproved"):
                                st.error("Disapproved (Flagged False Positive)")
                                if st.button("Revert to Pending", key=f"rev_{det_id}", use_container_width=True):
                                    update_grievance_status(det_id, status="Pending Review", ref_no="PENDING_REVIEW")
                                    st.rerun()

                        st.markdown("<hr style='margin:10px 0; border:0; border-top:1px solid #f1f5f9;'>", unsafe_allow_html=True)

        elif "Official Grievance Dossier" in gov_mode or "Grievance Portal" in gov_mode:
            st.markdown("##### Government Grievance Transmission Channel")
            st.caption("Direct integration with CPGRAMS (MoRTH), NHAI (Rajmargyatra), PMGSY (Meri Sadak), and Municipal PWD divisions.")

            if defects_df.empty:
                st.info("No defect records found in SQLite. Record defects in Live Dashcam or Inspection Studio first.")
            else:
                defect_options = []
                for _, r in defects_df.iterrows():
                    stat = r.get("grievance_status") or "Unreported"
                    lat_str = f"{r['latitude']:.4f}" if pd.notnull(r.get("latitude")) else "No GPS"
                    lon_str = f"{r['longitude']:.4f}" if pd.notnull(r.get("longitude")) else ""
                    gps_lbl = f"({lat_str}, {lon_str})" if lon_str else f"({lat_str})"
                    defect_options.append(f"[{r['id']}] {r['detection_id']} | Sev: {r['severity_category']} | {gps_lbl} | Status: {stat}")

                sel_idx = st.selectbox("Select Defect Record to File:", range(len(defect_options)), format_func=lambda i: defect_options[i])
                row = defects_df.iloc[sel_idx]

                selected_defect = {
                    "detection_id": row["detection_id"],
                    "track_id": row.get("track_id", "TRK_0"),
                    "latitude": float(row["latitude"]) if pd.notnull(row["latitude"]) else 28.6139,
                    "longitude": float(row["longitude"]) if pd.notnull(row["longitude"]) else 77.2090,
                    "severity_category": row.get("severity_category", "Medium"),
                    "severity_score": float(row.get("severity_score", 0.5)),
                    "anomaly_score": float(row.get("anomaly_score", -0.2)),
                    "pixel_area": int(row.get("pixel_area", 4000)),
                    "image_path": row.get("image_path") if pd.notnull(row.get("image_path")) and os.path.exists(str(row.get("image_path"))) else None,
                    "source_media": row.get("source_media", "camera_feed"),
                    "timestamp": row.get("timestamp"),
                    "current_status": row.get("grievance_status", "Unreported"),
                    "current_ref_no": row.get("grievance_ref_no"),
                }

                col_gv1, col_gv2 = st.columns(2)
                with col_gv1:
                    portal_keys = list(config.GOV_PORTALS.keys())
                    chosen_portal_key = st.selectbox("Select Authority:", portal_keys, format_func=lambda k: f"{k}: {config.GOV_PORTALS[k]['name']}")
                    geo_info = reverse_geocode_location(selected_defect["latitude"], selected_defect["longitude"])
                    st.caption(f"Jurisdiction: {geo_info['formatted_address']}")

                with col_gv2:
                    payload = build_grievance_payload(selected_defect, portal_key=chosen_portal_key)
                    pdf_path = generate_complaint_pdf(payload, defect_image_path=selected_defect.get("image_path"))
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("Dispatch Grievance Petition", type="primary", use_container_width=True):
                            res = dispatch_grievance_to_portal(selected_defect, portal_key=chosen_portal_key)
                            c_notif = CitizenNotifier()
                            c_notif.notify(
                                event_dict={**selected_defect, "reference_no": res["reference_number"], "target_portal": res["target_portal"]},
                                notification_type="GRIEVANCE_FILED",
                            )
                            st.success(f"Filed to {res['target_portal']} (Ref: {res['reference_number']}).")
                    with col_b2:
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f_pdf:
                                st.download_button("Download Official PDF Dossier", f_pdf.read(), os.path.basename(pdf_path), "application/pdf", use_container_width=True)

        elif "Work Orders" in gov_mode:
            st.markdown("##### Municipal PWD Work Orders & Statutory 72-Hour SLA Resolution")
            st.caption("Enforces Section 10 MoRTH 72-hour repair compliance, generates Schedule of Rates (SoR) BOQ, and certifies repair closures.")

            wo_mgr = MunicipalWorkOrderManager()
            wo_list = wo_mgr.get_all_work_orders()
            wo_df = wo_mgr.get_all_work_orders_df()

            col_w1, col_w2 = st.columns([2, 1])
            with col_w1:
                if not wo_df.empty:
                    st.dataframe(wo_df, use_container_width=True)
                else:
                    st.caption("No work orders issued yet.")
            with col_w2:
                with st.form("create_wo_form"):
                    st.markdown("**Issue New Repair Work Order**")
                    f_reg_id = st.text_input("Grievance Reference:", value="NHAI/E/2026/87373")
                    f_road = st.text_input("Road Location:", value="NH-48 Corridor Km 42")
                    f_contractor = st.selectbox("Assign EPC Contractor:", ["L&T Infrastructure", "Dilip Buildcon", "IRB Infrastructure", "Municipal PWD Division"])
                    f_depth = st.number_input("Measured Depth (cm):", value=5.2)
                    f_vol = st.number_input("Void Volume (L):", value=3.4)
                    f_repair = st.number_input("Asphalt Required (kg):", value=8.2)
                    wo_submit = st.form_submit_button("Generate Work Order & BOQ")

                if wo_submit:
                    new_wo = wo_mgr.create_work_order(
                        registry_id=f_reg_id,
                        reference_no=f_reg_id,
                        road_name=f_road,
                        assigned_contractor=f_contractor,
                        measured_depth_cm=f_depth,
                        cavity_volume_liters=f_vol,
                        asphalt_repair_mass_kg=f_repair,
                    )
                    st.success(f"Work Order {new_wo['work_order_id']} issued! Total Cost: ₹{new_wo['total_cost_inr']:,}")
                    st.rerun()

        elif "Contractor" in gov_mode:
            st.markdown("##### Highway Contractor Warranty & DLP Penalty Audit")
            st.caption("Audits Defect Liability Period (DLP) compliance under MoRTH Section 10.3 and calculates statutory liquidated damages (₹25,000/day).")

            auditor = HighwayContractorAuditor()
            projects = auditor.get_all_projects()

            if projects:
                proj_names = [f"{p['project_code']}: {p['project_name']} ({p['contractor_name']})" for p in projects]
                sel_proj_idx = st.selectbox("Select Highway EPC Contract:", range(len(projects)), format_func=lambda i: proj_names[i])
                chosen_proj = projects[sel_proj_idx]

                audit_res = auditor.audit_project_stretch(chosen_proj["project_id"])

                c_aud1, c_aud2, c_aud3, c_aud4 = st.columns(4)
                c_aud1.metric("Defects on Stretch", audit_res["total_defects_found"])
                c_aud2.metric("Contractor Liability", f"₹{audit_res['contractor_liability_inr']:,}")
                c_aud3.metric("Liquidated Penalties", f"₹{audit_res['daily_liquidated_damages_inr']:,}")
                c_aud4.metric("Contractor Grade", audit_res["contractor_performance_tier"])

                pdf_audit_path = auditor.generate_audit_pdf(audit_res)
                if os.path.exists(pdf_audit_path):
                    with open(pdf_audit_path, "rb") as f_audit:
                        st.download_button("Download Statutory Contractor Audit Dossier (PDF)", f_audit.read(), os.path.basename(pdf_audit_path), "application/pdf", use_container_width=True)

        elif "PCI" in gov_mode:
            st.markdown("##### Pavement Condition Index (PCI) & Distress Taxonomy")
            st.caption("Standardized roadway quality rating (0 to 100) according to IRC:82-2015 and ASTM D6433.")

            distress_clf = PavementDistressClassifier()
            defects_list = defects_df.to_dict(orient="records") if not defects_df.empty else []
            pci_res = distress_clf.compute_pci_rating(defects_list)

            c_pci1, c_pci2 = st.columns(2)
            with c_pci1:
                st.metric("Pavement Condition Index (PCI)", f"{pci_res['pci_score']} / 100", pci_res["rating"])
                st.info(f"Recommended Maintenance Strategy: {pci_res['pwd_strategy']}")
            with c_pci2:
                if pci_res["breakdown"]:
                    b_df = pd.DataFrame(list(pci_res["breakdown"].items()), columns=["Distress Category", "Count"])
                    st.bar_chart(b_df.set_index("Distress Category"))
                else:
                    st.caption("No defects recorded yet.")


if __name__ == "__main__":
    main()
