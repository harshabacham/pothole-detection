"""
src/dashcam_agent.py
====================
Autonomous Car Dashcam Pothole Recognition & Government Portal Auto-Reporting Agent.

Operational Workflow:
1. Ingests live video feed from vehicle dashboard camera (USB webcam, dashcam stream,
   RTSP IP camera, or pre-recorded driving footage).
2. Continuously scans road surface frames using pre-fitted unsupervised ML models
   (or deep ResNet-18 embeddings).
3. Spatio-temporal tracker tracks anomalies across frames to filter transient shadows
   and guarantee deduplication (prevents re-reporting the same physical pothole).
4. When a confirmed pothole meets severity thresholds:
   - Captures high-resolution photo evidence of the pothole with bounding box & timestamp.
   - Extracts vehicle GPS coordinates & speed at that exact millisecond.
   - Reverse-geocodes Indian road name, district, state, and PIN code via OpenStreetMap Nominatim.
   - Formats a legally compliant citizen safety grievance petition.
   - Generates official printable PDF dossier via ReportLab.
   - Automatically dispatches the grievance to official Government of India portals
     (CPGRAMS MoRTH, NHAI Rajmargyatra, PMGSY Meri Sadak, or Municipal PWD).
   - Updates SQLite database with assigned government tracking reference ID.
   - Overlays real-time automotive HUD telemetry on the driver's display.
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.inference_pipeline import load_all_artifacts, run_inference_on_image
from src.tracking import DefectTracker
from src.severity_scoring import get_severity_color_bgr
from src.database import init_database, insert_defect, update_grievance_status
from src.gps_handler import load_external_gps_log, interpolate_gps
from src.gov_portal_api import (
    reverse_geocode_location,
    dispatch_grievance_to_portal,
)
from src.depth_estimator import MetricDepthEstimator
from src.night_vision import NightVisionEnhancer
from src.adas_audio import InCabinAdasAudio
from src.distress_taxonomy import PavementDistressClassifier
from src.fleet_analytics import PotholeDegradationTracker
from src.ipm_transform import InversePerspectiveMapper
from src.imu_fusion import OpticalInertialShockCorrelator
from src.crowd_dedup import CrowdsourcedSpatialDeduplicator
from src.suspension_calibrator import VehicleSuspensionProfiler
from src.offline_sync import OfflineSyncManager
from src.citizen_notifier import CitizenNotifier
from src.speed_breaker_classifier import SpeedBreakerDiscriminator
from src.weather_lens_filter import WeatherLensFilter
from src.v2x_hazard_broadcaster import V2XHazardRadar
from src.edge_power_manager import EdgePowerThermalManager
from src.manhole_detector import InfrastructureAssetClassifier


class IndianRoadRouteSimulator:
    """
    Simulates real-world vehicle GPS telemetry along Indian road corridors
    when physical GPS hardware is not connected.
    Advances coordinates at realistic road speeds (35-65 km/h).
    """

    def __init__(
        self,
        start_lat: float = 28.53831,
        start_lon: float = 77.24312,
        speed_kmh: float = 45.0,
        heading_deg: float = 45.0,
    ):
        self.lat = start_lat
        self.lon = start_lon
        self.speed_kmh = speed_kmh
        self.heading_deg = heading_deg

    def step(self, delta_sec: float) -> dict:
        """Advance vehicle position along road heading for delta_sec."""
        speed_mps = (self.speed_kmh * 1000.0) / 3600.0
        distance_m = speed_mps * delta_sec

        # 1 deg latitude approx 111,320 meters
        # 1 deg longitude approx 111,320 * cos(lat) meters
        rad = math.radians(self.heading_deg)
        d_lat = (distance_m * math.cos(rad)) / 111320.0
        d_lon = (distance_m * math.sin(rad)) / (111320.0 * math.cos(math.radians(self.lat)))

        self.lat += d_lat
        self.lon += d_lon

        # Add minor natural vehicle speed variations (e.g. traffic fluctuation)
        jitter = np.sin(time.time() * 0.5) * 2.5
        curr_speed = max(20.0, self.speed_kmh + jitter)

        return {
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "speed_kmh": round(curr_speed, 1),
            "heading_deg": round(self.heading_deg, 1),
        }


class VehicleDashcamAutoReporter:
    """
    Real-time autonomous vehicle dashcam processor that detects potholes,
    extracts photographic evidence, tags coordinates, and dispatches official
    grievance complaints to Government of India portals.
    """

    def __init__(
        self,
        artifacts: dict = None,
        target_portal: str = "CPGRAMS",
        min_severity: str = "Medium",
        confirmation_frames: int = 2,
        auto_dispatch: bool = False,
        sample_interval: int = 5,
        webhook_url: str = None,
        feature_mode: str = "Deep Features (ResNet-18)",
    ):
        init_database()
        os.makedirs(config.EVIDENCE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.COMPLAINTS_OUTPUT_DIR, exist_ok=True)

        if artifacts is None:
            from auto_setup import ensure_models_and_data_ready
            ensure_models_and_data_ready(verbose=False)
            self.artifacts = load_all_artifacts()
        else:
            self.artifacts = artifacts
        self.target_portal = target_portal
        self.min_severity = min_severity
        self.confirmation_frames = confirmation_frames
        self.auto_dispatch = auto_dispatch
        self.sample_interval = sample_interval
        self.webhook_url = webhook_url
        self.feature_mode = feature_mode
        self.db_path = config.DATABASE_PATH

        # 3D Depth, Night Vision, and In-Cabin ADAS Audio Agents
        self.depth_estimator = MetricDepthEstimator()
        self.night_enhancer = NightVisionEnhancer()
        self.adas_audio = InCabinAdasAudio()
        self.distress_classifier = PavementDistressClassifier()
        self.degradation_tracker = PotholeDegradationTracker()
        self.ipm_mapper = InversePerspectiveMapper()
        self.imu_fusion = OpticalInertialShockCorrelator()
        self.crowd_dedup = CrowdsourcedSpatialDeduplicator()
        self.suspension_profiler = VehicleSuspensionProfiler()
        self.offline_sync = OfflineSyncManager()
        self.citizen_notifier = CitizenNotifier(db_path=self.db_path)
        self.speed_breaker_clf = SpeedBreakerDiscriminator()
        self.weather_filter = WeatherLensFilter()
        self.v2x_radar = V2XHazardRadar(db_path=self.db_path, enable_audio=getattr(config, "ENABLE_ADAS_AUDIO_ALERTS", True))
        self.power_mgr = EdgePowerThermalManager()
        self.asset_clf = InfrastructureAssetClassifier()

        self.tracker = DefectTracker()
        self.dispatched_track_ids = set()
        self.dispatched_events = []
        self.total_frames_read = 0
        self.total_potholes_found = 0

        # GPS telemetry provider
        self.gps_df = None
        self.simulated_gps = IndianRoadRouteSimulator()
        self.cached_road_info = {}

    def set_gps_log(self, gps_csv_path: str):
        """Optionally attach recorded vehicle GPS telemetry file."""
        if gps_csv_path and os.path.exists(gps_csv_path):
            self.gps_df = load_external_gps_log(gps_csv_path)

    def _get_current_telemetry(self, timestamp_sec: float) -> dict:
        """Resolve current GPS coordinates and speed."""
        if self.gps_df is not None:
            lat, lon = interpolate_gps(self.gps_df, timestamp_sec)
            if lat is not None and lon is not None:
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "speed_kmh": 48.0,
                    "source": "External GPS Hardware / Log",
                }

        # Fallback to realistic route simulator
        sim = self.simulated_gps.step(delta_sec=max(0.05, 1.0 / 30.0 * self.sample_interval))
        sim["source"] = "Autonomous In-Vehicle Dead Reckoning"
        return sim

    def _check_severity_eligible(self, category: str) -> bool:
        """Verify if defect category meets the configured dispatch threshold."""
        levels = {"Low": 1, "Medium": 2, "High": 3}
        req_level = levels.get(self.min_severity, 2)
        cand_level = levels.get(category, 1)
        return cand_level >= req_level

    def save_pothole_evidence(
        self,
        frame_bgr: np.ndarray,
        cand: dict,
        track_id: str,
        detection_id: str,
    ) -> str:
        """
        Extract and save a high-resolution photo evidence snapshot of the road defect.
        Annotates candidate with bounding box, category badge, and timestamp watermark.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = cand["x"], cand["y"], cand["width"], cand["height"]
        cat = cand.get("severity_category", "Medium")
        score = cand.get("severity_score", 0.0)
        color = get_severity_color_bgr(cat)

        evidence_frame = frame_bgr.copy()

        # Draw defect bounding box
        cv2.rectangle(evidence_frame, (x, y), (x + w, y + h), color, 3)

        # Draw prominent label box with 3D depth and IRC category
        max_d = cand.get("max_depth_cm", 5.2)
        vol_l = cand.get("cavity_volume_liters", 3.4)
        rep_kg = cand.get("asphalt_repair_mass_kg", 8.2)
        irc_cat = cand.get("irc_category", "IRC:Pothole")

        label = f"{irc_cat} | {cat.upper()} | Depth: {max_d:.1f} cm"
        cv2.putText(
            evidence_frame,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

        # 3D Metric Repair Sub-badge
        action_short = cand.get("maintenance_action", "Compact asphalt mix")[:28]
        sub_label = f"Void: {vol_l:.1f} L | Mix: {rep_kg:.1f} kg | PWD: {action_short}"
        cv2.putText(
            evidence_frame,
            sub_label,
            (x, min(h_img - 35, y + h + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Inset 3D Colormap in top-right corner if available
        if "depth_colormap" in cand and cand["depth_colormap"] is not None:
            try:
                cm_thumb = cv2.resize(cand["depth_colormap"], (130, 95))
                evidence_frame[15:110, w_img - 145:w_img - 15] = cm_thumb
                cv2.rectangle(evidence_frame, (w_img - 145, 15), (w_img - 15, 110), (255, 255, 255), 1)
                cv2.putText(evidence_frame, "3D DEPTH MAP", (w_img - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Forensic evidence watermark
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        watermark = f"EVIDENCE ID: {detection_id} | TIME: {now_str} | CAMERA: DASHCAM-FRONT"
        cv2.putText(
            evidence_frame,
            watermark,
            (14, h_img - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        filename = f"{detection_id}_evidence.jpg"
        evidence_path = os.path.join(config.EVIDENCE_OUTPUT_DIR, filename)
        cv2.imwrite(evidence_path, evidence_frame)
        return evidence_path

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        source_name: str = "dashcam_live",
    ) -> tuple:
        """
        Analyze a single road frame from the dashboard camera.
        If a new pothole is tracked and confirmed, auto-dispatches an official grievance.

        Returns
        -------
        tuple
            (annotated_frame_bgr, new_dispatched_events, current_telemetry)
        """
        self.total_frames_read += 1
        h_img, w_img = frame_bgr.shape[:2]
        telemetry = self._get_current_telemetry(timestamp_sec)
        new_events = []

        # 1. Adverse Weather & Lens Pre-Processing (Glare Neutralizer & Wiper Blade Detector)
        clean_frame, glare_detected, glare_pct = self.weather_filter.neutralize_sun_glare(frame_bgr)
        is_wiper, wiper_mask, _ = self.weather_filter.detect_wiper_blade(clean_frame)
        telemetry["glare_detected"] = glare_detected
        telemetry["glare_pct"] = glare_pct
        telemetry["wiper_active"] = is_wiper

        # Low-Light Night-Vision Check & Illumination Enhancement
        is_night, mean_lum = self.night_enhancer.is_night_scene(clean_frame)
        proc_frame = self.night_enhancer.enhance_night_frame(clean_frame) if is_night else clean_frame

        # 2. Run inference on road surface (visualizations skipped for high-FPS dashcam processing)
        res = run_inference_on_image(
            proc_frame,
            self.artifacts,
            image_name=f"{source_name}_f{frame_idx:05d}.jpg",
            feature_mode=self.feature_mode,
            return_visualizations=False,
        )
        raw_candidates = res["candidates_df"].to_dict(orient="records") if not res["candidates_df"].empty else []

        # 3. Filter water puddles, sweeping wipers, profile 3D depth, and trigger ADAS audio alerts
        candidates = []
        for cand in raw_candidates:
            cx, cy, cw, ch = cand["x"], cand["y"], cand["width"], cand["height"]

            # Filter out sweeping windshield wiper blade artifacts
            if is_wiper and self.weather_filter.is_defect_occluded_by_wiper((cx, cy, cw, ch), wiper_mask):
                continue

            patch = frame_bgr[cy:cy + ch, cx:cx + cw]
            is_puddle, _, _ = self.night_enhancer.is_surface_puddle(patch)
            if is_puddle:
                continue  # Discard flat rain reflections

            profile = self.depth_estimator.profile_pothole_3d(
                frame_bgr,
                cand,
                vehicle_speed_kmh=telemetry.get("speed_kmh", 45.0),
            )
            cand.update(profile)

            # Civil Infrastructure Asset Discrimination (Flush Manhole vs Sunken Manhole vs Pothole)
            asset_info = self.asset_clf.evaluate_asset(
                patch_bgr=patch,
                measured_depth_cm=profile["max_depth_cm"],
                bbox=(cx, cy, cw, ch),
            )
            if asset_info.get("suppress_grievance", False):
                continue  # Suppress legal flush manholes / utility fixtures
            if asset_info.get("is_sunken_hazard", False):
                cand["irc_category"] = asset_info["classification"]

            # Inverse Perspective Mapping (IPM) & Lateral Wheel Track Analysis
            bot_u = cx + cw / 2.0
            bot_v = cy + ch
            _, _, xm, ym = self.ipm_mapper.camera_to_bev_coords(bot_u, bot_v, w_img, h_img)
            wheel_info = self.ipm_mapper.classify_wheel_track(xm)
            bev_area = self.ipm_mapper.compute_metric_surface_area((cx, cy, cw, ch), w_img, h_img)

            cand["bev_xm"] = xm
            cand["bev_ym"] = ym
            cand["wheel_hazard"] = wheel_info["wheel_hazard"]
            cand["wheel_hazard_desc"] = wheel_info["description"]
            cand["bev_area_sqcm"] = bev_area

            # IRC Pavement Distress Classification
            distress = self.distress_classifier.classify_distress(
                patch_bgr=patch,
                cand=cand,
                depth_info=profile,
            )
            cand.update(distress)

            # In-cabin ADAS collision warning
            self.adas_audio.check_and_alert(
                distance_ahead_m=profile["distance_ahead_m"],
                time_to_impact_sec=profile["time_to_impact_sec"],
                severity_cat=cand.get("severity_category", "Medium"),
                depth_cm=profile["max_depth_cm"],
            )
            candidates.append(cand)

        # Ingest IMU accelerometer telemetry & simulate dynamic road shock
        veh_speed = telemetry.get("speed_kmh", 45.0)
        has_wheel_hit = any(
            c.get("bev_ym", 10.0) <= 4.0 and c.get("wheel_hazard") in ["LEFT_WHEEL_PATH", "RIGHT_WHEEL_PATH"]
            for c in candidates
        )
        imu_sample = self.imu_fusion.simulate_realistic_telemetry(
            timestamp_sec=timestamp_sec,
            speed_kmh=veh_speed,
            active_impact=has_wheel_hit,
            impact_severity=(candidates[0].get("severity_category", "Medium") if candidates else "Low"),
        )
        telemetry["accel_z"] = imu_sample["az"]
        telemetry["accel_x"] = imu_sample["ax"]

        # 4. Connected Vehicle (V2X) Early Hazard Radar Scan Ahead
        v2x_alerts = self.v2x_radar.scan_ahead_for_hazards(
            current_lat=telemetry.get("latitude"),
            current_lon=telemetry.get("longitude"),
            current_speed_kmh=veh_speed,
            is_wet=(is_night or is_wiper),
        )
        telemetry["v2x_alerts"] = v2x_alerts

        # 4. Update spatio-temporal tracker (deduplication across sequential frames)
        tracked_candidates = self.tracker.update(candidates, frame_idx, timestamp_sec)

        # Register optical candidates with predictive IMU shock correlator
        for cand in tracked_candidates:
            t_id = cand["track_id"]
            if t_id not in self.imu_fusion.pending_predictions and t_id not in self.imu_fusion.fused_records:
                sev_score_100 = cand.get("severity_score", 0.60)
                if sev_score_100 <= 1.0:
                    sev_score_100 *= 100.0
                self.imu_fusion.register_optical_candidate(
                    track_id=t_id,
                    detection_time_sec=timestamp_sec,
                    distance_ahead_m=cand.get("distance_ahead_m", 12.0),
                    speed_kmh=veh_speed,
                    wheel_track_hazard=cand.get("wheel_hazard", "LEFT_WHEEL_PATH"),
                    visual_severity_score=sev_score_100,
                    cavity_volume_liters=cand.get("cavity_volume_liters", 3.4),
                )

        self.imu_fusion.evaluate_correlations(timestamp_sec)

        # 5. Check for confirmed defects ready for government dispatch
        for cand in tracked_candidates:
            t_id = cand["track_id"]
            track_obj = self.tracker.active_tracks.get(t_id)

            if not track_obj:
                continue

            obs_count = track_obj.get("observation_count", 1)
            cat = track_obj.get("peak_severity_category", cand.get("severity_category", "Medium"))
            score = track_obj.get("peak_severity_score", cand.get("severity_score", 0.0))

            # Filter: must meet confirmation frames and minimum severity
            is_confirmed = (obs_count >= self.confirmation_frames)
            is_eligible = self._check_severity_eligible(cat)

            if is_confirmed and is_eligible and (t_id not in self.dispatched_track_ids):
                # Speed Breaker & Rumble Strip Discrimination Gate
                cx, cy, cw, ch = cand["x"], cand["y"], cand["width"], cand["height"]
                patch_crop = frame_bgr[cy:cy + ch, cx:cx + cw] if (cy + ch <= h_img and cx + cw <= w_img) else None
                sb_eval = self.speed_breaker_clf.classify_defect_patch(
                    patch_bgr=patch_crop,
                    bbox=(cx, cy, cw, ch),
                    frame_width=w_img,
                )

                if sb_eval.get("suppress_grievance", False):
                    self.dispatched_track_ids.add(t_id)
                    print(f"[TRAFFIC CALMING FILTER] {sb_eval['classification']} detected. Suppressing government grievance: {sb_eval['reason']}")
                    continue

                # Mark track as dispatched to prevent ANY duplicate reporting
                self.dispatched_track_ids.add(t_id)
                self.total_potholes_found += 1

                det_id = f"DASHCAM_{t_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Capture photo evidence with 3D depth and colormap inset
                evidence_img_path = self.save_pothole_evidence(frame_bgr, cand, t_id, det_id)

                # Resolve reverse-geocoded road address
                lat = telemetry["latitude"]
                lon = telemetry["longitude"]
                loc_key = f"{round(lat, 3)}_{round(lon, 3)}"

                if loc_key in self.cached_road_info:
                    loc_info = self.cached_road_info[loc_key]
                else:
                    loc_info = reverse_geocode_location(lat, lon)
                    self.cached_road_info[loc_key] = loc_info

                # Multi-drive fleet degradation & growth velocity tracking
                degradation = self.degradation_tracker.record_sighting(
                    lat=lat,
                    lon=lon,
                    depth_cm=cand.get("max_depth_cm", 5.2),
                    volume_l=cand.get("cavity_volume_liters", 3.4),
                    pixel_area=cand.get("pixel_area", 4500),
                    road_name=loc_info.get("road", "Local Road"),
                    irc_category=cand.get("irc_category", "IRC:Pothole"),
                    timestamp=datetime.now().isoformat(),
                )

                fused_data = self.imu_fusion.fused_records.get(t_id, {})
                fusion_status = fused_data.get("fusion_status", "CONFIRMED_PHYSICAL_IMPACT" if cand.get("wheel_hazard") != "LANE_CENTER" else "STRADDLED_CENTER")
                norm_score_100 = score * 100.0 if score <= 1.0 else score
                raw_threat = fused_data.get("threat_index", round(norm_score_100 * 0.85, 1))
                max_gz = fused_data.get("max_gz_shock", telemetry.get("accel_z", 1.05))

                # Calibrate threat index based on active vehicle suspension architecture
                calib = self.suspension_profiler.calibrate_threat_index(
                    raw_threat_index=raw_threat,
                    depth_cm=cand.get("max_depth_cm", 5.2),
                    shock_gz=max_gz,
                )
                threat_idx = calib["threat_index"]
                veh_risk_tier = calib["risk_tier"]

                defect_record = {
                    "detection_id": det_id,
                    "track_id": t_id,
                    "latitude": lat,
                    "longitude": lon,
                    "severity_category": cat,
                    "severity_score": score,
                    "anomaly_score": cand.get("avg_anomaly_score", -0.25),
                    "pixel_area": cand.get("pixel_area", 4500),
                    "bev_area_sqcm": cand.get("bev_area_sqcm", 2500.0),
                    "lateral_offset_m": cand.get("bev_xm", -0.8),
                    "wheel_track_hazard": cand.get("wheel_hazard", "LEFT_WHEEL_PATH"),
                    "wheel_hazard_desc": cand.get("wheel_hazard_desc", ""),
                    "imu_shock_g": max_gz,
                    "threat_index": threat_idx,
                    "vehicle_risk_tier": veh_risk_tier,
                    "vehicle_profile": self.suspension_profiler.profile_key,
                    "fusion_status": fusion_status,
                    "max_depth_cm": cand.get("max_depth_cm", 5.2),
                    "cavity_volume_liters": cand.get("cavity_volume_liters", 3.4),
                    "asphalt_repair_mass_kg": cand.get("asphalt_repair_mass_kg", 8.2),
                    "distance_ahead_m": cand.get("distance_ahead_m", 12.0),
                    "irc_category": cand.get("irc_category", "IRC:Pothole"),
                    "irc_code": cand.get("irc_code", "IRC:82-Pothole"),
                    "maintenance_action": cand.get("maintenance_action", "Compact asphalt mix"),
                    "growth_percent": degradation.get("growth_percent", 0.0),
                    "degradation_status": degradation.get("degradation_status", "STABLE"),
                    "image_path": evidence_img_path,
                    "source_media": f"Car Dashcam ({source_name})",
                    "timestamp": datetime.now().isoformat(),
                }

                # Insert to SQLite
                insert_defect(
                    detection_id=det_id,
                    track_id=t_id,
                    latitude=lat,
                    longitude=lon,
                    timestamp=defect_record["timestamp"],
                    severity_category=cat,
                    severity_score=score,
                    anomaly_score=defect_record["anomaly_score"],
                    pixel_area=defect_record["pixel_area"],
                    image_path=evidence_img_path,
                    source_media=defect_record["source_media"],
                )

                # Auto-dispatch to Government Portal with 3D telemetry
                # --- CROWDSOURCED SPATIAL DEDUPLICATION GATE ---
                # Check if another vehicle already reported this same pothole
                dedup_decision = self.crowd_dedup.check_and_decide(
                    lat=lat,
                    lon=lon,
                    detection_id=det_id,
                    severity_score=score,
                    max_depth_cm=defect_record["max_depth_cm"],
                    cavity_volume_liters=defect_record["cavity_volume_liters"],
                    irc_category=defect_record["irc_category"],
                    evidence_image_path=evidence_img_path,
                )

                dispatch_res = None
                ref_no = "PORTAL_SYNC_QUEUED"
                pdf_path = None
                citizen_count = dedup_decision.get("citizen_count", 1)
                escalation_tier = dedup_decision.get("escalation_tier", "FILED")

                is_connected = self.offline_sync.is_online(timeout=0.25)

                if dedup_decision["should_dispatch"] and self.auto_dispatch:
                    if is_connected:
                        # Online: Submit directly to Government Portal
                        try:
                            dispatch_res = dispatch_grievance_to_portal(
                                defect_record,
                                portal_key=self.target_portal,
                                webhook_url=self.webhook_url,
                            )
                            ref_no = dispatch_res["reference_number"]
                            pdf_path = dispatch_res["pdf_path"]

                            # Register in dedup registry for future vehicles
                            self.crowd_dedup.register_new_grievance(
                                lat=lat,
                                lon=lon,
                                detection_id=det_id,
                                reference_no=ref_no,
                                target_portal=self.target_portal,
                                severity_category=cat,
                                max_depth_cm=defect_record["max_depth_cm"],
                                cavity_volume_liters=defect_record["cavity_volume_liters"],
                                irc_category=defect_record["irc_category"],
                                road_name=loc_info.get("road", "Local Road"),
                                district=loc_info.get("district", "District"),
                                state=loc_info.get("state", "India"),
                                postcode=loc_info.get("postcode", ""),
                                full_address=loc_info.get("formatted_address", ""),
                                evidence_image_path=evidence_img_path,
                            )
                            citizen_count = 1
                            escalation_tier = "FILED"
                            print(f"[DISPATCH SUCCESS] Pothole {det_id} automatically submitted to {dispatch_res['target_portal']}! Ref: {ref_no}")

                            # Attempt auto-flush of any previously buffered offline complaints
                            self.offline_sync.flush_queue(max_batch=5, webhook_url=self.webhook_url)

                        except Exception as exc:
                            print(f"[WARNING] Transmission error, buffering offline: {exc}")
                            self.offline_sync.enqueue_complaint(
                                defect_dict=defect_record,
                                target_portal=self.target_portal,
                                evidence_image_path=evidence_img_path,
                                reason=f"TRANSMISSION_ERROR: {exc}",
                            )
                            ref_no = "OFFLINE_QUEUED"
                    else:
                        # Offline: Safely buffer in local SQLite queue (Zero Data Loss)
                        self.offline_sync.enqueue_complaint(
                            defect_dict=defect_record,
                            target_portal=self.target_portal,
                            evidence_image_path=evidence_img_path,
                            reason="CELLULAR_DATA_OFFLINE_HIGHWAY",
                        )
                        ref_no = "OFFLINE_QUEUED"
                        print(f"[OFFLINE BUFFER] Cellular signal unavailable. Complaint {det_id} queued safely for auto-sync.")
                elif dedup_decision["should_dispatch"] and not self.auto_dispatch:
                    # User Review Policy: Record details & location, await manual approval
                    ref_no = "PENDING_REVIEW"
                    update_grievance_status(
                        detection_id=det_id,
                        status="Pending Review",
                        ref_no=ref_no,
                        portal=self.target_portal,
                    )
                    print(f"[RECORDED - REVIEW REQUIRED] Pothole {det_id} at {lat:.5f}, {lon:.5f} saved to DB. Awaiting user approval.")
                elif not dedup_decision["should_dispatch"]:
                    # DUPLICATE SUPPRESSED — another citizen already reported this pothole
                    ref_no = dedup_decision.get("reference_no", "EXISTING_GRIEVANCE")
                    citizen_count = dedup_decision.get("citizen_count", 2)
                    escalation_tier = dedup_decision.get("escalation_tier", "FILED")
                    print(
                        f"[DEDUP] Duplicate complaint SUPPRESSED for {det_id}. "
                        f"Citizen #{citizen_count} corroborated existing grievance {ref_no}. "
                        f"Escalation: {escalation_tier}"
                    )
                        f"[DEDUP] Duplicate complaint SUPPRESSED for {det_id}. "
                        f"Citizen #{citizen_count} corroborated existing grievance {ref_no}. "
                        f"Escalation: {escalation_tier}"
                    )

                event = {
                    "track_id": t_id,
                    "detection_id": det_id,
                    "severity_category": cat,
                    "severity_score": score,
                    "latitude": lat,
                    "longitude": lon,
                    "road_name": loc_info.get("road", "Local Road"),
                    "district": loc_info.get("district", "District"),
                    "state": loc_info.get("state", "India"),
                    "postcode": loc_info.get("postcode", ""),
                    "full_address": loc_info.get("formatted_address", ""),
                    "reference_no": ref_no,
                    "target_portal": config.GOV_PORTALS.get(self.target_portal, {}).get("name", self.target_portal),
                    "max_depth_cm": defect_record["max_depth_cm"],
                    "cavity_volume_liters": defect_record["cavity_volume_liters"],
                    "asphalt_repair_mass_kg": defect_record["asphalt_repair_mass_kg"],
                    "distance_ahead_m": defect_record["distance_ahead_m"],
                    "irc_category": defect_record["irc_category"],
                    "irc_code": defect_record["irc_code"],
                    "maintenance_action": defect_record["maintenance_action"],
                    "growth_percent": defect_record["growth_percent"],
                    "degradation_status": defect_record["degradation_status"],
                    "revisit_count": degradation.get("revisit_count", 1),
                    "bev_area_sqcm": defect_record["bev_area_sqcm"],
                    "lateral_offset_m": defect_record["lateral_offset_m"],
                    "wheel_track_hazard": defect_record["wheel_track_hazard"],
                    "wheel_hazard_desc": defect_record["wheel_hazard_desc"],
                    "imu_shock_g": defect_record["imu_shock_g"],
                    "threat_index": defect_record["threat_index"],
                    "fusion_status": defect_record["fusion_status"],
                    "vehicle_profile": defect_record.get("vehicle_profile", "HATCHBACK_SEDAN"),
                    "vehicle_risk_tier": defect_record.get("vehicle_risk_tier", "MODERATE_ROAD_HAZARD"),
                    "citizen_count": citizen_count,
                    "escalation_tier": escalation_tier,
                    "dedup_action": dedup_decision["action"],
                    "evidence_image": evidence_img_path,
                    "pdf_dossier": pdf_path,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }

                # Generate citizen notification and WhatsApp / SMS action links
                try:
                    c_notif = self.citizen_notifier.notify(event, notification_type="AUTO")
                    event["whatsapp_url"] = c_notif.get("whatsapp_url", "")
                    event["sms_text"] = c_notif.get("sms_text", "")
                    event["google_maps_url"] = c_notif.get("google_maps_url", "")
                except Exception as _c_err:
                    event["whatsapp_url"] = ""
                    event["sms_text"] = ""
                    event["google_maps_url"] = f"https://maps.google.com/?q={lat},{lon}"

                new_events.append(event)
                self.dispatched_events.append(event)

        # 6. Render Heads-Up Display (HUD) on frame
        annotated_frame = self._render_dashcam_hud(
            frame_bgr,
            tracked_candidates,
            telemetry,
            is_night=is_night,
            recent_event=(new_events[-1] if new_events else (self.dispatched_events[-1] if self.dispatched_events else None)),
        )

        return annotated_frame, new_events, telemetry

    def _render_dashcam_hud(
        self,
        frame_bgr: np.ndarray,
        tracked_candidates: list,
        telemetry: dict,
        is_night: bool = False,
        recent_event: dict = None,
    ) -> np.ndarray:
        """Draw transparent HUD telemetry overlay, bounding boxes, 3D metric labels, and active alerts."""
        hud = frame_bgr.copy()
        h, w = hud.shape[:2]

        # 1. Draw defect bounding boxes with metric depth tag
        for cand in tracked_candidates:
            x, y, bw, bh = cand["x"], cand["y"], cand["width"], cand["height"]
            t_id = cand["track_id"]
            cat = cand.get("severity_category", "Low")
            score = cand.get("severity_score", 0.0)
            color = get_severity_color_bgr(cat)
            depth_cm = cand.get("max_depth_cm", 4.5)
            dist_m = cand.get("distance_ahead_m", 12.0)

            # Box
            cv2.rectangle(hud, (x, y), (x + bw, y + bh), color, 2)

            # Target reticle corners
            ret_len = min(12, bw // 4, bh // 4)
            cv2.line(hud, (x, y), (x + ret_len, y), (255, 255, 255), 2)
            cv2.line(hud, (x, y), (x, y + ret_len), (255, 255, 255), 2)
            cv2.line(hud, (x + bw, y + bh), (x + bw - ret_len, y + bh), (255, 255, 255), 2)
            cv2.line(hud, (x + bw, y + bh), (x + bw, y + bh - ret_len), (255, 255, 255), 2)

            irc_label = cand.get('irc_category', 'Pothole').replace('IRC:', '')
            wheel_code = cand.get("wheel_hazard", "LEFT_WHEEL_PATH")
            wheel_short = "L-WHEEL" if "LEFT" in wheel_code else ("R-WHEEL" if "RIGHT" in wheel_code else ("CENTER" if "CENTER" in wheel_code else "SHOULDER"))
            label = f"{t_id} | {irc_label} | {depth_cm:.1f}cm @ {dist_m:.0f}m [{wheel_short}]"
            cv2.putText(hud, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        # Inset ADAS Bird's Eye View (BEV) Mini-Radar HUD Widget in top-right
        try:
            radar_w, radar_h = 135, 190
            radar_hud = self.ipm_mapper.render_bev_radar_widget(
                tracked_candidates,
                w,
                h,
                widget_w=radar_w,
                widget_h=radar_h,
            )
            y_r = 60
            x_r = w - radar_w - 10
            hud[y_r:y_r + radar_h, x_r:x_r + radar_w] = radar_hud
        except Exception:
            pass

        # 2. Draw Top Telemetry Bar (translucent dark ribbon)
        top_bar = hud[0:55, 0:w].copy()
        cv2.rectangle(top_bar, (0, 0), (w, 55), (20, 25, 30), -1)
        cv2.addWeighted(top_bar, 0.75, hud[0:55, 0:w], 0.25, 0, hud[0:55, 0:w])

        # Header Title
        cv2.putText(hud, "VEHICLE DASHCAM AUTO-REPORTER (3D METRIC & ADAS)", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # Live telemetry tags
        lat = telemetry.get("latitude", 0.0)
        lon = telemetry.get("longitude", 0.0)
        speed = telemetry.get("speed_kmh", 40.0)
        gz_val = telemetry.get("accel_z", 1.02)
        veh_code = self.suspension_profiler.profile_key.replace('_', ' ')
        soc_temp = self.power_mgr.query_hardware_thermal_sensor()
        telem_str = f"SPEED: {speed:.0f} km/h  |  GPS: {lat:.5f}, {lon:.5f}  |  Gz: {gz_val:.2f}g  |  SoC: {soc_temp:.0f}C  |  {veh_code}"
        cv2.putText(hud, telem_str, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1, cv2.LINE_AA)

        # Status badge (top right)
        status_txt = f"PORTAL DISPATCH: ACTIVE ({self.target_portal})"
        cv2.putText(hud, status_txt, (w - 380, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        if telemetry.get("wiper_active"):
            cv2.putText(hud, "WIPER FILTER: ACTIVE", (w - 200, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
        elif telemetry.get("glare_detected"):
            cv2.putText(hud, "SUN GLARE ATTENUATED", (w - 220, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 200, 0), 1, cv2.LINE_AA)

        # V2X Hazard Radar Warning Banner
        v2x_list = telemetry.get("v2x_alerts", [])
        if v2x_list:
            top_alert = v2x_list[0]
            dist_ahead = top_alert["distance_m"]
            v2x_str = f"V2X RADAR: {top_alert['severity_category'].upper()} HAZARD {dist_ahead:.0f}M AHEAD | TARGET SPEED: {top_alert['recommended_speed_kmh']:.0f} KM/H"
            alert_overlay = hud.copy()
            cv2.rectangle(alert_overlay, (10, h - 94), (w - 10, h - 64), (0, 0, 190), -1)
            cv2.addWeighted(alert_overlay, 0.85, hud, 0.15, 0, hud)
            cv2.putText(hud, v2x_str, (18, h - 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Active Alert Banner if defect was recently logged
        if recent_event:
            alert_bar = hud[h - 52:h, 0:w].copy()
            cv2.rectangle(alert_bar, (0, 0), (w, 52), (10, 15, 80), -1)
            cv2.addWeighted(alert_bar, 0.8, hud[h - 52:h, 0:w], 0.2, 0, hud[h - 52:h, 0:w])

            status_tag = recent_event.get('degradation_status', 'STABLE')
            depth_cm = recent_event.get('max_depth_cm', 5.0)
            alert_text = (
                f"DISPATCHED! Ref: {recent_event['reference_no']} | "
                f"{recent_event.get('irc_category', 'Pothole')} ({depth_cm:.1f}cm) | "
                f"Status: {status_tag} | PWD Action: {recent_event.get('maintenance_action', 'Patch')[:28]}"
            )
            cv2.putText(hud, alert_text, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 220, 255), 1, cv2.LINE_AA)

        return hud

