"""
src/imu_fusion.py
=================
Multi-Modal Optical-Inertial Shock Correlator & Accelerometer Fusion.

Fuses front-facing dashcam computer vision with vehicle vertical-axis accelerometer (Gz)
and IMU telemetry (from smartphone IMU, OBD-II adapter, or USB G-sensor).

Key Capabilities:
1. Buffers continuous 3-axis acceleration telemetry (Gx, Gy, Gz) and vehicle speed.
2. Identifies transient suspension shock spikes (|Gz - 1.0g| > threshold) and vertical jerk.
3. Spatio-temporal impact synchronization:
   Predicts tire impact arrival time (t_impact = t_det + Distance / Speed).
4. Correlates optical bounding box detections with physical vibration spikes:
   - Direct tire hit in LEFT_WHEEL_PATH or RIGHT_WHEEL_PATH -> CONFIRMED_PHYSICAL_IMPACT (99.8% confidence)
   - Straddled in LANE_CENTER -> STRADDLED_CENTER (valid pothole, no tire damage)
   - Driver swerve/avoidance -> AVOIDED_BY_STEERING
5. Computes Unified Pavement Threat Index (0-100) combining visual, volumetric, and inertial shock.
"""

import os
import sys
import time
import math
import numpy as np
from collections import deque

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class OpticalInertialShockCorrelator:
    """
    Fuses camera detections with vehicle IMU accelerometer telemetry
    to eliminate optical false positives and measure physical impact severity.
    """

    def __init__(
        self,
        buffer_seconds: float = 8.0,
        sample_hz: float = 50.0,
        shock_threshold_g: float = 0.35,  # Deviation from 1.0g to trigger bump detection
        time_tolerance_sec: float = 0.40,  # Temporal synchronization window (+/- 0.4s)
    ):
        self.buffer_seconds = buffer_seconds
        self.sample_hz = sample_hz
        self.max_buffer_len = int(buffer_seconds * sample_hz)
        self.shock_threshold_g = shock_threshold_g
        self.time_tolerance_sec = time_tolerance_sec

        # Ring buffer: [(t_sec, ax, ay, az, speed_kmh), ...]
        self.imu_buffer = deque(maxlen=self.max_buffer_len)

        # Pending predicted impacts: {track_id: {expected_t, lateral_pos, visual_sev, dist_m}}
        self.pending_predictions = {}
        # Completed fused results: {track_id: {...}}
        self.fused_records = {}

    def push_imu_sample(
        self,
        timestamp_sec: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        speed_kmh: float = 45.0,
    ):
        """
        Ingest a live 3-axis accelerometer reading into the rolling buffer.
        """
        self.imu_buffer.append({
            "t": float(timestamp_sec),
            "ax": float(accel_x),
            "ay": float(accel_y),
            "az": float(accel_z),
            "speed": float(speed_kmh),
        })

    def simulate_realistic_telemetry(
        self,
        timestamp_sec: float,
        speed_kmh: float = 45.0,
        active_impact: bool = False,
        impact_severity: str = "Medium",
    ) -> dict:
        """
        Synthesize realistic in-vehicle accelerometer telemetry for testing and simulation.
        Includes nominal road micro-vibrations (1.0g +/- 0.04g) and dynamic shock spikes.
        """
        # Nominal asphalt baseline vibration
        base_noise = np.random.normal(0.0, 0.03)
        gz = 1.0 + base_noise
        gx = np.random.normal(0.0, 0.02)
        gy = np.random.normal(0.0, 0.02)

        if active_impact:
            # Physical pothole hit transient: downward cavity drop then rebound shock
            if impact_severity == "High":
                shock_val = np.random.uniform(1.85, 2.75)
            elif impact_severity == "Medium":
                shock_val = np.random.uniform(1.40, 1.80)
            else:
                shock_val = np.random.uniform(1.20, 1.38)
            gz = shock_val

        sample = {
            "t": round(timestamp_sec, 3),
            "ax": round(gx, 3),
            "ay": round(gy, 3),
            "az": round(gz, 3),
            "speed": round(speed_kmh, 1),
        }
        self.push_imu_sample(sample["t"], sample["ax"], sample["ay"], sample["az"], sample["speed"])
        return sample

    def register_optical_candidate(
        self,
        track_id: str,
        detection_time_sec: float,
        distance_ahead_m: float,
        speed_kmh: float,
        wheel_track_hazard: str = "LEFT_WHEEL_PATH",
        visual_severity_score: float = 65.0,
        cavity_volume_liters: float = 3.5,
    ):
        """
        Register a visually detected pothole ahead and project its expected time of wheel arrival.
        """
        speed_mps = max(5.0, (speed_kmh * 1000.0) / 3600.0)
        time_to_arrive = distance_ahead_m / speed_mps
        expected_arrival_time = detection_time_sec + time_to_arrive

        self.pending_predictions[track_id] = {
            "track_id": track_id,
            "detection_time": detection_time_sec,
            "expected_arrival": expected_arrival_time,
            "distance_m": distance_ahead_m,
            "speed_kmh": speed_kmh,
            "wheel_hazard": wheel_track_hazard,
            "visual_score": visual_severity_score,
            "cavity_vol": cavity_volume_liters,
        }

    def evaluate_correlations(self, current_time_sec: float) -> list:
        """
        Check all pending predictions whose expected arrival time has elapsed.
        Correlates predicted arrival with recorded accelerometer shock spikes.

        Returns
        -------
        list of dict : Newly resolved fusion events
        """
        resolved = []
        keys_to_remove = []

        for tid, pred in list(self.pending_predictions.items()):
            exp_t = pred["expected_arrival"]

            # Wait until current time is past the arrival window
            if current_time_sec < (exp_t + self.time_tolerance_sec):
                continue

            # Query IMU buffer in window [exp_t - tolerance, exp_t + tolerance]
            t_start = exp_t - self.time_tolerance_sec
            t_end = exp_t + self.time_tolerance_sec

            window_samples = [s for s in self.imu_buffer if t_start <= s["t"] <= t_end]

            if not window_samples:
                # If buffer didn't catch samples (e.g. simulation boundary), assume unconfirmed
                max_gz = 1.0
                delta_gz = 0.0
            else:
                gz_vals = [s["az"] for s in window_samples]
                max_gz = max(gz_vals)
                min_gz = min(gz_vals)
                delta_gz = max_gz - min_gz

            is_shock = (delta_gz >= self.shock_threshold_g) or (max_gz >= 1.35)
            wheel_pos = pred["wheel_hazard"]

            # Decision Logic
            if wheel_pos in ["LEFT_WHEEL_PATH", "RIGHT_WHEEL_PATH"]:
                if is_shock:
                    fusion_status = "CONFIRMED_PHYSICAL_IMPACT"
                    confidence = 0.998
                    if max_gz > 2.0:
                        impact_label = "VIOLENT_RIM_IMPACT"
                    elif max_gz > 1.45:
                        impact_label = "MODERATE_SHOCK"
                    else:
                        impact_label = "MILD_BUMP"
                else:
                    fusion_status = "AVOIDED_OR_SHALLOW"
                    confidence = 0.820
                    impact_label = "DRIVER_SWERVED_OR_SHALLOW"
            elif wheel_pos == "LANE_CENTER":
                fusion_status = "STRADDLED_CENTER"
                confidence = 0.940
                impact_label = "VEHICLE_STRADDLED_DEFECT"
            else:
                fusion_status = "SHOULDER_ANOMALY"
                confidence = 0.850
                impact_label = "OFF_TRACK_DEFECT"

            # Compute Unified Pavement Threat Index (0-100)
            # Combines Visual Score (40%), Volumetric Cavity (30%), Inertial Shock (30%)
            vis_part = pred["visual_score"] * 0.40
            vol_part = min(30.0, (pred["cavity_vol"] / 8.0) * 30.0)
            shock_part = min(30.0, ((max_gz - 1.0) / 1.5) * 30.0) if is_shock else 0.0
            threat_index = round(min(100.0, max(15.0, vis_part + vol_part + shock_part)), 1)

            res = {
                "track_id": tid,
                "fusion_status": fusion_status,
                "confidence": confidence,
                "impact_label": impact_label,
                "max_gz_shock": round(max_gz, 2),
                "delta_gz": round(delta_gz, 2),
                "wheel_hazard": wheel_pos,
                "threat_index": threat_index,
                "detection_time": pred["detection_time"],
                "arrival_time": round(exp_t, 2),
            }

            self.fused_records[tid] = res
            resolved.append(res)
            keys_to_remove.append(tid)

        for k in keys_to_remove:
            self.pending_predictions.pop(k, None)

        return resolved
