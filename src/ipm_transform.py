"""
src/ipm_transform.py
====================
Inverse Perspective Mapping (IPM) & Top-Down Bird's Eye View (BEV) Transformer.

Converts forward-facing trapezoidal dashboard camera road perspective into an
ortho-rectified metric bird's-eye grid.

Key Capabilities:
1. Calculates homography matrix mapping road plane perspective to top-down grid.
2. Ortho-rectifies road surface so distant and near potholes have true scale.
3. Maps pixel coordinates to physical metric road coordinates (X, Y) in meters.
4. Categorizes lateral lane hazard position:
   - LEFT_WHEEL_PATH   : Direct tire impact hazard (suspension damage)
   - LANE_CENTER       : Straddled between tires (low vehicle impact)
   - RIGHT_WHEEL_PATH  : Direct tire impact hazard
   - ROAD_SHOULDER     : Outside active traffic lane
5. Computes perspective-invariant true physical surface area in cm².
6. Generates an ADAS Bird's Eye View radar HUD widget for live vehicle displays.
"""

import os
import sys
import math
import cv2
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class InversePerspectiveMapper:
    """
    Inverse Perspective Mapping (IPM) engine for vehicular dashcam video.
    """

    def __init__(
        self,
        camera_height_m: float = 1.35,      # Typical car dashboard/windshield height
        camera_pitch_deg: float = -8.0,      # Forward downward tilt angle
        lane_width_m: float = 3.75,          # Standard Indian highway lane width (IRC:73)
        longitudinal_range_m: tuple = (3.0, 25.0),  # Metric lookahead range ahead of bumper
        bev_size: tuple = (400, 600),        # Width, Height of BEV canvas (px)
    ):
        self.camera_height_m = camera_height_m
        self.camera_pitch_deg = camera_pitch_deg
        self.lane_width_m = lane_width_m
        self.y_min_m, self.y_max_m = longitudinal_range_m
        self.x_half_span_m = 2.2  # Lateral span from -2.2m (left) to +2.2m (right)
        self.bev_w, self.bev_h = bev_size

        # Calibration matrices cache
        self.H_ipm = None
        self.H_inv = None
        self._cached_frame_size = None

    def _setup_matrices(self, img_w: int, img_h: int):
        """
        Compute perspective homography transformation matrices for given input dimensions.
        """
        if self._cached_frame_size == (img_w, img_h) and self.H_ipm is not None:
            return

        # Define source trapezoid on the forward road plane
        # Horizon cut at ~48% height, bottom spreads out to cover vehicle hood/lane
        top_y = int(img_h * 0.50)
        bot_y = int(img_h * 0.96)
        
        # Converging lane trapezoid in camera view
        top_left = [int(img_w * 0.36), top_y]
        top_right = [int(img_w * 0.64), top_y]
        bot_right = [int(img_w * 0.90), bot_y]
        bot_left = [int(img_w * 0.10), bot_y]

        src_pts = np.float32([top_left, top_right, bot_right, bot_left])

        # Destination rectangle in top-down bird's-eye view canvas
        # Top of canvas = y_max_m (25m ahead), Bottom of canvas = y_min_m (3m ahead)
        pad_x = int(self.bev_w * 0.12)
        dst_top_left = [pad_x, 0]
        dst_top_right = [self.bev_w - pad_x, 0]
        dst_bot_right = [self.bev_w - pad_x, self.bev_h]
        dst_bot_left = [pad_x, self.bev_h]

        dst_pts = np.float32([dst_top_left, dst_top_right, dst_bot_right, dst_bot_left])

        self.H_ipm = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self._cached_frame_size = (img_w, img_h)

    def warp_to_bev(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Warp camera perspective frame into top-down ortho-rectified Bird's Eye View.
        """
        h, w = frame_bgr.shape[:2]
        self._setup_matrices(w, h)
        bev = cv2.warpPerspective(
            frame_bgr,
            self.H_ipm,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(15, 15, 20),
        )
        return bev

    def camera_to_bev_coords(self, u: float, v: float, img_w: int, img_h: int) -> tuple:
        """
        Map a pixel coordinate (u, v) from camera image into (x_bev, y_bev) and
        metric road coordinates (x_meters, y_meters) relative to vehicle bumper center.

        Returns
        -------
        tuple : (x_bev, y_bev, x_meter, y_meter)
        """
        self._setup_matrices(img_w, img_h)
        pt = np.array([[[u, v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_ipm)[0][0]
        x_bev, y_bev = float(transformed[0]), float(transformed[1])

        # Clamp inside BEV canvas bounds for metric interpolation
        y_norm = max(0.0, min(1.0, 1.0 - (y_bev / float(self.bev_h))))
        y_meter = self.y_min_m + y_norm * (self.y_max_m - self.y_min_m)

        x_norm = max(0.0, min(1.0, x_bev / float(self.bev_w)))
        x_meter = -self.x_half_span_m + x_norm * (2.0 * self.x_half_span_m)

        return round(x_bev, 1), round(y_bev, 1), round(x_meter, 2), round(y_meter, 2)

    def classify_wheel_track(self, x_meter: float) -> dict:
        """
        Classify lateral hazard position across the traffic lane.
        Standard vehicle track width is ~1.6m (wheels at -0.8m and +0.8m).
        """
        # Wheel tire width allowance is +/- 0.35m around track line
        if -1.25 <= x_meter <= -0.45:
            return {
                "wheel_hazard": "LEFT_WHEEL_PATH",
                "severity_multiplier": 1.4,
                "description": "Direct Left Wheel Path — Severe tire blowout & suspension hazard!",
                "color_bgr": (0, 0, 255),
            }
        elif +0.45 <= x_meter <= +1.25:
            return {
                "wheel_hazard": "RIGHT_WHEEL_PATH",
                "severity_multiplier": 1.4,
                "description": "Direct Right Wheel Path — Severe tire blowout & suspension hazard!",
                "color_bgr": (0, 0, 255),
            }
        elif -0.45 < x_meter < +0.45:
            return {
                "wheel_hazard": "LANE_CENTER",
                "severity_multiplier": 0.85,
                "description": "Lane Center — Vehicle straddles defect (Tires avoid rim impact)",
                "color_bgr": (0, 220, 255),
            }
        else:
            return {
                "wheel_hazard": "ROAD_SHOULDER",
                "severity_multiplier": 0.70,
                "description": "Outside active wheel track / Road shoulder",
                "color_bgr": (0, 255, 120),
            }

    def compute_metric_surface_area(self, bbox: tuple, img_w: int, img_h: int) -> float:
        """
        Compute true physical surface area in cm² using perspective-invariant BEV mapping.
        """
        x, y, w, h = bbox
        # Clamp coordinates to road trapezoid bounds
        y_clamped = max(int(img_h * 0.50), min(int(img_h * 0.95), y))
        h_clamped = max(10, min(int(img_h * 0.96) - y_clamped, h))
        x_clamped = max(0, min(img_w - 10, x))
        w_clamped = max(10, min(img_w - x_clamped, w))

        corners = np.float32([
            [x_clamped, y_clamped],
            [x_clamped + w_clamped, y_clamped],
            [x_clamped + w_clamped, y_clamped + h_clamped],
            [x_clamped, y_clamped + h_clamped]
        ]).reshape(-1, 1, 2)

        self._setup_matrices(img_w, img_h)
        bev_corners = cv2.perspectiveTransform(corners, self.H_ipm)
        bev_area_px = cv2.contourArea(bev_corners)

        # Scale factor: total BEV area covers (2 * x_half_span_m) * (y_max_m - y_min_m) m²
        ground_area_sqm = (2.0 * self.x_half_span_m) * (self.y_max_m - self.y_min_m)
        canvas_px = float(self.bev_w * self.bev_h)
        sqm_per_px = ground_area_sqm / canvas_px

        area_sqm = bev_area_px * sqm_per_px
        area_sqcm = area_sqm * 10000.0  # 1 m² = 10,000 cm²

        # Realistic physical pothole bound (50 cm² to 25,000 cm²)
        return max(50.0, min(25000.0, round(area_sqcm, 1)))

    def render_bev_radar_widget(
        self,
        defects_list: list,
        img_w: int,
        img_h: int,
        widget_w: int = 160,
        widget_h: int = 240,
    ) -> np.ndarray:
        """
        Render a high-tech ADAS Bird's Eye View Mini-Radar HUD widget
        suitable for overlaying onto the windshield dashcam video feed.
        """
        radar = np.zeros((widget_h, widget_w, 3), dtype=np.uint8)
        radar[:] = (20, 22, 28)  # Sleek dark automotive dashboard tint

        # Lane boundary lines
        lane_left_x = int(widget_w * 0.22)
        lane_right_x = int(widget_w * 0.78)
        cv2.line(radar, (lane_left_x, 10), (lane_left_x, widget_h - 35), (80, 85, 95), 1, cv2.LINE_AA)
        cv2.line(radar, (lane_right_x, 10), (lane_right_x, widget_h - 35), (80, 85, 95), 1, cv2.LINE_AA)

        # Dashed center line
        for y_dash in range(15, widget_h - 35, 14):
            cv2.line(radar, (widget_w // 2, y_dash), (widget_w // 2, y_dash + 7), (120, 125, 135), 1)

        # Distance range arcs (5m, 10m, 15m, 20m)
        ranges = [(5, 0.80), (10, 0.60), (15, 0.40), (20, 0.20)]
        for dist_m, y_ratio in ranges:
            y_pos = int(widget_h * y_ratio)
            cv2.line(radar, (lane_left_x - 10, y_pos), (lane_right_x + 10, y_pos), (45, 50, 60), 1, cv2.LINE_AA)
            cv2.putText(radar, f"{dist_m}m", (widget_w - 28, y_pos + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 155, 165), 1, cv2.LINE_AA)

        # Vehicle Icon at bottom
        veh_w, veh_h = 32, 22
        veh_x = (widget_w - veh_w) // 2
        veh_y = widget_h - 30
        cv2.rectangle(radar, (veh_x, veh_y), (veh_x + veh_w, veh_y + veh_h), (0, 200, 255), -1)
        cv2.putText(radar, "CAR", (veh_x + 4, veh_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        # Wheel track guide lines (dotted)
        lw_x = veh_x + 3
        rw_x = veh_x + veh_w - 3
        cv2.line(radar, (lw_x, 10), (lw_x, veh_y), (40, 45, 55), 1)
        cv2.line(radar, (rw_x, 10), (rw_x, veh_y), (40, 45, 55), 1)

        # Plot detected defects as radar blips
        for d in defects_list:
            bbox = d.get("bbox")
            if not bbox:
                continue
            bx, by, bw, bh = bbox
            bot_u = bx + bw / 2.0
            bot_v = by + bh

            _, _, xm, ym = self.camera_to_bev_coords(bot_u, bot_v, img_w, img_h)

            # Map xm, ym to radar widget coordinates
            norm_y = max(0.0, min(1.0, (ym - self.y_min_m) / (self.y_max_m - self.y_min_m)))
            r_y = int((widget_h - 35) - norm_y * (widget_h - 50))

            norm_x = max(0.0, min(1.0, (xm + self.x_half_span_m) / (2.0 * self.x_half_span_m)))
            r_x = int(norm_x * widget_w)

            track_info = self.classify_wheel_track(xm)
            color = track_info["color_bgr"]

            # Blip dot
            cv2.circle(radar, (r_x, r_y), 5, color, -1)
            cv2.circle(radar, (r_x, r_y), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Distance tag
            cv2.putText(radar, f"{ym:.0f}m", (r_x + 8, r_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

        # Widget Border & Title
        cv2.rectangle(radar, (0, 0), (widget_w - 1, widget_h - 1), (0, 220, 255), 1)
        cv2.rectangle(radar, (0, 0), (widget_w - 1, 14), (35, 40, 50), -1)
        cv2.putText(radar, "BEV ROAD RADAR", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 200), 1, cv2.LINE_AA)

        return radar
