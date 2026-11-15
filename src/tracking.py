"""
src/tracking.py
================
Spatial-temporal defect tracking and association across video frames.
Prevents duplicate reporting of the same physical road defect observed across
consecutive video frames.
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.bounding_boxes import compute_iou


class DefectTracker:
    """
    IoU-based spatial-temporal tracker for road surface defect candidates.
    Maintains track persistence, updates peak severity scores, and prevents
    multiple independent records for the same physical pothole.
    """

    def __init__(self, iou_threshold: float = None, max_age: int = None):
        self.iou_threshold = iou_threshold if iou_threshold is not None else config.VIDEO_TRACKING_IOU_THRESHOLD
        self.max_age = max_age if max_age is not None else config.MAX_TRACK_AGE_FRAMES

        self.next_track_id = 1
        self.active_tracks = {}  # track_id -> dict
        self.closed_tracks = []  # list of finalized track dicts

    def update(self, frame_candidates: list, frame_idx: int, timestamp_sec: float) -> list:
        """
        Associate frame candidates with existing tracks or spawn new tracks.

        Parameters
        ----------
        frame_candidates : list of dicts
            Candidates identified in the current frame.
        frame_idx : int
            Current frame index.
        timestamp_sec : float
            Timestamp in seconds.

        Returns
        -------
        list of dicts
            Candidate records enriched with their persistent 'track_id'.
        """
        matched_candidates = []
        unmatched_candidates = list(range(len(frame_candidates)))
        unmatched_tracks = list(self.active_tracks.keys())

        # Compute IoU matching matrix
        matches = []
        for c_idx in unmatched_candidates:
            cand = frame_candidates[c_idx]
            best_iou = 0.0
            best_t_id = None

            for t_id in unmatched_tracks:
                track = self.active_tracks[t_id]
                iou = compute_iou(cand, track["last_box"])
                if iou > best_iou:
                    best_iou = iou
                    best_t_id = t_id

            if best_iou >= self.iou_threshold and best_t_id is not None:
                matches.append((c_idx, best_t_id, best_iou))

        # Sort matches by best IoU
        matches = sorted(matches, key=lambda m: m[2], reverse=True)
        used_cand = set()
        used_track = set()

        for c_idx, t_id, iou in matches:
            if c_idx in used_cand or t_id in used_track:
                continue

            used_cand.add(c_idx)
            used_track.add(t_id)

            # Update existing track
            cand = frame_candidates[c_idx]
            track = self.active_tracks[t_id]

            track["last_box"] = {k: cand[k] for k in ["x", "y", "width", "height"]}
            track["last_frame"] = frame_idx
            track["last_timestamp"] = timestamp_sec
            track["observation_count"] += 1
            track["max_anomaly_score"] = max(track["max_anomaly_score"], cand.get("max_anomaly_score", 0.0))
            track["avg_anomaly_score"] = max(track["avg_anomaly_score"], cand.get("avg_anomaly_score", 0.0))

            # Maintain peak severity
            new_sev_score = cand.get("severity_score", 0.0)
            if new_sev_score > track["peak_severity_score"]:
                track["peak_severity_score"] = new_sev_score
                track["peak_severity_category"] = cand.get("severity_category", "Low")

            cand_copy = cand.copy()
            cand_copy["track_id"] = t_id
            matched_candidates.append(cand_copy)

        # Spawn new tracks for unmatched candidates
        for c_idx in unmatched_candidates:
            if c_idx not in used_cand:
                cand = frame_candidates[c_idx]
                t_id = f"DEFECT_TRK_{self.next_track_id:04d}"
                self.next_track_id += 1

                new_track = {
                    "track_id": t_id,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "first_timestamp": timestamp_sec,
                    "last_timestamp": timestamp_sec,
                    "observation_count": 1,
                    "last_box": {k: cand[k] for k in ["x", "y", "width", "height"]},
                    "max_anomaly_score": cand.get("max_anomaly_score", 0.0),
                    "avg_anomaly_score": cand.get("avg_anomaly_score", 0.0),
                    "peak_severity_score": cand.get("severity_score", 0.0),
                    "peak_severity_category": cand.get("severity_category", "Low"),
                    "pixel_area": cand.get("pixel_area", 0),
                }

                self.active_tracks[t_id] = new_track

                cand_copy = cand.copy()
                cand_copy["track_id"] = t_id
                matched_candidates.append(cand_copy)

        # Retire stale tracks
        stale_ids = []
        for t_id, track in self.active_tracks.items():
            if frame_idx - track["last_frame"] > self.max_age:
                stale_ids.append(t_id)

        for s_id in stale_ids:
            self.closed_tracks.append(self.active_tracks.pop(s_id))

        return matched_candidates

    def finalize(self) -> list:
        """
        Flush all remaining active tracks into closed tracks list and return all unique tracks.
        """
        for t_id, track in self.active_tracks.items():
            self.closed_tracks.append(track)
        self.active_tracks = {}
        return self.closed_tracks

