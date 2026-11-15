"""
src/adas_audio.py
=================
In-Cabin Driver ADAS Audio Warning System.

Provides real-time auditory driver alerts when the vehicle approaches severe
road potholes or asphalt hazards, giving the driver time to brake or steer.
Features non-blocking asynchronous audio beeps and spoken voice warnings.
"""

import os
import sys
import time
import threading
import subprocess

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class InCabinAdasAudio:
    """
    Non-blocking in-cabin auditory collision warning agent.
    """

    def __init__(
        self,
        enabled: bool = None,
        cooldown_sec: float = None,
        warning_dist_m: float = None,
    ):
        self.enabled = enabled if enabled is not None else config.ENABLE_ADAS_AUDIO_ALERTS
        self.cooldown_sec = cooldown_sec if cooldown_sec is not None else config.ADAS_COOLDOWN_SECONDS
        self.warning_dist_m = warning_dist_m if warning_dist_m is not None else config.ADAS_WARNING_DISTANCE_METERS

        self.last_alert_time = 0.0
        self.alert_lock = threading.Lock()

    def _play_windows_beep(self, freq: int = 1200, duration_ms: int = 250):
        """Play hardware frequency tone on Windows."""
        try:
            import winsound
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass

    def _speak_voice_alert_async(self, message: str):
        """Synthesize spoken voice warning asynchronously via Windows Speech API."""
        def _speak():
            try:
                # Use powershell system speech (built into Windows, 100% offline)
                cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{message}')"
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True,
                    timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except Exception:
                # Fallback to beep
                self._play_windows_beep(1000, 300)

        threading.Thread(target=_speak, daemon=True).start()

    def check_and_alert(
        self,
        distance_ahead_m: float,
        time_to_impact_sec: float,
        severity_cat: str,
        depth_cm: float = 5.0,
    ) -> bool:
        """
        Evaluate if an alert should be triggered for approaching pothole hazard.

        Returns
        -------
        bool: True if an audio alert was emitted.
        """
        if not self.enabled:
            return False

        now = time.time()
        if now - self.last_alert_time < self.cooldown_sec:
            return False

        # Only alert for hazards within warning distance and Medium/High severity
        is_hazard = (distance_ahead_m <= self.warning_dist_m and severity_cat in ["Medium", "High"])

        if is_hazard:
            with self.alert_lock:
                self.last_alert_time = now

            # Immediate chime tone
            threading.Thread(target=lambda: self._play_windows_beep(1400, 200), daemon=True).start()

            # Voice warning
            if distance_ahead_m < 15.0:
                msg = f"Warning! Pothole {int(distance_ahead_m)} meters ahead! Slow down!"
            else:
                msg = f"Caution! Road defect {int(distance_ahead_m)} meters ahead!"

            self._speak_voice_alert_async(msg)
            return True

        return False

    def trigger_pothole_voice_alert(
        self,
        severity_category: str = "High",
        depth_cm: float = 5.0,
        distance_m: float = 20.0,
        custom_msg: str = None,
    ):
        """Trigger an immediate spoken voice warning."""
        if not self.enabled:
            return
        msg = custom_msg or f"Warning! {severity_category} hazard {int(distance_m)} meters ahead! Slow down!"
        self._speak_voice_alert_async(msg)

