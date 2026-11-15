"""
scripts/test_camera.py
======================
1-Click Interactive Camera Hardware Selector & Dashcam Preview.

Press:
- '0', '1', '2' : Switch live camera preview.
- 'ENTER' or 's': Save current camera as default Car Dashcam.
- 'q' or 'ESC'  : Exit preview.
"""

import os
import sys
import cv2

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.camera_setup import get_available_cameras, save_camera_config, get_saved_camera_config


def main():
    print("=" * 70)
    print("  CAR DASHBOARD CAMERA: 1-CLICK CONNECTION & SETUP WIZARD")
    print("=" * 70)
    print("[INFO] Probing connected camera devices...")

    cameras = get_available_cameras(max_probe=4)
    if not cameras:
        print("[WARNING] No physical USB cameras detected.")
        print("          Please plug your dashboard camera into a USB port and try again.")
        print("          Or use Smartphone WiFi stream / driving video simulation.")
        input("\nPress ENTER to close...")
        return

    saved_cfg = get_saved_camera_config()
    current_idx = saved_cfg.get("camera_index", cameras[0]["index"])

    # Ensure current_idx is in detected cameras
    available_indices = [c["index"] for c in cameras]
    if current_idx not in available_indices:
        current_idx = available_indices[0]

    print(f"\n[OK] Found {len(cameras)} active camera device(s):")
    for c in cameras:
        mark = " <-- [CURRENT ACTIVE DASHCAM]" if c["index"] == current_idx else ""
        print(f"     - Camera {c['index']}: Resolution {c['width']}x{c['height']}{mark}")

    print("\n[INSTRUCTIONS]:")
    print("  - A live video window will open.")
    print("  - If it points at your face (interior webcam), press '1' to switch to road dashcam.")
    print("  - Press 'ENTER' to save your road dashcam.")
    print("  - Press 'q' to quit.")
    print("=" * 70)

    cap = cv2.VideoCapture(current_idx)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            display = frame.copy()

            # Top overlay
            cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 25), -1)
            cv2.putText(
                display,
                f"LIVE CAMERA {current_idx} ({w}x{h}) - Pointing at Road?",
                (14, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "Press [0,1] to Switch | Press [ENTER] to Save as Car Dashcam | [Q] to Exit",
                (14, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Car Dashcam Setup Wizard - Road View Check", display)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord("q"), 27]:
                print("[INFO] Exited without changing settings.")
                break
            elif key in [13, ord("s"), ord("S")]:  # ENTER or 's'
                save_camera_config(camera_index=current_idx, source_type="usb_camera")
                print(f"\n[SUCCESS] Saved Camera {current_idx} as your default Car Dashcam!")
                print("          All autonomous surveillance scripts will now use this camera automatically.")
                break
            elif key in [ord("0"), ord("1"), ord("2"), ord("3")]:
                target_new = int(chr(key))
                if target_new in available_indices and target_new != current_idx:
                    print(f"[INFO] Switching to Camera {target_new}...")
                    cap.release()
                    current_idx = target_new
                    cap = cv2.VideoCapture(current_idx)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
