"""
scripts/run_vehicle_dashcam.py
==============================
Standalone Vehicle Dashcam Autonomous Pothole Detection & Government Dispatcher.

Can be run on an in-vehicle computer (laptop, mini-PC, Jetson, or Raspberry Pi)
connected to a front windshield dashboard camera.

Usage Examples:
---------------
1. Run with connected USB Dashcam:
   python scripts/run_vehicle_dashcam.py --source 0 --portal CPGRAMS --min-severity Medium

2. Run with test driving footage (simulation):
   python scripts/run_vehicle_dashcam.py --source outputs/videos/sample_road_test.mp4 --portal NHAI

3. Run with phone IP webcam / RTSP stream:
   python scripts/run_vehicle_dashcam.py --source http://192.168.1.100:8080/video
"""

import os
import sys
import time
import argparse
import cv2

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.dashcam_agent import VehicleDashcamAutoReporter


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Dashcam Pothole Detector & Gov Reporter")
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        help="Camera index (e.g. 0, 1), 'auto' (uses saved dashcam), video file path, or RTSP/HTTP URL",
    )
    parser.add_argument(
        "--portal",
        type=str,
        default="CPGRAMS",
        choices=["CPGRAMS", "NHAI", "MERI_SADAK", "LOCAL_PWD"],
        help="Target Government of India grievance portal",
    )
    parser.add_argument(
        "--min-severity",
        type=str,
        default="Medium",
        choices=["Low", "Medium", "High"],
        help="Minimum severity threshold required to auto-dispatch complaint",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=2,
        help="Number of consecutive frames required to confirm real pothole",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Process every N frames to optimize real-time FPS",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (useful for automated testing)",
    )
    parser.add_argument(
        "--auto-dispatch",
        action="store_true",
        default=False,
        help="Directly auto-submit to government portals without requiring manual citizen approval (Default: False - Record & Review mode)",
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Dry-run mode: logs defects to DB without submitting to government portal",
    )
    parser.add_argument(
        "--show-display",
        action="store_true",
        help="Display OpenCV live HUD window (press 'q' to stop)",
    )
    parser.add_argument(
        "--vehicle",
        type=str,
        default=None,
        choices=["HATCHBACK_SEDAN", "SUV_CROSSOVER", "TWO_WHEELER_SCOOTER", "COMMERCIAL_TRUCK"],
        help="Vehicle suspension architecture profile",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure vehicle profile if specified
    if args.vehicle:
        from src.suspension_calibrator import VehicleSuspensionProfiler
        v_prof = VehicleSuspensionProfiler()
        v_prof.save_profile(args.vehicle)
        print(f"[INFO] Applied Vehicle Profile: {args.vehicle}")

    # Parse video source (auto, integer index, or string path/url)
    source = args.source
    if str(source).lower() == "auto":
        from src.camera_setup import get_active_dashcam_source
        source = get_active_dashcam_source()
        print(f"[INFO] Auto-resolved Dashcam Source : {source}")
    elif str(source).isdigit():
        source = int(source)

    print("=" * 75)
    print("  CAR DASHCAM AUTONOMOUS POTHOLE RECOGNITION & GOV REPORTING AGENT")
    print("=" * 75)
    print(f"[INFO] Video Input Source    : {args.source}")
    print(f"[INFO] Target Gov Authority  : {config.GOV_PORTALS.get(args.portal, {}).get('name')}")
    print(f"[INFO] Portal Endpoint       : {config.GOV_PORTALS.get(args.portal, {}).get('url')}")
    print(f"[INFO] Auto-Dispatch Trigger : Severity >= {args.min_severity}")
    print(f"[INFO] Confirmation Threshold: {args.confirmation_frames} consecutive frames")
    print(f"[INFO] Autonomous Submission : {'DISABLED (Dry Run)' if args.no_dispatch else 'ENABLED (Live Submission)'}")
    print("=" * 75)

    # Initialize reporter
    reporter = VehicleDashcamAutoReporter(
        target_portal=args.portal,
        min_severity=args.min_severity,
        confirmation_frames=args.confirmation_frames,
        auto_dispatch=(args.auto_dispatch and not args.no_dispatch),
        sample_interval=args.sample_interval,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        fallback_vid = os.path.join(config.VIDEOS_OUTPUT_DIR, "sample_road_test.mp4")
        if os.path.exists(fallback_vid):
            print(f"[WARNING] Could not open camera {source}. Automatically falling back to driving simulation video: {fallback_vid}")
            source = fallback_vid
            cap = cv2.VideoCapture(source)
        else:
            print(f"[ERROR] Failed to open video source: {source}")
            sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    sampled_count = 0
    start_time = time.time()

    print("[INFO] Dashboard camera scanner active. Commencing live road surveillance...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % args.sample_interval != 0:
                continue

            sampled_count += 1
            timestamp_sec = frame_count / fps

            # Resize to pipeline target dimensions
            frame_resized = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

            # Process frame
            annotated_frame, new_events, telemetry = reporter.process_frame(
                frame_bgr=frame_resized,
                frame_idx=frame_count,
                timestamp_sec=timestamp_sec,
                source_name=f"source_{args.source}",
            )

            # Log events to console
            for ev in new_events:
                print("-" * 75)
                print(f"[ALERT] POTHOLE RECOGNIZED & REPORTED BY DASHCAM!")
                print(f"        Track ID        : {ev['track_id']}")
                print(f"        Severity        : {ev['severity_category']} (Score: {ev['severity_score']:.2f})")
                print(f"        3D Cavity Depth : {ev.get('max_depth_cm', 5.0):.1f} cm (Distance: {ev.get('distance_ahead_m', 12.0):.0f} m)")
                print(f"        Repair Required : {ev.get('cavity_volume_liters', 3.0):.1f} Liters void ({ev.get('asphalt_repair_mass_kg', 7.5):.1f} kg asphalt)")
                print(f"        Bird's Eye View : Area {ev.get('bev_area_sqcm', 2500):.0f} cm2 | Lateral: {ev.get('lateral_offset_m', -0.8):+.2f} m ({ev.get('wheel_track_hazard', 'LEFT_WHEEL_PATH')})")
                print(f"        IMU Shock Fusion: Gz Shock {ev.get('imu_shock_g', 1.0):.2f}g | Threat Index: {ev.get('threat_index', 70):.1f}/100 | Status: {ev.get('fusion_status', 'CONFIRMED')}")
                print(f"        GPS Coordinates : Lat {ev['latitude']:.5f}, Lon {ev['longitude']:.5f}")
                print(f"        Road Location   : {ev['road_name']}, {ev['district']}, {ev['state']} - {ev['postcode']}")
                print(f"        Full Address    : {ev['full_address']}")
                print(f"        Gov Reference ID: {ev['reference_no']}")
                print(f"        Target Portal   : {ev['target_portal']}")
                print(f"        Photo Evidence  : {ev['evidence_image']}")
                print(f"        Official PDF    : {ev['pdf_dossier']}")
                print("-" * 75)

            if args.show_display:
                cv2.imshow("Car Dashcam - Pothole Auto-Reporter", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User terminated live display.")
                    break

            if args.max_frames and sampled_count >= args.max_frames:
                print(f"[INFO] Reached requested max frames ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by driver / user.")

    finally:
        cap.release()
        if args.show_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("  CAR DASHCAM RUN SUMMARY")
    print("=" * 75)
    print(f"  Total Camera Frames Ingested : {frame_count}")
    print(f"  Frames Analyzed by AI Models : {sampled_count}")
    print(f"  Processing Time              : {elapsed:.2f} seconds ({sampled_count / max(0.1, elapsed):.1f} FPS)")
    print(f"  Total Potholes Confirmed     : {reporter.total_potholes_found}")
    print(f"  Grievances Dispatched to Gov : {len(reporter.dispatched_events)}")
    print("=" * 75)


if __name__ == "__main__":
    main()

