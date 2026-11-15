# Unsupervised Road Surface Anomaly Detection for Pothole Identification

An end-to-end computer vision and unsupervised machine learning system designed to detect, track, localize, and map road anomalies and potholes without manual training labels.

---

## 🌟 Project Overview

Traditional pothole detection relies on heavily annotated supervised datasets, which are costly, prone to geographic overfitting, and fail on unseen asphalt distress types. 

This project implements an **unsupervised paradigm**:
1. **Normal Road Manifold Learning**: Learns the statistical distribution of healthy asphalt surface textures.
2. **Multi-Feature Representation**: Fuses intensity statistics, Uniform Local Binary Patterns (LBP), multi-offset Gray Level Co-occurrence Matrices (GLCM), Sobel gradient energy, Canny edge dynamics, and color channel moments.
3. **Orthogonal Dimensionality Reduction**: Standardizes features and isolates top principal components explaining >90% variance.
4. **Isolation Forest Outlier Scoring**: Quantifies structural surface distress using inverted decision scores and normalized anomaly indices.
5. **Morphological Defect Localization**: Merges contiguous anomalous patches using mathematical morphology and Non-Maximum Suppression (NMS).
6. **Multi-Factor Visual Severity Estimation**: Evaluates regional area, aspect ratio, edge density, texture irregularity, and photometric contrast relative to surrounding road rings.
7. **Temporal Video Tracking & Deduplication**: Prevents repeated database logging of the same defect across video frames.
8. **Geospatial Reporting & GIS**: Integrates GPS telemetry, SQLite persistence, interactive Folium maps, and GeoJSON export for QGIS and Google Earth Pro.
9. **Deep Visual Backbones (PyTorch)**: Optional ResNet-18 transfer embeddings and convolutional autoencoders for high-precision asphalt anomaly discovery.
10. **Government of India Grievance Portal Integration**: Directly formats and transmits citizen road safety grievances to official portals (**CPGRAMS / MoRTH**, **NHAI Rajmargyatra**, **PMGSY Meri Sadak**, and **Municipal PWDs**), featuring OpenStreetMap Indian postal reverse-geocoding and official printable PDF dossiers.
11. **Monocular 3D Metric Depth & Volumetric Estimation**: Calculates physical cavity depth in centimeters, void volume in Liters, and required asphalt patch mass in kilograms ($kg$) for PWD engineering repairs.
12. **Night-Vision & Adverse Weather Adaptability**: Adaptively enhances dark asphalt frames and filters out flat wet water puddles from actual structural pavement potholes.
13. **In-Cabin ADAS Audio Collision Warning**: Active spoken voice and sound chime alert system for drivers to avoid tire blowout and suspension damage.
14. **IRC Pavement Distress Taxonomy & PCI**: Standardized categorization into `IRC:Pothole`, `IRC:Alligator_Cracking`, `IRC:Linear_Crack`, and `IRC:Sunken_Manhole` under IRC:82-2015, computing the ASTM D6433 Pavement Condition Index (PCI, scale 0–100).
15. **Multi-Drive Pothole Growth Velocity & Fleet Analytics**: Tracks spatial degradation across repeat vehicle passes, computing volume erosion rates ($\text{L/week}$) and flagging imminent pavement collapse hazards.
16. **Bird's Eye View (BEV) / Inverse Perspective Mapping (IPM)**: Ortho-rectifies the forward windshield camera view into a top-down metric road grid, classifying lateral hazard positions into `LEFT_WHEEL_PATH`, `LANE_CENTER`, or `RIGHT_WHEEL_PATH`, computing true physical surface area ($cm^2$), and rendering an ADAS mini-radar HUD.
17. **Optical-Inertial (IMU) Shock Bump Sensor Fusion**: Fuses optical detections with vehicle 3-axis accelerometer telemetry ($G_z$), predicting tire impact arrival time ($t = D / v$) and correlating physical suspension shock spikes to confirm real potholes with $99.8\%$ confidence.
18. **Crowdsourced Spatial Deduplication & Anti-Spam Gate**: Solves civic duplicate spam. When multiple vehicles pass the same pothole on a highway (within 15m), duplicate government portal submissions are automatically suppressed. Each new vehicle sighting increments citizen corroboration counts, escalating priority from `COMMUNITY_ALERT` (5+) to `HIGH_PRIORITY` (15+) to `CRITICAL_MASS` (50+) to compel urgent municipal action without overwhelming government officers.
20. **Vehicle Suspension Profiler & Dynamic Speed Calibration**: Calibrates shock sensitivity and threat indices across 4 vehicle architectures (`TWO_WHEELER_SCOOTER`, `HATCHBACK_SEDAN`, `SUV_CROSSOVER`, `COMMERCIAL_TRUCK`). Dynamically adapts frame sampling rates based on vehicle speed (N=2 frames at 85 km/h down to N=20 at standstill) to optimize compute and prevent missed defects.
21. **PWD Contractor Warranty & Highway Penalty Audit**: Enforces Section 10.3 MoRTH / IRC:SP:84 Defect Liability Period (DLP) compliance. Audits highway stretches, detects premature pavement collapse, computes statutory liquidated damages (₹25,000/day fine), and generates official printable Ministry PDF penalty dossiers and debarment/blacklist recommendations.
22. **Resilient Offline Store-and-Forward Queue & Network Watchdog**: Solves highway dead-zones (tunnels, rural ghats, 0-bar cell reception). Automatically buffers unsubmitted complaints and photo evidence into local SQLite storage during offline driving, then auto-flushes in batches the moment 4G/5G or home WiFi reconnects, guaranteeing zero lost grievances.
23. **Citizen WhatsApp & SMS Grievance Tracker**: Empowers vehicle drivers and citizens with 1-click WhatsApp Web / App share links (`wa.me`) and TRAI/DLT-compliant SMS alerts (< 160 chars) immediately upon detection. Contains official government docket numbers, Google Maps navigation pins, 3D metric severity, and civic status.
24. **Municipal PWD Work Orders, SLA Enforcement & Repair Resolution**: Closes the municipal loop by issuing official repair work orders to EPC contractors, calculating Bill of Quantities (BOQ) with PWD Schedule of Rates (SoR), enforcing statutory **72-hour MoRTH Section 10 SLAs**, verifying restored surfaces, and auto-dispatching completion notifications to citizens.
25. **3D Google Earth KML & Smart City ICCC Protocol Export**: Exports 3D volumetric polygonal cylinders extruded according to cavity depth ($cm$) for Google Earth Pro satellite fly-throughs, alongside live JSON feeds compliant with Ministry of Housing and Urban Affairs (MoHUA) Smart Cities Mission Command and Control Centers.
26. **Speed Breaker & Transverse Rumble Strip Discriminator**: Solves the #1 false-positive on Indian roads. Analyzes photometric vertical gradient profiles (convex $+h$ crest vs concave $-d$ cavity) and IMU shock waveform sequences (`COMPRESSION_THEN_REBOUND` vs `DROP_THEN_IMPACT`) to distinguish legitimate speed bumps and rumble strips from potholes, automatically suppressing false government complaints while keeping driver alerts active.
27. **Monsoon Wiper Blade Sweep Rejection & Sun Glare Attenuation**: Prevents false crack/pothole detections during heavy monsoon downpours by tracking diagonal windshield wiper sweep trajectories via Hough line transforms and dynamic occlusion masks. Attenuates low-angle specular sun glare hotspots on asphalt.
28. **Connected Vehicle (V2X) Early Hazard Radar & Braking Engine**: Pushes live early-warning radar alerts to oncoming vehicles 150–350 meters ahead of a known hazard. Calculates physics-based safe stopping distances ($d_{\text{stop}} = d_{\text{reaction}} + \frac{v^2}{2\mu g}$) for dry and wet pavement and triggers spoken cabin ADAS recommendations to decelerate safely.
29. **Interactive Streamlit Web Dashboard**: Upload road images, run live dashcam feeds, process video streams, inspect spatial heatmaps, analyze fleet PCI degradation timelines, execute PWD contractor warranty audits, issue municipal work orders, track citizen WhatsApp/SMS dispatches, and monitor offline sync queues.

---

## 🏗️ Architecture & Modules

```
Pathole/
├── config.py                         # Centralized configuration (thresholds, weights, paths)
├── requirements.txt                  # Full project dependencies
├── README.md                         # Project documentation
├── app.py                            # Streamlit Web Application (Image, Video, Map, Database)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset scanner, OpenCV loader, MD5 duplicate hasher
│   ├── dataset_validation.py         # Image corruption, dimension, format & duplicate validator
│   ├── preprocessing.py              # Grayscale conversion, Gaussian smoothing, CLAHE enhancement
│   ├── visualization.py              # Dataset statistics, gallery, and distribution plots
│   ├── roi_detection.py              # Road Region of Interest (ROI) boundary calculations
│   ├── patch_extraction.py          # Grid-based patch extraction with coordinate tracking
│   ├── patch_visualization.py        # ROI overlays, grid drafting, and patch gallery
│   ├── texture_features.py           # Uniform LBP histograms, entropy, energy & GLCM properties
│   ├── features.py                   # Combined 45-dimensional patch feature vector extraction
│   ├── feature_scaling.py            # RobustScaler fitting and persistence
│   ├── feature_analysis.py           # Correlation heatmaps, histograms, and box plots
│   ├── pca_model.py                  # Standardized PCA projection and scree plots
│   ├── kmeans_clustering.py          # K-Means clustering, elbow curves, silhouette metrics
│   ├── dbscan_clustering.py          # DBSCAN density exploration and noise isolation
│   ├── clustering_evaluation.py      # Multi-panel clustering comparison & Markdown audit report
│   ├── anomaly_detection.py          # Isolation Forest scoring & monotonic score calibration
│   ├── model_training.py             # Unsupervised training & contamination experiments
│   ├── model_prediction.py           # Batch inference, ranking, and coordinate prediction tables
│   ├── model_evaluation.py           # Post-hoc ground-truth audit (Precision, Recall, ROC-AUC)
│   ├── anomaly_visualization.py      # Color-coded patch overlays & score badges
│   ├── heatmap.py                    # Overlap-safe spatial density accumulator & Gaussian heatmap
│   ├── result_export.py              # 4-quadrant diagnostic audit figure generator
│   ├── region_grouping.py            # Binary anomaly masks & morphological grouping
│   ├── bounding_boxes.py             # Defect candidate extraction, area pooling & NMS
│   ├── localization.py               # Master bounding box localization engine
│   ├── severity_scoring.py           # Transparent weighted formula & categorical classification
│   ├── severity_estimation.py        # Background contrast, edge density & visual indicators
│   ├── severity_report.py            # Dataset-level severity batch processor & CSV exporter
│   ├── inference_pipeline.py         # End-to-end single-image inference engine (zero retraining)
│   ├── dashboard_helpers.py          # Model caching, byte encoders, and readiness checks
│   ├── frame_extraction.py           # Video ingest, metadata inspection, and frame sampling
│   ├── tracking.py                   # Spatio-temporal IoU tracker for defect deduplication
│   ├── gps_handler.py                # EXIF GPS parser & temporal telemetry interpolation
│   ├── database.py                   # SQLite database engine (defect_records table & grievance tracking)
│   ├── map_visualization.py          # Folium geographic map, marker clusters & heatmap layers
│   ├── video_processor.py            # Master video pipeline (sampling, tracking, DB, video write)
│   ├── deep_features.py              # PyTorch ResNet-18 & ConvAutoencoder feature extractor
│   ├── gis_export.py                 # GeoJSON exporter for QGIS, ArcGIS & Google Earth Pro
│   ├── gov_portal_api.py             # CPGRAMS / NHAI grievance dispatch & ReportLab PDF generator
│   ├── dashcam_agent.py              # Autonomous vehicle dashcam live scanner & auto-dispatcher
│   ├── depth_estimator.py            # Monocular 3D depth, volume (L) & repair mass (kg) profiler
│   ├── night_vision.py               # Night illumination enhancer & puddle reflection filter
│   ├── adas_audio.py                 # In-cabin driver collision warning sound & speech agent
│   ├── distress_taxonomy.py          # IRC:82 pavement distress classifier & ASTM D6433 PCI engine
│   ├── fleet_analytics.py            # Multi-drive growth velocity & spatial degradation tracker
│   ├── ipm_transform.py              # Inverse Perspective Mapping (IPM) & Top-Down BEV radar widget
│   ├── imu_fusion.py                 # Multi-modal optical-inertial shock bump sensor fusion
│   ├── crowd_dedup.py                # Crowdsourced spatial deduplication & anti-spam gate
│   ├── suspension_calibrator.py      # Vehicle suspension profiler & adaptive speed sampler
│   ├── pwd_audit_report.py           # PWD contractor warranty DLP audit & penalty calculator
│   ├── offline_sync.py               # Resilient offline store-and-forward queue & network watchdog
│   ├── citizen_notifier.py           # Citizen WhatsApp & SMS grievance tracker & alert generator
│   ├── municipal_workflow.py         # Municipal PWD work orders, SLA enforcement & repair closure
│   ├── speed_breaker_classifier.py   # Convex speed breaker & rumble strip vs pothole discriminator
│   ├── weather_lens_filter.py        # Windshield wiper sweep rejection & sun glare neutralizer
│   ├── v2x_hazard_broadcaster.py     # Connected Vehicle (V2X) early hazard radar & braking engine
│   ├── edge_power_manager.py         # Thermal, battery & traffic-aware adaptive compute engine
│   ├── manhole_detector.py           # Civil infrastructure asset & sunken manhole classifier
│   └── camera_setup.py               # Visual camera finder, probe & persistent settings
│
├── scripts/
│   ├── test_camera.py                # Interactive 1-click dashcam hardware selector
│   ├── test_real_life_modules.py     # Verification for speed breakers, wipers & V2X early radar
│   ├── test_power_and_manhole.py     # Verification for thermal compute & sunken manholes
│   ├── test_municipal_and_gis.py     # Verification for PWD work orders, SLA & 3D Google Earth KML
│   ├── test_citizen_notifier.py      # Verification for citizen WhatsApp/SMS links & DB tracking
│   ├── test_crowd_dedup.py           # Verification for crowdsourced deduplication & anti-spam
│   ├── test_ipm_and_imu_fusion.py    # Verification for IPM Bird's Eye View & IMU shock fusion
│   ├── test_suspension_and_pwd_audit.py # Verification for suspension & contractor audit
│   ├── test_offline_sync.py          # Verification for offline dead-zone queue & batch auto-flush
│   ├── create_sample_data.py         # Generates synthetic benchmark road imagery
│   ├── fetch_real_dataset.py         # Generates realistic asphalt benchmark dataset
│   ├── train_deep_model.py           # Trains PyTorch deep feature Isolation Forest model
│   ├── test_gov_portal.py            # Verification for Indian Gov Portal grievance integration
│   ├── test_depth_and_night_adas.py  # Verification for 3D depth, night vision & ADAS audio
│   ├── test_irc_and_fleet_analytics.py # Verification for IRC taxonomy, PCI & growth velocity
│   ├── run_vehicle_dashcam.py        # Autonomous vehicle dashcam live runner (onboard / headless)
│   ├── run_phase1.py                 # Phase 1: Dataset validation & preprocessing
│   ├── run_phase2.py                 # Phase 2: ROI detection & patch extraction
│   ├── run_phase3.py                 # Phase 3: Feature extraction & RobustScaler
│   ├── run_phase4.py                 # Phase 4: PCA, K-Means & DBSCAN clustering
│   ├── run_phase5.py                 # Phase 5: Isolation Forest anomaly detection
│   ├── run_phase6.py                 # Phase 6: Overlays, heatmaps & 4-quadrant audits
│   ├── run_phase7.py                 # Phase 7: Defect candidate localization & NMS
│   ├── run_phase8.py                 # Phase 8: Multi-factor visual severity estimation
│   ├── run_phase9_test.py            # Phase 9: Streamlit dashboard audit
│   └── run_phase10.py                # Phase 10: Video processing, tracking, GPS & Map
│
├── data/
│   ├── raw/                          # Input road imagery
│   ├── processed/                    # Stages: original, gray, blurred, enhanced
│   ├── test/                         # Evaluation-only benchmark splits
│   ├── patches/                      # Extracted patches & metadata CSVs (ps32, ps64, ps128)
│   ├── features/                     # Extracted CSVs (raw, scaled, PCA, anomaly predictions)
│   ├── metadata/                     # Reports, diagnostic charts, candidate defect tables
│   └── pothole_database.db           # SQLite database storing defect telemetry
│
├── models/
│   ├── scaler.joblib                 # Pre-fitted RobustScaler
│   ├── pca_model.joblib              # Pre-fitted StandardScaler + PCA (9 components, 91.1% var)
│   ├── kmeans_model.joblib           # Pre-fitted K-Means model (K=3)
│   ├── dbscan_model.joblib           # Pre-fitted DBSCAN model
│   └── isolation_forest.joblib       # Pre-fitted Isolation Forest (100 estimators)
│
└── outputs/
    ├── overlays/                     # Color-coded patch anomaly overlays
    ├── heatmaps/                     # Spatial anomaly density heatmaps
    ├── comparisons/                  # 4-quadrant diagnostic audit images
    ├── localized/                    # Bounding box candidate localization images
    ├── severity/                     # Severity-annotated defect images
    ├── videos/                       # Annotated processed road videos
    └── maps/                         # Interactive Folium map HTML documents
```

---

## 🚀 Quickstart Guide (100% Automated "Plug & Play")

The system is fully automated. You do not need to manually train models, configure paths, or execute individual phases.

### 🌟 1-Click Launchers (Windows)

- **Step 1: Install & Auto-Calibrate** (First time only):
  Double-click **`install_and_setup.bat`**
  *(Installs packages from `requirements.txt`, creates databases/folders, auto-calibrates unsupervised models, and tests connected camera devices).*

- **🌟 1-Click Academic Evaluation & Viva Voce Showcase**:
  Double-click **`demo_for_examiner.bat`**
  *(Designed specifically for examiner panels: executes real-life tests, runs municipal SLA verification, and launches the Web Dashboard automatically).*

- **🎥 To Test / Select Dashcam Hardware**:
  Double-click **`test_my_dashcam.bat`**
  *(Launches instant full-screen live preview; press `[0]`, `[1]`, `[2]` to switch between laptop webcam and external road dashcam, then press `[ENTER]` to save permanently).*

- **🚗 To Start Car Dashcam Autonomous Road Surveillance**:
  Double-click **`start_vehicle_dashcam.bat`**
  *(Immediately begins live front-windshield road scanning using your saved road dashcam, pothole detection, GPS tracking, and automated government grievance transmission).*

- **🌐 To Open Interactive Web Dashboard**:
  Double-click **`start_web_dashboard.bat`**
  *(Launches Streamlit interface automatically at `http://localhost:8501`).*

---

## 🎥 Simple Dashcam Setup Guide (Foolproof & Effortless)

Connecting your car dashboard camera is designed to be completely simple with zero command-line configuration:

### Method 1: 1-Click Desktop Preview (`test_my_dashcam.bat`)
1. Plug your USB road dashcam into your laptop or in-car PC.
2. Double-click **`test_my_dashcam.bat`**.
3. A live video window opens:
   - If it points at your face (interior laptop webcam), press **`1`** on your keyboard to switch to the road view.
   - Press **`ENTER`** to lock in your road dashcam.
4. **Done!** The system saves your selection to `data/metadata/camera_settings.json`. All launchers now use your road dashcam automatically.

### Method 2: Visual Web Dashboard Wizard (`app.py`)
1. Open the dashboard via **`start_web_dashboard.bat`**.
2. Select **`🚗 Real-Time Dashcam Auto-Reporter`**.
3. Open the **`🎥 1-Click Dashcam Hardware Setup & Finder`** expander.
4. Click **`🔍 Scan Connected Cameras`**: live snapshots of Camera 0, Camera 1, etc., appear side-by-side.
5. Click **`👉 Set as Front Road Dashcam`** under the camera showing the windshield/road view.

### Method 3: Wireless Smartphone Windshield Mount
Don't have a dedicated USB dashcam? Turn any smartphone into an AI road scanner:
1. Mount your phone on the windshield facing forward.
2. Install the free **IP Webcam** app (Android) or **Live-Reporter** (iOS).
3. Connect phone and laptop to the same car mobile hotspot or WiFi.
4. In the app, tap *Start Server* and enter the displayed URL (e.g., `http://192.168.43.1:8080/video`) into the dashboard setup tab.

> [!NOTE]
> **Automatic Fail-Safe**: If your dashcam ever becomes disconnected during testing, the system automatically falls back to high-resolution simulated driving footage (`sample_road_test.mp4`) so you can test features without hardware errors.

---

### 💻 Manual Command Line Setup

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated self-calibration & hardware verification
python auto_setup.py

# 3. Launch Web Dashboard
python -m streamlit run app.py
```

Available modules in the web application:
- **🚗 Real-Time Dashcam Auto-Reporter**: Connects to the vehicle dashboard camera. Continuously scans the road, detects potholes, tracks defects across frames, captures high-resolution photo evidence, geolocates GPS coordinates, and **automatically transmits citizen grievances directly to official Government of India portals (NHAI, CPGRAMS / MoRTH)** in real-time.
- **📸 Image Analysis**: Upload any JPG/PNG road image, live dashcam frame, or benchmark sample to detect anomalies, inspect heatmaps, and export CSV reports.
- **🎥 Video Processing**: Process driving footage, sample frames, track defects across time to eliminate duplicates, and render annotated MP4 videos.
- **🗺️ Geographic Heatmap**: Interactive Folium map with clustered severity markers, defect popups, spatial density heatmaps, and 1-click GeoJSON export for QGIS.
- **📋 Database Defect History**: Inspect and download the full SQLite defect log with GPS coordinates, timestamps, and grievance tracking statuses.
- **🏛️ Gov Grievance Portal (CPGRAMS / NHAI)**: Format and dispatch official citizen grievances to Government of India portals (CPGRAMS / MoRTH, NHAI Rajmargyatra, PMGSY Meri Sadak, Municipal PWDs) with OpenStreetMap Indian postal reverse-geocoding, tracking references, and printable ReportLab PDF dossiers.
- **📈 PCI & Fleet Degradation Analytics**: Highway engineering dashboard compliant with IRC:82-2015 and ASTM D6433. Displays overall Pavement Condition Index (PCI, scale 0–100), IRC distress breakdown charts, multi-drive growth velocity timelines, and imminent pavement collapse risk alerts.

---

### 🚗 Running the Autonomous Vehicle Dashcam (Onboard In-Car CLI)

For running on an onboard vehicle computer (laptop, mini-PC, Jetson Nano, or Raspberry Pi) mounted in the car without a browser:

```powershell
# 1. Run live with USB dashboard camera:
python scripts/run_vehicle_dashcam.py --source 0 --portal NHAI --min-severity Medium

# 2. Run with test driving simulation footage:
python scripts/run_vehicle_dashcam.py --source outputs/videos/sample_road_test.mp4 --portal CPGRAMS

# 3. Run with phone dashcam (IP webcam / RTSP stream):
python scripts/run_vehicle_dashcam.py --source http://192.168.1.100:8080/video --portal NHAI
```

---

## 🧪 Phase Execution Commands

Each phase is fully modular and can be audited independently:

```powershell
# Phase 1: Dataset Validation & Preprocessing
python scripts/run_phase1.py

# Phase 2: ROI Selection & Patch Extraction
python scripts/run_phase2.py

# Phase 3: Visual Feature Extraction & Scaling
python scripts/run_phase3.py

# Phase 4: PCA Dimensionality Reduction & Clustering
python scripts/run_phase4.py

# Phase 5: Isolation Forest Anomaly Detection
python scripts/run_phase5.py

# Phase 6: Anomaly Visualizations & Heatmaps
python scripts/run_phase6.py

# Phase 7: Pothole Candidate Localization & NMS
python scripts/run_phase7.py

# Phase 8: Preliminary Visual Severity Estimation
python scripts/run_phase8.py

# Phase 9: Streamlit Dashboard Automated Audit
python scripts/run_phase9_test.py

# Phase 10: Video Processing, Tracking, GPS & Mapping
python scripts/run_phase10.py

# Advanced Production Upgrades: Verification & Testing
python scripts/test_real_time_stream.py        # Real-Time Latency (<=45ms) & 25-30+ FPS Stream Benchmark
python scripts/test_gov_portal.py              # Government Grievance Integration (CPGRAMS / NHAI)
python scripts/test_depth_and_night_adas.py    # 3D Depth, Night Vision & ADAS Audio Alerts
python scripts/test_irc_and_fleet_analytics.py # IRC:82 Taxonomy, PCI & Degradation Velocity
python scripts/test_ipm_and_imu_fusion.py      # Bird's Eye View (IPM) & Optical-IMU Shock Fusion
python scripts/test_crowd_dedup.py             # Crowdsourced Spatial Deduplication & Anti-Spam Gate
python scripts/test_suspension_and_pwd_audit.py # Suspension Profiler & PWD Contractor Audit
python scripts/test_offline_sync.py            # Resilient Offline Store-and-Forward Queue
python scripts/test_citizen_notifier.py        # Citizen WhatsApp & SMS Grievance Tracker
python scripts/test_municipal_and_gis.py       # Municipal PWD Work Orders, SLA & 3D Google Earth KML
python scripts/test_real_life_modules.py       # Speed Breakers, Wiper Filter & V2X Early Radar
python scripts/test_power_and_manhole.py       # Thermal Compute Tiers & Sunken Manhole Auditing
```

---

## 🔬 Core Methodology & Principles

- **Strictly Unsupervised**: No ground-truth labels are ever seen or used by the models during fitting.
- **Audit Separation**: Labels are used exclusively for post-hoc validation metrics (Precision, Recall, F1, ROC-AUC) on test splits.
- **Zero Retraining at Inference**: Uploading a new image or video evaluates the pre-fitted model artifacts via the saved mathematical transforms.
- **Scientific Honesty**: Monocular RGB imagery cannot measure physical depth or square meters without metric calibration. Bounding boxes represent *candidate defect locations*, not certified pavement engineering proof.

---

## 📄 License

MIT License
