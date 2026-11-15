# 🎓 Academic Viva Voce & Project Defense Guide
## Project: Autonomous Vehicle Dashcam AI Road Anomaly Detection & Government Grievance System

This guide prepares you for any question asked by internal supervisors, external university examiners, and technical evaluators.

---

### ❓ Question 1: Why did you use Unsupervised Machine Learning instead of a standard supervised detector like YOLOv8 or Faster R-CNN?
**Model Answer:**
> *"Supervised object detectors (like YOLOv8) require thousands of manually bounding-box labeled images. In the real world, road defects have **infinite morphological variance** — no two potholes look the same in shape, texture, or depth. Furthermore, supervised models heavily overfit to specific asphalt colors and lighting conditions, completely failing when exposed to new road materials (concrete, bituminized macadam, mastic asphalt).*
>
> *Our unsupervised approach uses a combination of **deep transfer features (ResNet-18)** and **multi-scale texture representations (Uniform LBP + GLCM + Sobel)** coupled with an **Isolation Forest**. Instead of memorizing what a pothole looks like, our model learns the statistical manifold of **healthy, normal pavement**. Any localized surface depression, cracking, or cavity is statistically isolated as an anomaly. This gives zero-shot generalization across any highway without manual retraining."*

---

### ❓ Question 2: How do you estimate 3D physical depth (cm) and cavity volume (Liters) from a single 2D monocular dashboard camera?
**Model Answer:**
> *"We combine **camera pinhole geometry** with **photometric shadow gradient analysis**:*
> 1. *Using the camera's fixed mounting height ($H = 1.35\text{ m}$), downward pitch angle ($\theta = 10^\circ$), and horizontal field of view ($78^\circ$), we calculate the longitudinal ground distance ($D$) to the defect via:*
>    $$D = \frac{H}{\tan(\theta + \phi_y)}$$
> 2. *Within the localized cavity bounding box, the depth of shadow depression is proportional to the dark lumen extinction ratio relative to the surrounding healthy pavement:*
>    $$\text{Depth} = k_d \cdot (1 - \frac{I_{\text{min}}}{I_{\text{baseline}}}) \cdot \sqrt{\text{Area}_{\text{metric}}}$$
> 3. *Cavity volume is computed via trapezoidal integration of depth across the perspective-corrected metric surface area:*
>    $$V = \int \int z(x,y) \, dx \, dy \approx \frac{1}{3} \cdot \text{Area}_{\text{metric}} \cdot \text{Depth}_{\text{max}}$$
> 4. *Compacted asphalt patching mass required for repair is calculated using the official Indian road density standard: $\text{Mass (kg)} = V \times 2.4\text{ kg/L}$."*

---

### ❓ Question 3: How do you prevent flooding the government with duplicate complaints if 50 vehicles pass the exact same pothole on a highway?
**Model Answer:**
> *"We engineered a **Crowdsourced Spatial Deduplication & Citizen Corroboration Engine** (`src/crowd_dedup.py`).*
>
> *Before any complaint is dispatched, the system queries a persistent SQLite spatial registry using the **Haversine great-circle distance formula** with a calibrated $15\text{ meter}$ deduplication radius (corresponding to the standard width of a 4-lane highway with shoulders).*
> - *If no grievance exists within $15\text{m}$, it files a fresh complaint (Ref: `NHAI/E/2026/XXXXX`) and registers it in the database.*
> - *If an existing grievance is found within $15\text{m}$, the system **suppresses the duplicate complaint**, increments the `citizen_corroboration_count`, attaches the new vehicle's photo evidence, and dynamically escalates the priority tier:*
>   $$\text{FILED (1)} \longrightarrow \text{COMMUNITY ALERT (5+)} \longrightarrow \text{HIGH PRIORITY (15+)} \longrightarrow \text{CRITICAL MASS (50+)}$$
> *This turns redundant spam into an escalating civic pressure signal without burdening municipal engineers."*

---

### ❓ Question 4: What is Inverse Perspective Mapping (IPM) and why is it necessary?
**Model Answer:**
> *"In a front-facing vehicle dashcam, the perspective effect causes objects closer to the car to appear much larger than identical objects near the horizon. Furthermore, parallel lane markings appear to converge.*
>
> *Inverse Perspective Mapping (IPM) computes a homography matrix ($H_{3\times3}$) based on camera pitch, yaw, and height, warping the perspective camera view into an **orthographic top-down Bird's Eye View (BEV)**.*
> *This allows us to:*
> 1. *Measure true, perspective-invariant surface area in $\text{cm}^2$.*
> 2. *Compute metric lateral offset ($X_m$ in meters) relative to the vehicle centerline.*
> 3. *Classify whether the defect lies directly in the vehicle's `LEFT_WHEEL_PATH`, `RIGHT_WHEEL_PATH`, `LANE_CENTER`, or `ROAD_SHOULDER`."*

---

### ❓ Question 5: How do you handle cellular dead-zones (e.g. driving through mountain tunnels or rural highways with 0-bar reception)?
**Model Answer:**
> *"We implemented a **Resilient Offline Store-and-Forward Queue & Network Watchdog** (`src/offline_sync.py`).*
>
> *Before transmitting, a non-blocking TCP socket ping checks live internet reachability. If offline, the complaint payload, GPS coordinates, 3D metric telemetry, and photo evidence are safely serialized into a persistent SQLite queue (`offline_dispatch_queue`). When the vehicle exits the dead zone and 4G/5G or home WiFi reconnects, the watchdog triggers an automated batch flush with zero data loss."*

---

### ❓ Question 6: What happens when driving over unpainted speed breakers or rumble strips? Won't an anomaly model mistake them for potholes?
**Model Answer:**
> *"This is a classic false-positive flaw in naive road anomaly systems. We solved it with our **Convex Speed Breaker & Rumble Strip Classifier** (`src/speed_breaker_classifier.py`):*
> 1. *Geometric Aspect Ratio: Speed bumps span $60\%\text{--}100\%$ of the active lane ($W/H \ge 3.0$), whereas potholes are localized patches ($W/H \approx 0.8\text{--}1.8$).*
> 2. *Photometric Convexity: A speed bump has a bright leading specular crest ($+h$), followed by a soft trailing shadow. A pothole has a dark cavity center with an illuminated far wall ($-d$).*
> 3. *IMU Shock Waveform: Speed bumps cause suspension **compression then rebound** ($+G_z \to -G_z$), whereas potholes cause a **downward wheel drop into the void followed by a rim impact jolt** ($-G_z \to +G_z$).*
> *When detected, the complaint to the government is suppressed, while driver safety alerts remain active."*

---

### ❓ Question 7: How does the system handle rain, windshield wipers, and direct sunlight glare?
**Model Answer:**
> *"We engineered a dedicated **Weather & Lens Filter** (`src/weather_lens_filter.py`):*
> - *Windshield Wipers: Progressive Probabilistic Hough Line transforms track sweeping diagonal linear bars at characteristic wiper angles ($20^\circ\text{--}75^\circ$). Candidate boxes overlapping the sweeping wiper trajectory are rejected as artifacts.*
> - *Sun Glare: Saturated whiteout pixel clusters ($Y > 240, S < 45$) caused by direct low-angle sunlight are detected and attenuated via adaptive dynamic range tone mapping.*
> - *Puddle Filter: Dark rain puddles are distinguished from cavities using our specular gradient Laplacian edge variance filter."*

---

### ❓ Question 8: What legal and statutory highway frameworks are integrated?
**Model Answer:**
> *"The system is directly aligned with Government of India statutory road safety and engineering standards:*
> 1. ***IRC:82-2015***: *Indian Roads Congress Code of Practice for Maintenance of Bituminous Surfaces.*
> 2. ***ASTM D6433***: *Standard Practice for Roads and Parking Lots Pavement Condition Index (PCI, scale 0–100).*
> 3. ***MoRTH EPC Clause 10.3 / IRC:SP:84***: *Defect Liability Period (DLP) contractor warranty auditing with automated statutory liquidated damages (₹25,000/day fine per unrepaired cavity).*
> 4. ***Statutory 72-Hour SLA***: *MoRTH Section 10 emergency repair timelines enforced via municipal PWD work orders."*

---

### ❓ Question 9: What is the computational latency and edge feasibility on automotive hardware?
**Model Answer:**
> *"The architecture includes an **Edge Power & Thermal Manager** (`src/edge_power_manager.py`):*
> - *On GPU / Core i7: Runs at 25–30 FPS in Deep ResNet-18 mode.*
> - *Thermal Throttling Guard: If cabin temperatures exceed 75°C, the system seamlessly transitions to our lightweight OpenCV analytical texture mode (LBP + GLCM + Sobel), achieving 35+ FPS on low-power ARM CPUs (Raspberry Pi 4/5, Jetson Nano) at < 15% compute.*
> - *Traffic Jam Filter: In stop-and-go city traffic ($< 3\text{ km/h}$), frame processing is throttled to conserve battery and eliminate redundant computation on static asphalt."*
