# config.py — Project-wide configuration for Pothole Detection Pipeline

# ==============================================================================
# Directory Paths
# ==============================================================================
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
TEST_DIR = "data/test"
METADATA_DIR = "data/metadata"
PATCH_DIR = "data/patches"

# ==============================================================================
# Image Dimensions (target size for resizing/processing)
# ==============================================================================
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ==============================================================================
# Gaussian Blur Parameters
# ==============================================================================
BLUR_KERNEL_SIZE = (5, 5)
BLUR_SIGMA = 0

# ==============================================================================
# CLAHE (Contrast Limited Adaptive Histogram Equalization) Parameters
# ==============================================================================
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
USE_CLAHE = True

# ==============================================================================
# Minimum Acceptable Image Dimensions for Validation
# ==============================================================================
MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64

# ==============================================================================
# Phase 2 — ROI and Patch Extraction
# ==============================================================================

# Road ROI as fractional offsets of image dimensions.
# For a front-facing road camera, the road typically appears in the lower portion.
# These are CONFIGURABLE starting defaults — adjust based on actual image content.
# top_frac=0.4 means ROI starts at 40% from the top of the image.
ROI_TOP_FRAC    = 0.4   # fraction of height from top where ROI begins
ROI_BOTTOM_FRAC = 1.0   # fraction of height from top where ROI ends
ROI_LEFT_FRAC   = 0.0   # fraction of width from left where ROI begins
ROI_RIGHT_FRAC  = 1.0   # fraction of width from left where ROI ends

# Patch sizes to experiment with (pixels, square patches)
PATCH_SIZES = [32, 64, 128]
DEFAULT_PATCH_SIZE = 64  # used by default in extraction

# Allow overlapping patches (stride < patch_size)
PATCH_OVERLAP = False
PATCH_STRIDE_FRAC = 0.5  # stride = patch_size * PATCH_STRIDE_FRAC when overlapping

# Minimum valid pixels required in a patch (reject patches with too many black pixels)
MIN_PATCH_VALID_FRAC = 0.5  # at least 50% of pixels must be non-zero


# ==============================================================================
# Phase 3 — Feature Extraction
# ==============================================================================

# Which patch size to use for feature extraction (must match an extracted size)
FEATURE_PATCH_SIZE = 64

# LBP (Local Binary Patterns) settings
LBP_RADIUS = 1
LBP_N_POINTS = 8         # typically 8 * radius
LBP_METHOD = "uniform"   # 'uniform' gives compact histograms

# GLCM (Gray Level Co-occurrence Matrix) settings
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, 0.785398, 1.5708, 2.35619]  # 0, 45, 90, 135 degrees in radians
GLCM_LEVELS = 256

# Canny edge detection thresholds
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Feature output paths
FEATURES_DIR = "data/features"
FEATURES_CSV = "data/features/features_ps64.csv"
SCALED_FEATURES_CSV = "data/features/features_scaled_ps64.csv"
SCALER_PATH = "models/scaler.joblib"


# ==============================================================================
# Phase 4 — PCA & Clustering (K-Means, DBSCAN)
# ==============================================================================
PCA_VARIANCE_RATIO_TARGET = 0.90  # Target cumulative explained variance
PCA_MODEL_PATH = "models/pca_model.joblib"
PCA_REDUCED_CSV = "data/features/features_pca_ps64.csv"

KMEANS_K_RANGE = [2, 3, 4, 5, 6]
KMEANS_DEFAULT_K = 3
KMEANS_MODEL_PATH = "models/kmeans_model.joblib"

DBSCAN_EPS_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
DBSCAN_MIN_SAMPLES_VALUES = [3, 5, 10]
DBSCAN_DEFAULT_EPS = 1.5
DBSCAN_DEFAULT_MIN_SAMPLES = 5
DBSCAN_MODEL_PATH = "models/dbscan_model.joblib"

CLUSTERING_OUTPUT_DIR = "outputs/clustering"


# ==============================================================================
# Phase 5 — Isolation Forest Anomaly Detection
# ==============================================================================
IFOREST_N_ESTIMATORS = 100
IFOREST_CONTAMINATION = 0.10  # Expected proportion of outliers in the dataset
IFOREST_MAX_SAMPLES = "auto"
IFOREST_RANDOM_STATE = 42

IFOREST_CONTAMINATION_EXPERIMENTS = [0.05, 0.10, 0.15, 0.20]
IFOREST_MODEL_PATH = "models/isolation_forest.joblib"
ANOMALY_PREDICTIONS_CSV = "data/features/anomaly_predictions_ps64.csv"
USE_PCA_FOR_ANOMALY = True  # Use orthogonal PCA features to reduce noise/collinearity


# ==============================================================================
# Phase 6 — Anomaly Visualization & Heatmap
# ==============================================================================
# Visual overlay colors (BGR for OpenCV, RGB for Matplotlib)
COLOR_NORMAL_BGR = (0, 200, 0)         # Green
COLOR_MODERATE_BGR = (0, 215, 255)     # Yellow
COLOR_HIGH_BGR = (0, 0, 230)           # Red

COLOR_NORMAL_RGB = (0, 200, 0)
COLOR_MODERATE_RGB = (255, 215, 0)
COLOR_HIGH_RGB = (230, 0, 0)

# Thresholds on normalized anomaly score [0.0, 1.0]
ANOMALY_SCORE_MODERATE = 0.40
ANOMALY_SCORE_HIGH = 0.65

# Blending parameters
OVERLAY_ALPHA = 0.35
HEATMAP_ALPHA = 0.55
HEATMAP_BLUR_SIGMA = 15

# Output directories
OUTPUTS_DIR = "outputs"
OVERLAYS_DIR = "outputs/overlays"
HEATMAPS_DIR = "outputs/heatmaps"
COMPARISONS_DIR = "outputs/comparisons"


# ==============================================================================
# Phase 7 — Pothole Localization & Spatial Grouping
# ==============================================================================
LOCALIZATION_SCORE_THRESHOLD = 0.40  # Normalized score threshold for candidate patches
MORPH_KERNEL_SIZE = (7, 7)
MIN_CANDIDATE_AREA = 256            # Minimum bounding box pixel area (e.g. 16x16)
MAX_CANDIDATE_AREA_FRAC = 0.65       # Max fraction of total image area to prevent false full-screen boxes
NMS_IOU_THRESHOLD = 0.35

LOCALIZED_OUTPUT_DIR = "outputs/localized"
LOCALIZED_CANDIDATES_CSV = "data/metadata/localized_candidates.csv"


# ==============================================================================
# Phase 8 — Visual Severity Estimation
# ==============================================================================
# Transparent severity formula component weights (must sum to 1.0)
SEVERITY_WEIGHT_SIZE     = 0.30
SEVERITY_WEIGHT_ANOMALY  = 0.30
SEVERITY_WEIGHT_EDGE     = 0.20
SEVERITY_WEIGHT_CONTRAST = 0.20

# Categorical thresholds on continuous severity score [0.0, 1.0]
SEVERITY_THRESH_MEDIUM = 0.35
SEVERITY_THRESH_HIGH   = 0.65

# Contrast margin: pixel band surrounding bounding box for background reference
CONTRAST_RING_MARGIN = 16

# Visual severity colors (BGR)
COLOR_SEV_LOW_BGR    = (0, 200, 0)      # Green
COLOR_SEV_MEDIUM_BGR = (0, 165, 255)    # Amber / Orange
COLOR_SEV_HIGH_BGR   = (0, 0, 230)      # Red

SEVERITY_OUTPUT_DIR = "outputs/severity"
SEVERITY_REPORT_CSV = "data/metadata/severity_report.csv"


# ==============================================================================
# Phase 10 — Video Processing, GPS & Geographic Heatmap
# ==============================================================================
VIDEO_SAMPLE_INTERVAL_FRAMES = 10  # Analyze every 10th frame (e.g. ~3 fps from 30 fps video)
VIDEO_TRACKING_IOU_THRESHOLD = 0.25 # IoU threshold to consider detections as the same physical defect
MAX_TRACK_AGE_FRAMES = 30          # Consecutive missed frames before closing a track

DATABASE_PATH = "data/pothole_database.db"
VIDEOS_OUTPUT_DIR = "outputs/videos"
MAPS_OUTPUT_DIR = "outputs/maps"

# Default fallback map center if no GPS provided (e.g. New Delhi / India center or configurable)
DEFAULT_MAP_CENTER = [28.6139, 77.2090]
DEFAULT_MAP_ZOOM = 13


# ==============================================================================
# Advanced Upgrades (Deep Learning, GIS & Live Stream)
# ==============================================================================
REAL_ROAD_SAMPLES_DIR = "data/real_road_samples"
USE_DEEP_FEATURES = False   # Toggle deep visual embeddings vs handcrafted features
DEEP_FEATURE_DIM = 128
AUTOENCODER_MODEL_PATH = "models/road_autoencoder.pth"
GEOJSON_OUTPUT_PATH = "outputs/maps/defect_survey.geojson"


# ==============================================================================
# Government of India Grievance & Complaint Portal Settings
# ==============================================================================
GOV_PORTALS = {
    "CPGRAMS": {
        "name": "CPGRAMS (Ministry of Road Transport & Highways - MoRTH)",
        "url": "https://pgportal.gov.in/",
        "category": "Road Surface Defect & Pothole Hazard",
        "ministry_code": "MORTH",
    },
    "NHAI": {
        "name": "NHAI Rajmargyatra / Sukhad Yatra (National Highways)",
        "url": "https://nhai.gov.in/",
        "category": "National Highway Maintenance & Safety",
        "ministry_code": "NHAI",
    },
    "MERI_SADAK": {
        "name": "Meri Sadak (PMGSY - Rural & State Roads)",
        "url": "https://omms.nic.in/",
        "category": "Rural Road Quality & Potholes",
        "ministry_code": "MORD_PMGSY",
    },
    "LOCAL_PWD": {
        "name": "State PWD / Municipal Corporation Grievance Cell",
        "url": "https://swachhata.co.in/",
        "category": "Municipal Road Maintenance",
        "ministry_code": "MUNICIPAL_PWD",
    },
}

DEFAULT_GOV_PORTAL = "CPGRAMS"
COMPLAINTS_OUTPUT_DIR = "outputs/complaints"
GOV_API_WEBHOOK_URL = None  # Configurable webhook URL for municipal smart-city API

# ==============================================================================
# Autonomous Vehicle Dashcam & Real-time Auto-Reporter Settings
# ==============================================================================
EVIDENCE_OUTPUT_DIR = "outputs/evidence"
DASHCAM_CAMERA_INDEX = 0          # Default USB camera / dashcam index
DASHCAM_SAMPLE_RATE = 5           # Analyze every N frames for vehicle real-time efficiency
DASHCAM_CONFIRMATION_FRAMES = 2   # Require defect seen across K consecutive frames to filter false shadows
DASHCAM_MIN_SEVERITY_AUTO_REPORT = "Medium"  # Minimum severity ("Low", "Medium", "High") to auto-dispatch
DASHCAM_AUTO_DISPATCH = True      # Automatically transmit to government portal without manual click

# ==============================================================================
# Monocular Metric Depth & Volumetric Estimation (3D Pothole Profiling)
# ==============================================================================
CAMERA_HEIGHT_METERS = 1.35            # Dashcam mounting height from pavement (meters)
CAMERA_PITCH_DEG = 10.0                # Camera downward tilt angle (degrees)
CAMERA_HFOV_DEG = 78.0                 # Dashcam horizontal field of view (degrees)
ASPHALT_DENSITY_KG_PER_LITER = 2.4     # Standard compacted hot/cold asphalt mix density
MIN_DEPTH_WARNING_CM = 3.0             # Cavities >= 3 cm flagged as tire hazard

# ==============================================================================
# Night-Vision & Adverse Weather Adaptability
# ==============================================================================
AUTO_NIGHT_VISION = True               # Automatically detects dark/night scenes & enhances
NIGHT_LUMINANCE_THRESHOLD = 75         # Mean image luminance threshold for night mode
PUDDLE_FILTER_ACTIVE = True            # Filter out flat rain puddles vs deep cavities

# ==============================================================================
# In-Cabin ADAS Collision Warning Audio System
# ==============================================================================
ENABLE_ADAS_AUDIO_ALERTS = True        # Spoken/sound audio alerts inside car
ADAS_WARNING_DISTANCE_METERS = 25.0    # Warning distance ahead of vehicle
ADAS_COOLDOWN_SECONDS = 3.0            # Cooldown between spoken voice alerts

# ==============================================================================
# IRC (Indian Roads Congress) Pavement Distress Taxonomy & PCI
# ==============================================================================
IRC_DISTRESS_TYPES = [
    "IRC:Pothole",
    "IRC:Alligator_Cracking",
    "IRC:Linear_Crack",
    "IRC:Sunken_Manhole",
    "IRC:Raveling_Surface",
]
PCI_EXCELLENT_THRESH = 85
PCI_POOR_THRESH = 55

# ==============================================================================
# Fleet Analytics & Pothole Growth Velocity
# ==============================================================================
FLEET_GEO_CLUSTER_RADIUS_METERS = 12.0  # GPS distance to associate repeat drive observations
VELOCITY_ALERT_GROWTH_PERCENT = 30.0    # Growth > 30% triggers rapid degradation alert
CRITICAL_COLLAPSE_DEPTH_CM = 8.0        # Potholes deeper than 8 cm flagged as collapse risk
