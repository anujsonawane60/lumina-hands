"""Configuration constants for Lumina Hands."""

# --- Camera ---
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_INDEX = 0

# --- MediaPipe ---
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# --- Joint Glow (warm fairy-light style) ---
JOINT_COLOR_BGR = (60, 180, 255)   # warm golden amber
JOINT_CORE_RADIUS = 5
JOINT_GLOW_RADIUS = 22            # soft outer glow
JOINT_GLOW_BLUR = 31              # Gaussian blur kernel size (must be odd)

# --- Intra-hand bone connections (subtle) ---
BONE_COLOR_BGR = (50, 140, 200)    # subtle warm tone
BONE_THICKNESS = 1
BONE_ALPHA = 0.3

# --- Cross-hand elastic strings ---
ELASTIC_ENABLED = True
ELASTIC_MAX_DISTANCE = 400         # px — strings snap beyond this
ELASTIC_FADE_START = 250           # px — start fading
ELASTIC_THICKNESS_CLOSE = 3
ELASTIC_THICKNESS_FAR = 1
ELASTIC_NUM_SEGMENTS = 30          # curve smoothness
ELASTIC_SAG_FACTOR = 0.08          # catenary sag amount
ELASTIC_GRADIENT_COLORS = [        # HSV hues for gradient along string
    (160, 200, 255),   # pink/magenta
    (140, 180, 255),   # warm pink
    (120, 160, 255),   # light purple
    (100, 200, 255),   # purple
    (80, 220, 255),    # blue-purple
    (60, 240, 255),    # cyan
]

# --- Motion Smoothing ---
SMOOTHING_FACTOR = 0.5   # 0 = no smoothing, 1 = frozen

# --- Finger topology ---
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]

PALM_CONNECTIONS = [
    (5, 9), (9, 13), (13, 17), (1, 5), (0, 17),
]

# Fingertip indices
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Cross-hand connection pairs (fingertip to fingertip)
CROSS_HAND_PAIRS = [
    (4, 4), (8, 8), (12, 12), (16, 16), (20, 20),     # matching tips
    (4, 8), (8, 12), (12, 16), (16, 20),               # offset tips
    (8, 4), (12, 8), (16, 12), (20, 16),               # reverse offset
]

# --- Background ---
BG_DARKEN = 0.25          # how much to darken camera feed (0=black, 1=full)

# --- Status text ---
STATUS_TEXT = "ELASTIC STRINGS ACTIVE  -  STRETCH TO SNAP"
