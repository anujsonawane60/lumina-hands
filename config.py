"""Configuration constants for Lumina Hands."""

# --- Camera ---
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_INDEX = 0

# --- MediaPipe ---
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# --- Glow Rendering ---
JOINT_RADIUS = 6
GLOW_LAYERS = 4          # number of expanding rings per joint
GLOW_EXPANSION = 4       # pixel growth per ring
GLOW_ALPHA_DECAY = 0.45  # opacity multiplier per ring

# --- Mesh / Strings ---
LINE_MAX_THICKNESS = 3
LINE_MIN_ALPHA = 0.15
LINE_MAX_ALPHA = 0.9
MESH_DISTANCE_THRESHOLD = 150  # px — connections beyond this fade out

# --- Color ---
COLOR_MODE = "rainbow"  # "rainbow" | "distance"
RAINBOW_SATURATION = 255
RAINBOW_VALUE = 255

# --- Motion Smoothing ---
SMOOTHING_FACTOR = 0.45  # 0 = no smoothing, 1 = frozen (exponential moving avg)

# --- Trails ---
TRAIL_LENGTH = 10        # number of past frames to keep
TRAIL_ALPHA_DECAY = 0.7  # per-frame opacity multiplier for trail dots
TRAIL_RADIUS = 3

# --- Particles ---
PARTICLES_ENABLED = True
PARTICLE_SPAWN_SPEED_THRESHOLD = 15  # px/frame — minimum fingertip speed to emit
PARTICLE_COUNT_PER_FRAME = 3
PARTICLE_LIFETIME = 20   # frames
PARTICLE_MAX_RADIUS = 4
PARTICLE_DRIFT_SPEED = 2.0

# --- Finger connections (MediaPipe landmark indices) ---
# Each finger's bone chain
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]

# Cross-finger mesh connections (palm + fingertip web)
CROSS_CONNECTIONS = [
    (5, 9), (9, 13), (13, 17),   # palm knuckle bridge
    (1, 5), (0, 17),              # palm edges
    (4, 8), (8, 12), (12, 16), (16, 20),  # fingertip bridge
]

# Fingertip indices (for particles + trails)
FINGERTIP_IDS = [4, 8, 12, 16, 20]
