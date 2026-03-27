"""Visual effects — optimized for real-time performance."""

import cv2
import numpy as np
from config import (
    JOINT_COLOR_BGR, JOINT_CORE_RADIUS, JOINT_GLOW_RADIUS, JOINT_GLOW_BLUR,
    BONE_COLOR_BGR, BONE_THICKNESS, BONE_ALPHA,
    ELASTIC_MAX_DISTANCE, ELASTIC_FADE_START,
    ELASTIC_THICKNESS_CLOSE, ELASTIC_THICKNESS_FAR,
    ELASTIC_NUM_SEGMENTS, ELASTIC_SAG_FACTOR, ELASTIC_GRADIENT_COLORS,
    FINGER_CHAINS, PALM_CONNECTIONS, CROSS_HAND_PAIRS,
)

# ---------------------------------------------------------------------------
#  Pre-compute gradient LUT (avoid per-frame HSV conversion)
# ---------------------------------------------------------------------------

def _build_gradient_lut(n: int) -> list[tuple]:
    """Pre-compute n BGR colors along the gradient."""
    colors = ELASTIC_GRADIENT_COLORS
    nc = len(colors) - 1
    lut = []
    for i in range(n):
        t = i / max(n - 1, 1)
        idx = t * nc
        j = min(int(idx), nc - 1)
        frac = idx - j
        h = int(colors[j][0] + (colors[j + 1][0] - colors[j][0]) * frac)
        s = int(colors[j][1] + (colors[j + 1][1] - colors[j][1]) * frac)
        v = int(colors[j][2] + (colors[j + 1][2] - colors[j][2]) * frac)
        pixel = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
        lut.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return lut

_GRADIENT_LUT = _build_gradient_lut(ELASTIC_NUM_SEGMENTS)
_GRADIENT_LUT_20 = _build_gradient_lut(20)

# Pre-compute bone pairs
_BONE_PAIRS = []
for _chain in FINGER_CHAINS:
    for _i in range(len(_chain) - 1):
        _BONE_PAIRS.append((_chain[_i], _chain[_i + 1]))
_BONE_PAIRS.extend(PALM_CONNECTIONS)


# ---------------------------------------------------------------------------
#  Reusable buffers (allocated once, reused every frame)
# ---------------------------------------------------------------------------

class _Buffers:
    """Lazy-initialized reusable buffers to avoid per-frame allocations."""
    glow_layer = None
    overlay = None
    string_layer = None
    shape = None

    @classmethod
    def get(cls, shape):
        if cls.shape != shape:
            cls.glow_layer = np.zeros(shape, dtype=np.uint8)
            cls.overlay = np.zeros(shape, dtype=np.uint8)
            cls.string_layer = np.zeros(shape, dtype=np.uint8)
            cls.shape = shape
        return cls


# ---------------------------------------------------------------------------
#  Glowing joints
# ---------------------------------------------------------------------------

def draw_glowing_joints(canvas: np.ndarray, all_landmarks: list[list[tuple[int, int]]]):
    """Draw warm glowing dots for all hands in one pass."""
    buf = _Buffers.get(canvas.shape)
    glow = buf.glow_layer
    glow[:] = 0

    # draw all glow circles onto one layer
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(glow, (x, y), JOINT_GLOW_RADIUS, JOINT_COLOR_BGR, -1, cv2.LINE_AA)

    # single blur for all joints
    cv2.GaussianBlur(glow, (JOINT_GLOW_BLUR, JOINT_GLOW_BLUR), 0, dst=glow)
    cv2.add(canvas, glow, dst=canvas)

    # bright cores directly on canvas (no overlay needed)
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(canvas, (x, y), JOINT_CORE_RADIUS, (200, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), JOINT_CORE_RADIUS + 2, JOINT_COLOR_BGR, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  Bones (all hands batched)
# ---------------------------------------------------------------------------

def draw_bones(canvas: np.ndarray, all_landmarks: list[list[tuple[int, int]]]):
    """Draw subtle bone connections for all hands in one pass."""
    buf = _Buffers.get(canvas.shape)
    ov = buf.overlay
    ov[:] = 0

    for landmarks in all_landmarks:
        n = len(landmarks)
        for i, j in _BONE_PAIRS:
            if i < n and j < n:
                cv2.line(ov, landmarks[i], landmarks[j], BONE_COLOR_BGR, BONE_THICKNESS, cv2.LINE_AA)

    cv2.addWeighted(ov, BONE_ALPHA, canvas, 1.0, 0, dst=canvas)


# ---------------------------------------------------------------------------
#  Elastic strings (optimized — one layer, one blur)
# ---------------------------------------------------------------------------

def _catenary_points_batch(p1, p2, num_seg: int, sag_factor: float):
    """Fast catenary using numpy vectorized ops."""
    t = np.linspace(0, 1, num_seg + 1)
    x = p1[0] + (p2[0] - p1[0]) * t
    y = p1[1] + (p2[1] - p1[1]) * t
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    y += dist * sag_factor * 4 * t * (1 - t)
    return np.column_stack((x, y)).astype(np.int32)


def draw_elastic_strings(canvas: np.ndarray, hand1: list[tuple[int, int]], hand2: list[tuple[int, int]]):
    """Draw all cross-hand strings onto one layer with one glow pass."""
    buf = _Buffers.get(canvas.shape)
    sl = buf.string_layer
    sl[:] = 0

    n1, n2 = len(hand1), len(hand2)
    lut = _GRADIENT_LUT
    num_seg = ELASTIC_NUM_SEGMENTS

    for idx1, idx2 in CROSS_HAND_PAIRS:
        if idx1 >= n1 or idx2 >= n2:
            continue

        p1 = hand1[idx1]
        p2 = hand2[idx2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > ELASTIC_MAX_DISTANCE:
            continue

        dist_ratio = min(dist / ELASTIC_MAX_DISTANCE, 1.0)
        thickness = max(
            ELASTIC_THICKNESS_FAR,
            int(ELASTIC_THICKNESS_CLOSE * (1.0 - dist_ratio) + ELASTIC_THICKNESS_FAR * dist_ratio)
        )

        pts = _catenary_points_batch(p1, p2, num_seg, ELASTIC_SAG_FACTOR)

        for i in range(num_seg):
            cv2.line(sl, tuple(pts[i]), tuple(pts[i + 1]), lut[i], thickness, cv2.LINE_AA)

    # single glow blur for all strings
    glow = cv2.GaussianBlur(sl, (7, 7), 0)
    cv2.add(sl, glow, dst=sl)

    # blend with distance-aware alpha
    cv2.addWeighted(sl, 0.8, canvas, 1.0, 0, dst=canvas)


# ---------------------------------------------------------------------------
#  Single-hand web
# ---------------------------------------------------------------------------

SINGLE_HAND_WEB = [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (4, 12), (8, 16), (12, 20),
    (4, 20),
]


def draw_single_hand_web(canvas: np.ndarray, landmarks: list[tuple[int, int]]):
    """Draw web connections within a single hand."""
    buf = _Buffers.get(canvas.shape)
    sl = buf.string_layer
    sl[:] = 0

    n = len(landmarks)
    lut = _GRADIENT_LUT_20

    for idx1, idx2 in SINGLE_HAND_WEB:
        if idx1 >= n or idx2 >= n:
            continue

        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        if dist > ELASTIC_MAX_DISTANCE:
            continue

        dist_ratio = min(dist / 250, 1.0)
        thickness = max(1, int(2 * (1.0 - dist_ratio) + 1))

        pts = _catenary_points_batch(p1, p2, 20, ELASTIC_SAG_FACTOR * 0.5)

        for i in range(20):
            cv2.line(sl, tuple(pts[i]), tuple(pts[i + 1]), lut[i], thickness, cv2.LINE_AA)

    glow = cv2.GaussianBlur(sl, (5, 5), 0)
    cv2.add(sl, glow, dst=sl)
    cv2.addWeighted(sl, 0.55, canvas, 1.0, 0, dst=canvas)
