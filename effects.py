"""Visual effects: glow, mesh, trails, particles, color mapping."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from config import (
    JOINT_RADIUS, GLOW_LAYERS, GLOW_EXPANSION, GLOW_ALPHA_DECAY,
    LINE_MAX_THICKNESS, LINE_MIN_ALPHA, LINE_MAX_ALPHA,
    MESH_DISTANCE_THRESHOLD, COLOR_MODE,
    RAINBOW_SATURATION, RAINBOW_VALUE,
    TRAIL_LENGTH, TRAIL_ALPHA_DECAY, TRAIL_RADIUS,
    PARTICLE_LIFETIME, PARTICLE_MAX_RADIUS, PARTICLE_DRIFT_SPEED,
    PARTICLE_SPAWN_SPEED_THRESHOLD, PARTICLE_COUNT_PER_FRAME,
    FINGER_CHAINS, CROSS_CONNECTIONS, FINGERTIP_IDS,
)


# ---------------------------------------------------------------------------
#  Color helpers
# ---------------------------------------------------------------------------

def hue_for_landmark(idx: int, total: int = 21) -> int:
    """Map landmark index to a hue value (0-179 for OpenCV HSV)."""
    return int((idx / total) * 179)


def hue_for_distance(dist: float, max_dist: float = MESH_DISTANCE_THRESHOLD) -> int:
    """Map distance to hue: close = red/warm (0), far = blue/cool (120)."""
    ratio = min(dist / max_dist, 1.0)
    return int(ratio * 120)


def hsv_to_bgr(h: int, s: int = RAINBOW_SATURATION, v: int = RAINBOW_VALUE):
    """Convert a single HSV pixel to BGR tuple."""
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def color_for_landmark(idx: int):
    if COLOR_MODE == "rainbow":
        return hsv_to_bgr(hue_for_landmark(idx))
    return hsv_to_bgr(90)  # default teal


def color_for_connection(dist: float):
    if COLOR_MODE == "distance":
        return hsv_to_bgr(hue_for_distance(dist))
    return None  # caller will blend endpoint colors


# ---------------------------------------------------------------------------
#  Overlay blending
# ---------------------------------------------------------------------------

def blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float):
    """Additive-blend overlay onto base in-place."""
    if alpha <= 0:
        return
    cv2.addWeighted(overlay, alpha, base, 1.0, 0, dst=base)


# ---------------------------------------------------------------------------
#  Glow joints
# ---------------------------------------------------------------------------

def draw_glowing_joints(canvas: np.ndarray, landmarks: list[tuple[int, int]], overlay: np.ndarray):
    """Draw multi-layered glow circles at each landmark."""
    for idx, (x, y) in enumerate(landmarks):
        color = color_for_landmark(idx)
        # core dot
        cv2.circle(canvas, (x, y), JOINT_RADIUS, color, -1, cv2.LINE_AA)
        # glow rings on overlay
        for ring in range(1, GLOW_LAYERS + 1):
            radius = JOINT_RADIUS + ring * GLOW_EXPANSION
            alpha = GLOW_ALPHA_DECAY ** ring
            overlay[:] = 0
            cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
            blend_overlay(canvas, overlay, alpha)


# ---------------------------------------------------------------------------
#  Mesh / string connections
# ---------------------------------------------------------------------------

def _draw_line(canvas: np.ndarray, p1, p2, color, alpha, thickness):
    overlay = np.zeros_like(canvas)
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    blend_overlay(canvas, overlay, alpha)


def draw_mesh(canvas: np.ndarray, landmarks: list[tuple[int, int]]):
    """Draw finger-chain and cross-connection lines with distance-based styling."""
    pairs: list[tuple[int, int]] = []

    # finger bone chains
    for chain in FINGER_CHAINS:
        for i in range(len(chain) - 1):
            pairs.append((chain[i], chain[i + 1]))

    # cross connections
    pairs.extend(CROSS_CONNECTIONS)

    for i, j in pairs:
        if i >= len(landmarks) or j >= len(landmarks):
            continue
        p1, p2 = landmarks[i], landmarks[j]
        dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # alpha fades with distance
        ratio = min(dist / MESH_DISTANCE_THRESHOLD, 1.0)
        alpha = LINE_MAX_ALPHA - ratio * (LINE_MAX_ALPHA - LINE_MIN_ALPHA)

        # thickness shrinks with distance
        thickness = max(1, int(LINE_MAX_THICKNESS * (1.0 - ratio * 0.7)))

        # color
        c = color_for_connection(dist)
        if c is None:
            # blend the two endpoint colors
            c1 = np.array(color_for_landmark(i), dtype=np.float32)
            c2 = np.array(color_for_landmark(j), dtype=np.float32)
            c = tuple(int(v) for v in ((c1 + c2) / 2))

        _draw_line(canvas, p1, p2, c, alpha, thickness)


# ---------------------------------------------------------------------------
#  Trailing effect
# ---------------------------------------------------------------------------

class TrailBuffer:
    def __init__(self):
        self.history: list[list[tuple[int, int]]] = []  # newest first

    def push(self, landmarks: list[tuple[int, int]]):
        self.history.insert(0, list(landmarks))
        if len(self.history) > TRAIL_LENGTH:
            self.history.pop()

    def draw(self, canvas: np.ndarray):
        overlay = np.zeros_like(canvas)
        for age, frame_lms in enumerate(self.history):
            if age == 0:
                continue  # current frame drawn separately
            alpha = TRAIL_ALPHA_DECAY ** age
            for idx in FINGERTIP_IDS:
                if idx >= len(frame_lms):
                    continue
                x, y = frame_lms[idx]
                color = color_for_landmark(idx)
                overlay[:] = 0
                cv2.circle(overlay, (x, y), TRAIL_RADIUS, color, -1, cv2.LINE_AA)
                blend_overlay(canvas, overlay, alpha * 0.6)


# ---------------------------------------------------------------------------
#  Particle system
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    max_life: int
    color: tuple


class ParticleSystem:
    def __init__(self):
        self.particles: list[Particle] = []
        self.prev_tips: dict[int, tuple[int, int]] = {}

    def update(self, landmarks: list[tuple[int, int]]):
        # spawn from fast-moving fingertips
        for tip_id in FINGERTIP_IDS:
            if tip_id >= len(landmarks):
                continue
            cx, cy = landmarks[tip_id]
            if tip_id in self.prev_tips:
                px, py = self.prev_tips[tip_id]
                speed = np.hypot(cx - px, cy - py)
                if speed > PARTICLE_SPAWN_SPEED_THRESHOLD:
                    for _ in range(PARTICLE_COUNT_PER_FRAME):
                        angle = np.random.uniform(0, 2 * np.pi)
                        spd = np.random.uniform(0.5, PARTICLE_DRIFT_SPEED)
                        self.particles.append(Particle(
                            x=float(cx), y=float(cy),
                            vx=np.cos(angle) * spd,
                            vy=np.sin(angle) * spd,
                            life=PARTICLE_LIFETIME,
                            max_life=PARTICLE_LIFETIME,
                            color=color_for_landmark(tip_id),
                        ))
            self.prev_tips[tip_id] = (cx, cy)

        # tick existing particles
        alive = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    def draw(self, canvas: np.ndarray):
        for p in self.particles:
            ratio = p.life / p.max_life
            radius = max(1, int(PARTICLE_MAX_RADIUS * ratio))
            alpha = ratio * 0.7
            overlay = np.zeros_like(canvas)
            cv2.circle(overlay, (int(p.x), int(p.y)), radius, p.color, -1, cv2.LINE_AA)
            blend_overlay(canvas, overlay, alpha)
