"""Hand gesture recognition for 3D object manipulation."""

import numpy as np
from config import FINGERTIP_IDS


def _dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _center(landmarks: list[tuple[int, int]]) -> tuple[float, float]:
    """Center of all landmarks."""
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    return sum(xs) / len(xs), sum(ys) / len(ys)


class GestureTracker:
    """Tracks hand gestures for rotation and zoom."""

    def __init__(self):
        self.prev_center = None        # for single-hand rotation
        self.prev_pinch_dist = None    # for two-hand zoom
        self.prev_two_center = None    # for two-hand rotation

        # accumulated rotation and zoom
        self.rot_x = 0.3   # slight tilt
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.zoom = 120.0

        # auto-rotate when no hands
        self.auto_rotate = True

        # sensitivity
        self.rot_sensitivity = 0.008
        self.zoom_sensitivity = 0.8

    def is_pinching(self, landmarks: list[tuple[int, int]]) -> bool:
        """Check if thumb and index finger are pinching."""
        if len(landmarks) < 21:
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        return _dist(thumb_tip, index_tip) < 50

    def is_open_palm(self, landmarks: list[tuple[int, int]]) -> bool:
        """Check if hand is open (all fingers extended)."""
        if len(landmarks) < 21:
            return False
        wrist = landmarks[0]
        tips_dist = sum(_dist(landmarks[t], wrist) for t in FINGERTIP_IDS)
        mid_dist = sum(_dist(landmarks[t - 2], wrist) for t in FINGERTIP_IDS)
        return tips_dist > mid_dist * 1.3

    def update(self, hands: list[list[tuple[int, int]]], dt: float):
        """Update rotation/zoom from hand gestures. Returns True if hands are controlling."""

        if len(hands) == 0:
            self.prev_center = None
            self.prev_pinch_dist = None
            self.prev_two_center = None
            # auto rotate
            if self.auto_rotate:
                self.rot_y += 0.6 * dt
                self.rot_x = 0.3 + np.sin(self.rot_y * 0.5) * 0.15
            return False

        if len(hands) == 1:
            hand = hands[0]
            self.prev_pinch_dist = None
            self.prev_two_center = None

            if self.is_pinching(hand):
                self.auto_rotate = False
                cx, cy = _center(hand)
                if self.prev_center is not None:
                    dx = cx - self.prev_center[0]
                    dy = cy - self.prev_center[1]
                    self.rot_y += dx * self.rot_sensitivity
                    self.rot_x += dy * self.rot_sensitivity
                self.prev_center = (cx, cy)
                return True
            else:
                self.prev_center = None
                # open palm = reset auto rotate
                if self.is_open_palm(hand):
                    self.auto_rotate = True
                return False

        if len(hands) >= 2:
            self.auto_rotate = False
            h1, h2 = hands[0], hands[1]
            c1 = _center(h1)
            c2 = _center(h2)

            # zoom: distance between hand centers
            pinch_dist = _dist(c1, c2)
            if self.prev_pinch_dist is not None:
                delta = pinch_dist - self.prev_pinch_dist
                self.zoom += delta * self.zoom_sensitivity
                self.zoom = np.clip(self.zoom, 40, 300)
            self.prev_pinch_dist = pinch_dist

            # rotation: average center movement
            mid = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
            if self.prev_two_center is not None:
                dx = mid[0] - self.prev_two_center[0]
                dy = mid[1] - self.prev_two_center[1]
                self.rot_y += dx * self.rot_sensitivity
                self.rot_x += dy * self.rot_sensitivity
            self.prev_two_center = mid

            return True

        return False
