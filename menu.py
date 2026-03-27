"""On-screen menu system with hand hover selection."""

import cv2
import numpy as np
import time


class MenuItem:
    def __init__(self, label: str, value: str, icon_fn=None):
        self.label = label
        self.value = value
        self.icon_fn = icon_fn  # callable(canvas, cx, cy, size) to draw icon
        self.hover_progress = 0.0  # 0 to 1 — fills up while hovering
        self.selected = False


def _draw_icon_cube(canvas, cx, cy, sz):
    s = sz // 3
    pts_back = np.array([
        [cx - s + 4, cy - s + 4], [cx + s + 4, cy - s + 4],
        [cx + s + 4, cy + s + 4], [cx - s + 4, cy + s + 4],
    ], dtype=np.int32)
    pts_front = np.array([
        [cx - s - 4, cy - s - 4], [cx + s - 4, cy - s - 4],
        [cx + s - 4, cy + s - 4], [cx - s - 4, cy + s - 4],
    ], dtype=np.int32)
    cv2.polylines(canvas, [pts_back], True, (100, 160, 200), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_front], True, (60, 180, 255), 1, cv2.LINE_AA)
    for i in range(4):
        cv2.line(canvas, tuple(pts_front[i]), tuple(pts_back[i]), (80, 170, 230), 1, cv2.LINE_AA)


def _draw_icon_pyramid(canvas, cx, cy, sz):
    s = sz // 3
    apex = (cx, cy - s)
    bl = (cx - s, cy + s)
    br = (cx + s, cy + s)
    bm = (cx, cy + s - 4)
    cv2.line(canvas, apex, bl, (60, 180, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, apex, br, (60, 180, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, bl, br, (60, 180, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, apex, bm, (100, 160, 200), 1, cv2.LINE_AA)
    cv2.line(canvas, bl, bm, (100, 160, 200), 1, cv2.LINE_AA)


def _draw_icon_diamond(canvas, cx, cy, sz):
    s = sz // 3
    pts = [(cx, cy - s - 4), (cx + s, cy), (cx, cy + s + 4), (cx - s, cy)]
    for i in range(4):
        cv2.line(canvas, pts[i], pts[(i + 1) % 4], (60, 180, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, pts[0], pts[2], (100, 160, 200), 1, cv2.LINE_AA)


def _draw_icon_sphere(canvas, cx, cy, sz):
    r = sz // 3
    cv2.ellipse(canvas, (cx, cy), (r, r), 0, 0, 360, (60, 180, 255), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (r, r // 2), 0, 0, 360, (100, 160, 200), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (r // 2, r), 0, 0, 360, (100, 160, 200), 1, cv2.LINE_AA)


def _draw_icon_hands(canvas, cx, cy, sz):
    s = sz // 3
    # simple hand icon — finger lines
    base = (cx, cy + s)
    for i, angle in enumerate([-40, -20, 0, 20, 40]):
        rad = np.radians(angle - 90)
        tip = (int(cx + np.cos(rad) * s * 1.2), int(cy + np.sin(rad) * s * 1.2 - 2))
        cv2.line(canvas, base, tip, (60, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, tip, 2, (200, 240, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  Menu class
# ---------------------------------------------------------------------------

HOVER_TIME = 1.2  # seconds to select


class Menu:
    def __init__(self):
        self.state = "main"  # "main" | "objects" | None (dismissed)
        self.items: list[MenuItem] = []
        self.result = None      # selected value
        self.box_size = 100
        self.gap = 20
        self.last_hover_idx = -1
        self.hover_start = 0.0
        self._build_main()

    def _build_main(self):
        self.items = [
            MenuItem("HAND\nSTRINGS", "hands", _draw_icon_hands),
            MenuItem("3D\nOBJECTS", "objects_menu", _draw_icon_cube),
        ]
        self.state = "main"
        self._reset_hover()

    def _build_objects(self):
        self.items = [
            MenuItem("CUBE", "cube", _draw_icon_cube),
            MenuItem("PYRAMID", "pyramid", _draw_icon_pyramid),
            MenuItem("DIAMOND", "diamond", _draw_icon_diamond),
            MenuItem("SPHERE", "sphere", _draw_icon_sphere),
        ]
        self.state = "objects"
        self._reset_hover()

    def _reset_hover(self):
        for item in self.items:
            item.hover_progress = 0.0
        self.last_hover_idx = -1
        self.hover_start = 0.0

    def is_active(self):
        return self.state is not None

    def get_result(self):
        r = self.result
        self.result = None
        return r

    def reset(self):
        self.result = None
        self._build_main()

    def _layout(self, frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) for each item, centered on screen."""
        n = len(self.items)
        total_w = n * self.box_size + (n - 1) * self.gap
        start_x = (frame_w - total_w) // 2
        cy = frame_h // 2
        rects = []
        for i in range(n):
            x = start_x + i * (self.box_size + self.gap)
            y = cy - self.box_size // 2
            rects.append((x, y, self.box_size, self.box_size))
        return rects

    def update(self, hands: list[list[tuple[int, int]]], frame_w: int, frame_h: int):
        """Check if any fingertip hovers over a menu item."""
        if not self.is_active():
            return

        rects = self._layout(frame_w, frame_h)
        now = time.time()

        # find which item is being hovered (index fingertip = landmark 8)
        hover_idx = -1
        for hand in hands:
            if len(hand) < 21:
                continue
            fx, fy = hand[8]  # index finger tip
            for i, (rx, ry, rw, rh) in enumerate(rects):
                if rx <= fx <= rx + rw and ry <= fy <= ry + rh:
                    hover_idx = i
                    break
            if hover_idx >= 0:
                break

        if hover_idx != self.last_hover_idx:
            # reset all progress
            for item in self.items:
                item.hover_progress = 0.0
            self.last_hover_idx = hover_idx
            self.hover_start = now
        elif hover_idx >= 0:
            elapsed = now - self.hover_start
            progress = min(elapsed / HOVER_TIME, 1.0)
            self.items[hover_idx].hover_progress = progress

            if progress >= 1.0:
                selected_value = self.items[hover_idx].value
                if selected_value == "objects_menu":
                    self._build_objects()
                elif selected_value == "hands":
                    self.result = "hands"
                    self.state = None
                else:
                    self.result = selected_value
                    self.state = None
        else:
            for item in self.items:
                item.hover_progress = max(0, item.hover_progress - 0.03)

    def draw(self, canvas: np.ndarray):
        if not self.is_active():
            return

        h, w = canvas.shape[:2]
        rects = self._layout(w, h)

        # dim background
        overlay = np.zeros_like(canvas)
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 1.0, 0, dst=canvas)

        # title
        title = "SELECT MODE" if self.state == "main" else "SELECT OBJECT"
        tsz = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = (w - tsz[0]) // 2
        ty = rects[0][1] - 40
        cv2.putText(canvas, title, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)

        subtitle = "Hover index finger to select"
        ssz = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(canvas, subtitle, ((w - ssz[0]) // 2, ty + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1, cv2.LINE_AA)

        for i, (rx, ry, rw, rh) in enumerate(rects):
            item = self.items[i]
            prog = item.hover_progress

            # box border color based on hover
            if prog > 0:
                border_color = (
                    int(80 + 120 * prog),
                    int(140 + 100 * prog),
                    int(200 + 55 * prog),
                )
                thickness = 2
            else:
                border_color = (60, 100, 140)
                thickness = 1

            # box background
            bg_alpha = 0.15 + 0.25 * prog
            box_overlay = np.zeros_like(canvas)
            cv2.rectangle(box_overlay, (rx, ry), (rx + rw, ry + rh), border_color, -1)
            cv2.addWeighted(box_overlay, bg_alpha, canvas, 1.0, 0, dst=canvas)

            # progress bar at bottom of box
            if prog > 0:
                bar_w = int(rw * prog)
                cv2.rectangle(canvas, (rx, ry + rh - 4), (rx + bar_w, ry + rh), border_color, -1)

            # border
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border_color, thickness, cv2.LINE_AA)

            # icon
            icon_cy = ry + rh // 2 - 8
            if item.icon_fn:
                item.icon_fn(canvas, rx + rw // 2, icon_cy, rw)

            # label (multi-line)
            lines = item.label.split("\n")
            label_y = ry + rh - 8 - (len(lines) - 1) * 14
            for line in lines:
                lsz = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
                lx = rx + (rw - lsz[0]) // 2
                cv2.putText(canvas, line, (lx, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1, cv2.LINE_AA)
                label_y += 14
