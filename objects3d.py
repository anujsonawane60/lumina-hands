"""3D wireframe objects with projection and glow rendering."""

import cv2
import numpy as np
from config import JOINT_COLOR_BGR, JOINT_GLOW_BLUR


# ---------------------------------------------------------------------------
#  3D Object definitions (vertices + edges)
# ---------------------------------------------------------------------------

def _cube():
    s = 1.0
    verts = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float64)
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # back face
        (4,5),(5,6),(6,7),(7,4),  # front face
        (0,4),(1,5),(2,6),(3,7),  # connectors
    ]
    return verts, edges, "CUBE"


def _pyramid():
    s = 1.2
    h = 1.5
    verts = np.array([
        [-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s],  # base
        [0, -h, 0],  # apex
    ], dtype=np.float64)
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # base
        (0,4),(1,4),(2,4),(3,4),  # sides
    ]
    return verts, edges, "PYRAMID"


def _diamond():
    s = 1.0
    h = 1.5
    verts = np.array([
        [-s, 0, 0], [s, 0, 0], [0, 0, -s], [0, 0, s],  # middle ring
        [0, -h, 0],  # top
        [0, h, 0],   # bottom
    ], dtype=np.float64)
    edges = [
        (0,2),(2,1),(1,3),(3,0),  # ring
        (0,4),(1,4),(2,4),(3,4),  # top
        (0,5),(1,5),(2,5),(3,5),  # bottom
    ]
    return verts, edges, "DIAMOND"


def _icosphere():
    """Simplified icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    s = 0.8
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64) * s
    edges = [
        (0,11),(0,5),(0,1),(0,7),(0,10),
        (1,5),(5,11),(11,10),(10,7),(7,1),
        (3,9),(3,4),(3,2),(3,6),(3,8),
        (9,4),(4,2),(2,6),(6,8),(8,9),
        (1,8),(1,9),(5,9),(5,4),(11,4),
        (11,2),(10,2),(10,6),(7,6),(7,8),
    ]
    return verts, edges, "SPHERE"


OBJECTS = [_cube, _pyramid, _diamond, _icosphere]


# ---------------------------------------------------------------------------
#  Projection + Rotation
# ---------------------------------------------------------------------------

def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Combined rotation matrix from Euler angles (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def project(verts: np.ndarray, rx: float, ry: float, rz: float,
            scale: float, cx: int, cy: int) -> np.ndarray:
    """Rotate and project 3D vertices to 2D screen coords."""
    R = rotation_matrix(rx, ry, rz)
    rotated = verts @ R.T

    # simple perspective projection
    fov = 4.0
    z_offset = 5.0
    z = rotated[:, 2] + z_offset
    z = np.clip(z, 0.1, None)

    screen_x = (rotated[:, 0] * fov / z * scale + cx).astype(np.int32)
    screen_y = (rotated[:, 1] * fov / z * scale + cy).astype(np.int32)

    return np.column_stack((screen_x, screen_y)), z


# ---------------------------------------------------------------------------
#  Gradient color for edges
# ---------------------------------------------------------------------------

_EDGE_COLORS_HSV = [
    (160, 200, 255), (140, 180, 255), (120, 160, 255),
    (100, 200, 255), (80, 220, 255), (60, 240, 255),
]

_EDGE_COLORS_BGR = []
for _h, _s, _v in _EDGE_COLORS_HSV:
    _px = np.uint8([[[_h, _s, _v]]])
    _bgr = cv2.cvtColor(_px, cv2.COLOR_HSV2BGR)[0][0]
    _EDGE_COLORS_BGR.append((int(_bgr[0]), int(_bgr[1]), int(_bgr[2])))


# ---------------------------------------------------------------------------
#  Render
# ---------------------------------------------------------------------------

class ObjectRenderer:
    """Renders a 3D wireframe object with glow."""

    def __init__(self, shape: tuple):
        self.glow_buf = np.zeros(shape, dtype=np.uint8)
        self.shape = shape

    def _ensure_buf(self, shape):
        if self.shape != shape:
            self.glow_buf = np.zeros(shape, dtype=np.uint8)
            self.shape = shape

    def draw(self, canvas: np.ndarray, verts: np.ndarray, edges: list,
             rx: float, ry: float, rz: float, scale: float,
             center_x: int, center_y: int):
        self._ensure_buf(canvas.shape)
        glow = self.glow_buf
        glow[:] = 0

        pts2d, depths = project(verts, rx, ry, rz, scale, center_x, center_y)

        # sort edges by average depth (back to front)
        edge_depths = [(e, (depths[e[0]] + depths[e[1]]) / 2) for e in edges]
        edge_depths.sort(key=lambda x: -x[1])

        for idx, (edge, avg_depth) in enumerate(edge_depths):
            i, j = edge
            color = _EDGE_COLORS_BGR[idx % len(_EDGE_COLORS_BGR)]

            # depth-based alpha: farther = dimmer
            alpha_depth = np.clip(1.0 - (avg_depth - 3.5) * 0.15, 0.3, 1.0)

            p1 = tuple(pts2d[i])
            p2 = tuple(pts2d[j])

            # draw on glow layer
            cv2.line(glow, p1, p2, color, 2, cv2.LINE_AA)

        # vertex dots on glow layer
        for i, pt in enumerate(pts2d):
            cv2.circle(glow, tuple(pt), 6, JOINT_COLOR_BGR, -1, cv2.LINE_AA)

        # blur for glow
        blurred = cv2.GaussianBlur(glow, (JOINT_GLOW_BLUR, JOINT_GLOW_BLUR), 0)
        cv2.add(canvas, blurred, dst=canvas)
        cv2.add(canvas, glow, dst=canvas)

        # bright vertex cores on top
        for pt in pts2d:
            cv2.circle(canvas, tuple(pt), 3, (200, 240, 255), -1, cv2.LINE_AA)
