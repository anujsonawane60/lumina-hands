# Lumina Hands

Real-time hand tracking visualization that transforms your hands into glowing geometric light sculptures using your webcam.

Built with **MediaPipe** for hand landmark detection and **OpenCV** for rendering — runs entirely on CPU.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

## Features

- **21-point hand tracking** — detects up to 2 hands simultaneously
- **Glowing joints** — multi-layered additive-blended circles at each landmark
- **Dynamic mesh connections** — lines between finger bones and cross-connections that react to distance (thickness, opacity, color)
- **Color modes** — rainbow gradient (per landmark) or distance-based (warm = close, cool = far), toggle live with `m`
- **Motion smoothing** — exponential moving average filter eliminates jitter
- **Fingertip trails** — afterglow effect from the last N frames with decaying opacity
- **Particle effects** — particles spawn from fast-moving fingertips, drift and fade out

## Prerequisites

- Python 3.9+
- A webcam

## Installation

```bash
git clone <repo-url>
cd lumina-hands
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Toggle color mode (rainbow / distance) |
| `p` | Toggle particle effects |

## Configuration

All parameters are tunable in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CAM_INDEX` | Webcam device index | `0` |
| `CAM_WIDTH` / `CAM_HEIGHT` | Capture resolution | `1280x720` |
| `MAX_HANDS` | Max hands to detect | `2` |
| `SMOOTHING_FACTOR` | Motion smoothing (0 = none, 1 = frozen) | `0.45` |
| `GLOW_LAYERS` | Number of glow rings per joint | `4` |
| `MESH_DISTANCE_THRESHOLD` | Max distance (px) before connections fade out | `150` |
| `COLOR_MODE` | `"rainbow"` or `"distance"` | `"rainbow"` |
| `TRAIL_LENGTH` | Number of past frames for trail effect | `10` |
| `PARTICLE_LIFETIME` | Particle duration in frames | `20` |

## Project Structure

```
lumina-hands/
├── main.py            # Entry point — webcam loop and rendering pipeline
├── tracker.py         # MediaPipe hand detection with EMA smoothing
├── effects.py         # Glow, mesh, trails, particles, color mapping
├── config.py          # All tunable constants
└── requirements.txt   # Dependencies
```

## Tech Stack

- **[MediaPipe](https://github.com/google/mediapipe)** — Hand landmark detection (21 points per hand)
- **[OpenCV](https://opencv.org/)** — Webcam capture and real-time rendering
- **[NumPy](https://numpy.org/)** — Vector math, interpolation, color mapping
