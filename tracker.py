"""Hand tracking with MediaPipe Tasks API + motion smoothing."""

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from config import (
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, SMOOTHING_FACTOR,
)


class HandTracker:
    def __init__(self):
        import os
        import urllib.request
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("Downloading hand_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=TRACKING_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # smoothing state — use numpy arrays for speed
        self._smooth: dict[int, np.ndarray] = {}  # hand_idx -> (21, 2) array
        self._frame_ts = 0
        self._alpha = SMOOTHING_FACTOR
        self._beta = 1.0 - SMOOTHING_FACTOR

    def process(self, frame_rgb: np.ndarray, frame_w: int, frame_h: int) -> list[list[tuple[int, int]]]:
        """Return list of hands, each hand is a list of 21 (x, y) pixel coords."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33
        results = self.detector.detect_for_video(mp_image, self._frame_ts)

        if not results.hand_landmarks:
            self._smooth.clear()
            return []

        all_hands: list[list[tuple[int, int]]] = []

        for hand_idx, hand_lms in enumerate(results.hand_landmarks):
            # vectorized extraction
            raw = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in hand_lms], dtype=np.float64)

            # EMA smoothing (vectorized)
            if hand_idx not in self._smooth or self._smooth[hand_idx].shape != raw.shape:
                self._smooth[hand_idx] = raw.copy()
            else:
                np.multiply(self._smooth[hand_idx], self._alpha, out=self._smooth[hand_idx])
                self._smooth[hand_idx] += self._beta * raw

            # convert to int tuples
            pts = self._smooth[hand_idx].astype(np.int32)
            landmarks = [tuple(pt) for pt in pts]
            all_hands.append(landmarks)

        return all_hands

    def release(self):
        self.detector.close()
