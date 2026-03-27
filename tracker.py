"""Hand tracking with MediaPipe + motion smoothing."""

import mediapipe as mp
import numpy as np
from config import (
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, SMOOTHING_FACTOR,
)

mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        # smoothed landmarks per hand index: {hand_idx: [(x,y), ...]}
        self._smooth: dict[int, list[np.ndarray]] = {}

    def process(self, frame_rgb, frame_w: int, frame_h: int) -> list[list[tuple[int, int]]]:
        """Return list of hands, each hand is a list of 21 (x, y) pixel coords."""
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            self._smooth.clear()
            return []

        all_hands: list[list[tuple[int, int]]] = []

        for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
            raw = []
            for lm in hand_lms.landmark:
                raw.append(np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float64))

            # exponential moving average smoothing
            if hand_idx not in self._smooth or len(self._smooth[hand_idx]) != len(raw):
                self._smooth[hand_idx] = list(raw)
            else:
                for i in range(len(raw)):
                    self._smooth[hand_idx][i] = (
                        SMOOTHING_FACTOR * self._smooth[hand_idx][i]
                        + (1 - SMOOTHING_FACTOR) * raw[i]
                    )

            landmarks = [(int(pt[0]), int(pt[1])) for pt in self._smooth[hand_idx]]
            all_hands.append(landmarks)

        return all_hands

    def release(self):
        self.hands.close()
