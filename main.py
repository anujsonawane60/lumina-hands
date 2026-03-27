"""Lumina Hands — Real-time hand tracking visualization."""

import cv2
import numpy as np
import time
from config import CAM_WIDTH, CAM_HEIGHT, CAM_INDEX, BG_DARKEN, STATUS_TEXT, ELASTIC_ENABLED
from tracker import HandTracker
from effects import draw_glowing_joints, draw_bones, draw_elastic_strings, draw_single_hand_web


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    tracker = HandTracker()
    prev_time = time.time()
    fps_smooth = 30.0

    print("Lumina Hands running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = tracker.process(rgb, w, h)

        # darken camera feed (in-place with uint8 multiply)
        canvas = cv2.multiply(frame, np.array([BG_DARKEN], dtype=np.float64))

        # --- draw layers (back to front) ---

        # 1. elastic strings or single-hand web
        if ELASTIC_ENABLED and len(hands) == 2:
            draw_elastic_strings(canvas, hands[0], hands[1])
        elif len(hands) == 1:
            draw_single_hand_web(canvas, hands[0])

        # 2. bone connections (all hands batched)
        if hands:
            draw_bones(canvas, hands)

        # 3. glowing joints (all hands batched — one blur pass)
        if hands:
            draw_glowing_joints(canvas, hands)

        # FPS (smoothed)
        now = time.time()
        dt = max(now - prev_time, 0.001)
        prev_time = now
        fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1
        cv2.putText(canvas, f"FPS: {int(fps_smooth)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        if hands:
            cv2.putText(canvas, f"Hands: {len(hands)}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # status bar
        text_size = cv2.getTextSize(STATUS_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(canvas, STATUS_TEXT, (tx, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow("Lumina Hands", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
