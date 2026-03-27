"""Lumina Hands — Real-time hand tracking visualization with 3D object interaction."""

import cv2
import numpy as np
import time
from config import CAM_WIDTH, CAM_HEIGHT, CAM_INDEX, BG_DARKEN, STATUS_TEXT, ELASTIC_ENABLED
from tracker import HandTracker
from effects import draw_glowing_joints, draw_bones, draw_elastic_strings, draw_single_hand_web
from objects3d import OBJECTS, ObjectRenderer
from gestures import GestureTracker
from menu import Menu


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    tracker = HandTracker()
    gesture = GestureTracker()
    menu = Menu()
    obj_renderer = None

    # current mode
    mode = None           # None = menu active, "hands" = strings, "cube"/"pyramid"/etc = 3D
    current_obj = None    # (verts, edges, name)

    prev_time = time.time()
    fps_smooth = 30.0

    OBJ_MAP = {
        "cube": 0,
        "pyramid": 1,
        "diamond": 2,
        "sphere": 3,
    }

    print("Lumina Hands running")
    print("  'q' = quit | 'r' = back to menu | 'ESC' = back to menu")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = tracker.process(rgb, w, h)

        # darken camera feed
        canvas = cv2.multiply(frame, np.array([BG_DARKEN], dtype=np.float64))

        now = time.time()
        dt = max(now - prev_time, 0.001)
        prev_time = now
        fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1

        # --- MENU MODE ---
        if menu.is_active():
            menu.update(hands, w, h)

            # still draw hand joints on top of menu for visual feedback
            if hands:
                draw_glowing_joints(canvas, hands)

            menu.draw(canvas)

            result = menu.get_result()
            if result:
                mode = result
                if result in OBJ_MAP:
                    verts, edges, name = OBJECTS[OBJ_MAP[result]]()
                    current_obj = (verts, edges, name)
                    gesture = GestureTracker()  # reset gesture state
                    if obj_renderer is None or obj_renderer.shape != canvas.shape:
                        obj_renderer = ObjectRenderer(canvas.shape)

        # --- HANDS MODE ---
        elif mode == "hands":
            if ELASTIC_ENABLED and len(hands) == 2:
                draw_elastic_strings(canvas, hands[0], hands[1])
            elif len(hands) == 1:
                draw_single_hand_web(canvas, hands[0])

            if hands:
                draw_bones(canvas, hands)
                draw_glowing_joints(canvas, hands)

            # status text
            text_size = cv2.getTextSize(STATUS_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            tx = (w - text_size[0]) // 2
            cv2.putText(canvas, STATUS_TEXT, (tx, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        # --- 3D OBJECT MODE ---
        elif current_obj is not None:
            verts, edges, name = current_obj

            # update gestures
            controlling = gesture.update(hands, dt)

            # draw the 3D object
            obj_renderer.draw(
                canvas, verts, edges,
                gesture.rot_x, gesture.rot_y, gesture.rot_z,
                gesture.zoom, w // 2, h // 2,
            )

            # draw hands on top
            if hands:
                draw_bones(canvas, hands)
                draw_glowing_joints(canvas, hands)

            # control hints
            obj_label = f"3D {name}"
            cv2.putText(canvas, obj_label, (15, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1, cv2.LINE_AA)

            if controlling:
                cv2.putText(canvas, "CONTROLLING", (15, h - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 100), 1, cv2.LINE_AA)
            elif gesture.auto_rotate:
                cv2.putText(canvas, "AUTO-ROTATE  |  Pinch to control", (15, h - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "Pinch to rotate  |  Open palm for auto", (15, h - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1, cv2.LINE_AA)

            hint = "Two hands = zoom  |  'R' = menu"
            cv2.putText(canvas, hint, (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 120), 1, cv2.LINE_AA)

        # --- HUD ---
        cv2.putText(canvas, f"FPS: {int(fps_smooth)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        if hands and not menu.is_active():
            cv2.putText(canvas, f"Hands: {len(hands)}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Lumina Hands", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') or key == 27:  # 'r' or ESC
            mode = None
            current_obj = None
            menu.reset()

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
