"""Lumina Hands — Real-time hand tracking visualization."""

import cv2
import numpy as np
from config import CAM_WIDTH, CAM_HEIGHT, CAM_INDEX, PARTICLES_ENABLED
from tracker import HandTracker
from effects import (
    draw_glowing_joints, draw_mesh,
    TrailBuffer, ParticleSystem,
)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    tracker = HandTracker()

    # per-hand effects (support up to 2 hands)
    trails = [TrailBuffer(), TrailBuffer()]
    particles = [ParticleSystem(), ParticleSystem()]

    print("Lumina Hands running — press 'q' to quit, 'm' to toggle color mode, 'p' to toggle particles")

    show_particles = PARTICLES_ENABLED

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect hands
        hands = tracker.process(rgb, w, h)

        # darken the camera feed for contrast
        canvas = (frame * 0.3).astype(np.uint8)

        # reusable overlay buffer
        overlay = np.zeros_like(canvas)

        for hand_idx, landmarks in enumerate(hands):
            # mesh connections (draw behind joints)
            draw_mesh(canvas, landmarks)

            # glowing joints
            draw_glowing_joints(canvas, landmarks, overlay)

            # trails
            if hand_idx < len(trails):
                trails[hand_idx].push(landmarks)
                trails[hand_idx].draw(canvas)

            # particles
            if show_particles and hand_idx < len(particles):
                particles[hand_idx].update(landmarks)
                particles[hand_idx].draw(canvas)

        # draw particles even when no hands (let existing ones fade)
        if show_particles:
            for ps in particles:
                if not hands:
                    ps.particles = [p for p in ps.particles if p.life > 0]
                    for p in ps.particles:
                        p.x += p.vx
                        p.y += p.vy
                        p.life -= 1
                ps.draw(canvas) if hands else None

        # FPS counter
        fps_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
        cv2.putText(canvas, fps_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # hand count
        if hands:
            cv2.putText(canvas, f"Hands: {len(hands)}", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Lumina Hands", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            import config
            config.COLOR_MODE = "distance" if config.COLOR_MODE == "rainbow" else "rainbow"
            print(f"Color mode: {config.COLOR_MODE}")
        elif key == ord('p'):
            show_particles = not show_particles
            print(f"Particles: {'ON' if show_particles else 'OFF'}")

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
