"""
main.py
--------
Point d'entrée de l'application de hand tracking.
Affiche le flux webcam avec les landmarks des mains et le nombre de doigts levés.
"""

import cv2

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.drawing import FPSCounter, draw_text


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=2)
    fps_counter = FPSCounter()

    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # effet miroir, plus naturel
        frame = detector.find_hands(frame)

        landmarks = detector.get_landmark_positions(frame)
        if landmarks:
            fingers = GestureRecognizer.count_fingers_up(landmarks)
            draw_text(frame, f"Doigts leves: {fingers}", position=(10, 80))

            if GestureRecognizer.is_pinching(landmarks):
                draw_text(frame, "Pincement detecte", position=(10, 120), color=(0, 255, 255))

        fps_counter.update_and_draw(frame)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # touche Échap pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
