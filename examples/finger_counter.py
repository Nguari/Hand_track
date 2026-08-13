"""
finger_counter.py
-------------------
Exemple : compte le nombre de doigts levés en temps réel.

À lancer depuis la racine du projet :
    python -m examples.finger_counter
"""

import cv2

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.drawing import draw_text


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        landmarks = detector.get_landmark_positions(frame)

        if landmarks:
            count = GestureRecognizer.count_fingers_up(landmarks)
            draw_text(frame, str(count), position=(50, 150), scale=3, color=(0, 255, 0))

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
