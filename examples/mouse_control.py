"""
mouse_control.py
------------------
Exemple : contrôle le curseur de la souris avec le bout de l'index,
et effectue un clic lors d'un pincement pouce-index.

Nécessite en plus : pip install pyautogui

À lancer depuis la racine du projet :
    python -m examples.mouse_control
"""

import cv2
import pyautogui

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.smoothing import PositionSmoother

# Sécurité pyautogui : déplacer la souris dans un coin arrête le script
pyautogui.FAILSAFE = True

SCREEN_W, SCREEN_H = pyautogui.size()
FRAME_REDUCTION = 100  # marge pour ne pas avoir à toucher les bords de la webcam


def main():
    cap = cv2.VideoCapture(0)
    cam_w, cam_h = int(cap.get(3)), int(cap.get(4))

    detector = HandDetector(max_num_hands=1)
    smoother = PositionSmoother(smoothing_factor=0.6)

    clicking = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        landmarks = detector.get_landmark_positions(frame)

        if landmarks:
            # Landmark 8 = bout de l'index
            _, x, y = landmarks[8]

            # Convertir la position webcam -> position écran
            screen_x = int(
                (x - FRAME_REDUCTION) / (cam_w - 2 * FRAME_REDUCTION) * SCREEN_W
            )
            screen_y = int(
                (y - FRAME_REDUCTION) / (cam_h - 2 * FRAME_REDUCTION) * SCREEN_H
            )
            screen_x, screen_y = smoother.update(screen_x, screen_y)

            # Limiter aux bords de l'écran
            screen_x = max(0, min(SCREEN_W - 1, screen_x))
            screen_y = max(0, min(SCREEN_H - 1, screen_y))

            pyautogui.moveTo(screen_x, screen_y)

            if GestureRecognizer.is_pinching(landmarks):
                if not clicking:
                    pyautogui.click()
                    clicking = True
            else:
                clicking = False

        cv2.imshow("Mouse Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
