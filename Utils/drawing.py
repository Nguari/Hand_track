"""
drawing.py
-----------
Fonctions d'affichage (FPS, texte, overlays) pour l'application.
"""

import cv2
import time


class FPSCounter:
    """Calcule et affiche les FPS sur la frame."""

    def __init__(self):
        self.prev_time = time.time()

    def update_and_draw(self, frame, position=(10, 40)):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if current_time != self.prev_time else 0
        self.prev_time = current_time

        cv2.putText(
            frame, f"FPS: {int(fps)}", position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        return frame


def draw_text(frame, text, position=(10, 80), color=(255, 255, 255), scale=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return frame
