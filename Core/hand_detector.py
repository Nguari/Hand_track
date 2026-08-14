"""
hand_detector.py
-----------------
Encapsule la logique de détection et de suivi des mains avec MediaPipe.

Compatible avec MediaPipe >= 0.10 (nouvelle "Tasks API" / HandLandmarker),
qui a remplacé l'ancienne API mp.solutions.hands.
"""

import os
import urllib.request

import cv2
import mediapipe as mp

# Connexions entre landmarks (squelette de la main), identiques à l'ancienne
# constante mp.solutions.hands.HAND_CONNECTIONS.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # majeur
    (9, 13), (13, 14), (14, 15), (15, 16),   # annulaire
    (13, 17), (17, 18), (18, 19), (19, 20),  # auriculaire
    (0, 17),                                  # base de la paume
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _ensure_model_downloaded():
    """Télécharge le modèle hand_landmarker.task s'il n'est pas déjà présent."""
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle hand_landmarker.task (une seule fois)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé :", MODEL_PATH)


class HandDetector:
    """Détecte et suit les mains dans un flux vidéo (nouvelle Tasks API)."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        _ensure_model_downloaded()

        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.results = None
        self._start_time = cv2.getTickCount()

    def _elapsed_ms(self):
        """Timestamp monotone en ms, requis par le mode VIDEO."""
        now = cv2.getTickCount()
        elapsed = (now - self._start_time) / cv2.getTickFrequency()
        return int(elapsed * 1000)

    def find_hands(self, frame, draw=True):
        """Traite une frame BGR et dessine le squelette de la main si demandé."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.results = self.landmarker.detect_for_video(mp_image, self._elapsed_ms())

        if self.results.hand_landmarks and draw:
            h, w, _ = frame.shape
            for hand_landmarks in self.results.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                for start_id, end_id in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_id], points[end_id], (0, 255, 0), 2)
                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        """
        Retourne une liste [(id, x, y), ...] des positions des landmarks
        en pixels pour une main donnée (0 = première main détectée).
        """
        landmark_list = []
        if self.results and self.results.hand_landmarks:
            if hand_index < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_index]
                h, w, _ = frame.shape
                for lm_id, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((lm_id, cx, cy))
        return landmark_list

    def num_hands_detected(self):
        """Retourne le nombre de mains actuellement détectées."""
        if self.results and self.results.hand_landmarks:
            return len(self.results.hand_landmarks)
        return 0

    def close(self):
        """Libère les ressources du détecteur (à appeler en fin de programme)."""
        self.landmarker.close()