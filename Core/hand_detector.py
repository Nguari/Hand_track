"""
hand_detector.py
-----------------
Encapsule la logique de détection et de suivi des mains avec MediaPipe.
"""

import cv2
import mediapipe as mp


class HandDetector:
    """Détecte et suit les mains dans une image ou un flux vidéo."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame, draw=True):
        """Traite une frame BGR et dessine les landmarks si demandé."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        """
        Retourne une liste [(id, x, y), ...] des positions des landmarks
        en pixels pour une main donnée (0 = première main détectée).
        """
        landmark_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_index]
                h, w, _ = frame.shape
                for lm_id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((lm_id, cx, cy))
        return landmark_list

    def num_hands_detected(self):
        """Retourne le nombre de mains actuellement détectées."""
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
