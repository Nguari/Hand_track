"""
gesture_recognizer.py
----------------------
Reconnaissance de gestes simples à partir des landmarks des mains.
"""

import math

# IDs MediaPipe des extrémités des doigts (pouce, index, majeur, annulaire, auriculaire)
FINGER_TIPS = [4, 8, 12, 16, 20]


class GestureRecognizer:
    """Analyse une liste de landmarks pour en déduire des gestes."""

    @staticmethod
    def distance(p1, p2):
        """Distance euclidienne entre deux points (id, x, y)."""
        _, x1, y1 = p1
        _, x2, y2 = p2
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def is_pinching(landmarks, threshold=40):
        """Détecte un pincement pouce-index (id 4 et 8)."""
        if len(landmarks) < 9:
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        return GestureRecognizer.distance(thumb_tip, index_tip) < threshold

    @staticmethod
    def is_three_finger_pinch(landmarks, threshold=45):
        """Détecte un pincement à 3 doigts : pouce (4), index (8), majeur (12) rapprochés."""
        if len(landmarks) < 13:
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        d1 = GestureRecognizer.distance(thumb_tip, index_tip)
        d2 = GestureRecognizer.distance(thumb_tip, middle_tip)
        d3 = GestureRecognizer.distance(index_tip, middle_tip)
        return d1 < threshold and d2 < threshold and d3 < threshold

    @staticmethod
    def fingers_down(landmarks, threshold=45):
        """Détecte une main fermée (tous les doigts repliés).

        La valeur de `threshold` est conservée pour compatibilité avec
        l'API existante, mais la détection se base sur l'état des doigts
        (levés ou repliés), pas sur une distance entre extrémités.
        """
        if len(landmarks) < 21:
            return False

        # Une main fermée corresond à 0 doigts levés
        return GestureRecognizer.count_fingers_up(landmarks) == 0

    @staticmethod
    def count_fingers_up(landmarks):
        """
        Compte le nombre de doigts levés.
        Nécessite les 21 landmarks d'une seule main.
        """
        if len(landmarks) < 21:
            return 0

        fingers_up = []

        # Pouce : comparaison horizontale (dépend de l'orientation de la main)
        if landmarks[4][1] > landmarks[3][1]:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # Les 4 autres doigts : comparaison verticale (tip au-dessus de l'articulation)
        for tip_id in FINGER_TIPS[1:]:
            if landmarks[tip_id][2] < landmarks[tip_id - 2][2]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        return sum(fingers_up)