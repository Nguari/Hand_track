"""
color_picker.py
------------------
Exemple : sélectionne une couleur dans une palette avec le bout de l'index,
et valide le choix avec un pincement pouce-index.

À lancer depuis la racine du projet :
    python -m examples.color_picker
"""

import cv2

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.drawing import draw_text

# Palette : (nom, couleur BGR)
PALETTE = [
    ("Rouge", (0, 0, 255)),
    ("Orange", (0, 140, 255)),
    ("Jaune", (0, 255, 255)),
    ("Vert", (0, 255, 0)),
    ("Cyan", (255, 255, 0)),
    ("Bleu", (255, 0, 0)),
    ("Violet", (255, 0, 160)),
    ("Blanc", (255, 255, 255)),
]

SWATCH_SIZE = 80
SWATCH_MARGIN = 15
SWATCH_Y = 20


def get_swatch_rects(frame_width):
    """Calcule les rectangles (x1, y1, x2, y2) de chaque case de couleur."""
    rects = []
    x = SWATCH_MARGIN
    for name, color in PALETTE:
        rects.append((x, SWATCH_Y, x + SWATCH_SIZE, SWATCH_Y + SWATCH_SIZE, name, color))
        x += SWATCH_SIZE + SWATCH_MARGIN
    return rects


def draw_palette(frame, rects, hovered_index):
    for i, (x1, y1, x2, y2, name, color) in enumerate(rects):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        border_color = (255, 255, 255) if i == hovered_index else (60, 60, 60)
        border_thickness = 4 if i == hovered_index else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)


def find_hovered_swatch(rects, x, y):
    """Retourne l'index de la case survolée par le point (x, y), ou None."""
    for i, (x1, y1, x2, y2, name, color) in enumerate(rects):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=1)

    selected_color = None
    selected_name = None
    was_pinching = False  # pour ne déclencher qu'une fois par pincement

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rects = get_swatch_rects(frame_w)

        frame = detector.find_hands(frame, draw=True)
        landmarks = detector.get_landmark_positions(frame)

        hovered_index = None

        if landmarks:
            # Landmark 8 = bout de l'index
            _, ix, iy = landmarks[8]
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

            hovered_index = find_hovered_swatch(rects, ix, iy)
            is_pinching = GestureRecognizer.is_pinching(landmarks)

            # Valider la sélection seulement au moment où le pincement démarre
            if is_pinching and not was_pinching and hovered_index is not None:
                selected_name = rects[hovered_index][4]
                selected_color = rects[hovered_index][5]

            was_pinching = is_pinching

        draw_palette(frame, rects, hovered_index)

        # Affichage de la couleur sélectionnée
        if selected_color is not None:
            box_y1 = frame_h - 120
            cv2.rectangle(frame, (15, box_y1), (215, frame_h - 20), selected_color, -1)
            cv2.rectangle(frame, (15, box_y1), (215, frame_h - 20), (255, 255, 255), 2)
            b, g, r = selected_color
            draw_text(frame, f"{selected_name}", position=(230, box_y1 + 40), scale=1)
            draw_text(frame, f"RGB({r},{g},{b})", position=(230, box_y1 + 80), scale=0.8)
        else:
            draw_text(frame, "Pince (pouce+index) sur une couleur pour la choisir",
                      position=(15, frame_h - 30), scale=0.7)

        cv2.imshow("Color Picker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Échap
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()