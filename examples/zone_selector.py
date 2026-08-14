"""
zone_selector.py
-------------------
Exemple : présente tes deux mains à la webcam et pince pouce-index sur
CHAQUE main pour définir une zone rectangulaire. La sélection démarre au
premier pincement simultané, puis suit la tendance de vos mains même si
vous relâchez le pincement. Un second pincement simultané fige la zone.

Touches :
    S      -> sauvegarde la zone sélectionnée comme image (crop_N.png)
    Échap  -> quitter

À lancer depuis la racine du projet :
    python -m examples.zone_selector
"""

import cv2

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.drawing import draw_text, show_countdown_on_frame, apply_color_filter_to_rect, draw_intensity_bar


# (nom affiché, ratio largeur/hauteur ; None = format libre)
ASPECT_RATIOS = [
    ("Libre", None),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("4:3", 4 / 3),
    ("1:1", 1 / 1),
]

# Filtres disponibles pour la zone
FILTERS = ["normal", "gray", "invert", "sepia", "saturated", "high_contrast"]

# Filtres disponibles pour la zone
FILTERS = ["normal", "gray", "invert", "sepia", "saturated", "high_contrast"]


def get_pinch_point(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]
    _, x1, y1 = thumb
    _, x2, y2 = index
    return ((x1 + x2) // 2, (y1 + y2) // 2)


    filter_index = 0
    filter_intensity = 1.0
    was_filter_pinching = False
    prev_filter_y = None
def normalize_rect(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def apply_aspect_ratio(rect, ratio, frame_w, frame_h):
    if ratio is None:
        return rect
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = x2 - x1
    new_width = width
    new_height = new_width / ratio
    new_x1 = int(cx - new_width / 2)
    new_x2 = int(cx + new_width / 2)
    new_y1 = int(cy - new_height / 2)
    new_y2 = int(cy + new_height / 2)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(frame_w, new_x2)
    new_y2 = min(frame_h, new_y2)
    return new_x1, new_y1, new_x2, new_y2


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=2)

    selecting = False
    confirmed_rect = None
    save_counter = 0
    ratio_index = 0
    filter_index = 0
    was_filter_pinching = False
    was_three_pinch = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        frame = detector.find_hands(frame, draw=True)

        landmarks_0 = detector.get_landmark_positions(frame, hand_index=0)
        landmarks_1 = detector.get_landmark_positions(frame, hand_index=1)
        num_hands = detector.num_hands_detected()

        live_rect = None
        three_pinch_now = False

        if num_hands >= 2 and landmarks_0 and landmarks_1:
            pinching_0 = GestureRecognizer.is_pinching(landmarks_0)
            pinching_1 = GestureRecognizer.is_pinching(landmarks_1)

            point_0 = get_pinch_point(landmarks_0)
            point_1 = get_pinch_point(landmarks_1)

            cv2.circle(frame, point_0, 12, (0, 255, 0) if pinching_0 else (0, 0, 255), -1)
            cv2.circle(frame, point_1, 12, (0, 255, 0) if pinching_1 else (0, 0, 255), -1)

            # toggle : démarrer la sélection au premier pincement double,
            # confirmer/figer au pincement double suivant
            if pinching_0 and pinching_1:
                if not selecting:
                    selecting = True
                    live_rect = normalize_rect(point_0, point_1)
                else:
                    confirmed_rect = normalize_rect(point_0, point_1)
                    selecting = False
            else:
                if selecting:
                    # suivre la tendance des mains même sans pincement
                    live_rect = normalize_rect(point_0, point_1)

            # Single-hand pinch while selecting -> cycle color filter
            single_hand_pinching = False
            if selecting:
                single_hand_pinching = (pinching_0 != pinching_1)

            if single_hand_pinching and not was_filter_pinching:
                filter_index = (filter_index + 1) % len(FILTERS)
                was_filter_pinching = True
            if not single_hand_pinching:
            frame = apply_color_filter_to_rect(frame, live_rect, FILTERS[filter_index], intensity=filter_intensity)
            frame = draw_intensity_bar(frame, live_rect, filter_intensity)

        # Détection du pincement à 3 doigts (sur l'une ou l'autre main),
            draw_text(frame, f"Filtre: {FILTERS[filter_index]} ({filter_intensity:.2f})", position=(x1, max(y1 - 35, 10)), color=(0, 255, 255), scale=0.6)
        if confirmed_rect and not selecting:
            for lm in (landmarks_0, landmarks_1):
                if lm and GestureRecognizer.is_three_finger_pinch(lm):
                    three_pinch_now = True
                    break

            if three_pinch_now and not was_three_pinch:
                ratio_index = (ratio_index + 1) % len(ASPECT_RATIOS)
                ratio_name, ratio_value = ASPECT_RATIOS[ratio_index]
                confirmed_rect = apply_aspect_ratio(confirmed_rect, ratio_value, frame_w, frame_h)
                print(f"Format vidéo : {ratio_name}")

        was_three_pinch = three_pinch_now

        # affichage
        if live_rect:
            x1, y1, x2, y2 = live_rect
            # appliquer filtre courant à la zone en cours
            frame = apply_color_filter_to_rect(frame, live_rect, FILTERS[filter_index])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            draw_text(frame, f"{x2 - x1} x {y2 - y1}", position=(x1, max(y1 - 15, 20)), scale=0.7)
            draw_text(frame, f"Filtre: {FILTERS[filter_index]}", position=(x1, max(y1 - 35, 10)), color=(0, 255, 255), scale=0.6)
        elif confirmed_rect:
            x1, y1, x2, y2 = confirmed_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ratio_name, _ = ASPECT_RATIOS[ratio_index]
            draw_text(frame, f"Zone: {x2 - x1}x{y2 - y1} | Format: {ratio_name} (pince 3 doigts = changer, S = sauver)",
                      position=(x1, max(y1 - 15, 20)), scale=0.6, color=(0, 255, 0))

        if not confirmed_rect and not live_rect:
            draw_text(frame, "Pince des DEUX mains pour definir une zone", position=(15, frame_h - 20), scale=0.7)

        cv2.imshow("Zone Selector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('s'), ord('S')) and confirmed_rect:
            x1, y1, x2, y2 = confirmed_rect
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                save_counter += 1
                filename = f"crop_{save_counter}.png"
                cv2.imwrite(filename, crop)
                print(f"Zone sauvegardee : {filename}")
                show_countdown_on_frame(frame, seconds=3, window_name="Zone Selector")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
