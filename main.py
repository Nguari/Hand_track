"""
main.py
--------
Point d'entrée principal de l'application de hand tracking. Combine :

- Détection du squelette de la/les main(s)
- Comptage des doigts levés (mode 1 main)
- Détection de pincement pouce-index (mode 1 main)
- Sélection de zone rectangulaire (mode 2 mains, pincement simultané)
- Changement de format vidéo de la zone (pincement à 3 doigts, une fois
  la zone figée) : Libre -> 16:9 -> 9:16 -> 4:3 -> 1:1 -> ...

Touches / gestes :
    Pince (1 main), une fois la zone figee -> sauvegarde (crop_N.png)
    S      -> sauvegarde aussi via le clavier
    Échap  -> quitter
"""

import cv2

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.drawing import FPSCounter, draw_text, show_countdown_on_frame, apply_color_filter_to_rect, draw_intensity_bar

# (nom affiché, ratio largeur/hauteur ; None = format libre, ne change rien)
ASPECT_RATIOS = [
    ("Libre", None),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("4:3", 4 / 3),
    ("1:1", 1 / 1),
]

# Filtres disponibles pour la zone
FILTERS = ["normal", "gray", "invert", "sepia", "saturated", "high_contrast"]


def get_pinch_point(landmarks):
    """Retourne le point médian (x, y) entre le pouce (4) et l'index (8)."""
    thumb = landmarks[4]
    index = landmarks[8]
    _, x1, y1 = thumb
    _, x2, y2 = index
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def normalize_rect(p1, p2):
    """Ordonne deux points en (x1, y1, x2, y2) avec x1<x2 et y1<y2."""
    x1, y1 = p1
    x2, y2 = p2
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def apply_aspect_ratio(rect, ratio, frame_w, frame_h):
    """
    Recalcule le rectangle pour respecter le ratio donné, en gardant le
    même centre et la même largeur (la hauteur s'ajuste). Si ratio est
    None, retourne le rectangle inchangé (format libre).
    """
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
    fps_counter = FPSCounter()

    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la webcam.")
        return

    # --- État de la sélection de zone (mode 2 mains) ---
    selecting = False           # les deux mains sont-elles en train de pincer ?
    confirmed_rect = None       # dernière zone validée (figée)
    ratio_index = 0
    filter_index = 0
    was_filter_pinching = False
    filter_intensity = 1.0
    prev_filter_y = None
    was_three_pinch = False     # pour ne déclencher le changement de format qu'une fois par geste
    was_single_pinch = False    # pour ne déclencher la sauvegarde qu'une fois par geste
    save_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # effet miroir, plus naturel
        frame_h, frame_w, _ = frame.shape
        frame = detector.find_hands(frame)

        landmarks_0 = detector.get_landmark_positions(frame, hand_index=0)
        landmarks_1 = detector.get_landmark_positions(frame, hand_index=1)
        num_hands = detector.num_hands_detected()

        live_rect = None
        three_pinch_now = False

        # --- Mode 1 main : comptage de doigts + pincement simple ---
        trigger_save = False
        single_pinch_now = False

        if num_hands == 1 and landmarks_0:
            fingers = GestureRecognizer.count_fingers_up(landmarks_0)
            draw_text(frame, f"Doigts leves: {fingers}", position=(10, 80))

            single_pinch_now = GestureRecognizer.is_pinching(landmarks_0)
            if single_pinch_now:
                draw_text(frame, "Pincement detecte", position=(10, 120), color=(0, 255, 255))

            # Si une zone est déjà figée, un pincement simple (une seule
            # main) déclenche la sauvegarde -- une seule fois par geste.
            # On sauvegarde sur front montant (pincement) ET front descendant (relâchement)
            if confirmed_rect and ((single_pinch_now and not was_single_pinch) or (not single_pinch_now and was_single_pinch)):
                trigger_save = True

        was_single_pinch = single_pinch_now

        # --- Mode 2 mains : sélection de zone ---
        if num_hands >= 2 and landmarks_0 and landmarks_1:
            pinching_0 = GestureRecognizer.is_pinching(landmarks_0)
            pinching_1 = GestureRecognizer.is_pinching(landmarks_1)

            point_0 = get_pinch_point(landmarks_0)
            point_1 = get_pinch_point(landmarks_1)

            cv2.circle(frame, point_0, 12, (0, 255, 0) if pinching_0 else (0, 0, 255), -1)
            cv2.circle(frame, point_1, 12, (0, 255, 0) if pinching_1 else (0, 0, 255), -1)

            # Toggle selection on double-pinching: start on first pinch, confirm on next
            if pinching_0 and pinching_1:
                if not selecting:
                    selecting = True
                    live_rect = normalize_rect(point_0, point_1)
                else:
                    confirmed_rect = normalize_rect(point_0, point_1)
                    selecting = False
            else:
                if selecting:
                    # follow hands' trend even when pinch is released
                    live_rect = normalize_rect(point_0, point_1)
                else:
                    selecting = False

            # Single-hand pinch while selecting -> cycle color filter and adjust intensity
            single_hand_pinching = False
            pinching_hand_y = None
            if selecting:
                single_hand_pinching = (pinching_0 != pinching_1)
                if single_hand_pinching:
                    # determine which hand is pinching and get its pinch y
                    pinching_hand = landmarks_0 if pinching_0 else landmarks_1
                    _, _, pinching_hand_y = pinching_hand[4]  # thumb tip y

            # short press (tap) cycles filter; hold+move adjusts intensity
            if single_hand_pinching and not was_filter_pinching:
                # initial pinch: treat as cycle to next filter
                filter_index = (filter_index + 1) % len(FILTERS)
                was_filter_pinching = True
                prev_filter_y = pinching_hand_y

            if single_hand_pinching and was_filter_pinching and prev_filter_y is not None and pinching_hand_y is not None:
                # vertical movement: positive dy (upward) increases intensity
                dy = prev_filter_y - pinching_hand_y
                sensitivity = 0.005
                filter_intensity = max(0.2, min(2.0, filter_intensity + dy * sensitivity))
                prev_filter_y = pinching_hand_y

            if not single_hand_pinching:
                was_filter_pinching = False
                prev_filter_y = None
        else:
            selecting = False

        # --- Changement de format (pincement à 3 doigts, zone déjà figée) ---
        if confirmed_rect and not selecting:
            for lm in (landmarks_0, landmarks_1):
                if lm and GestureRecognizer.is_three_finger_pinch(lm):
                    three_pinch_now = True
                    break

            if three_pinch_now and not was_three_pinch:
                ratio_index = (ratio_index + 1) % len(ASPECT_RATIOS)
                ratio_name, ratio_value = ASPECT_RATIOS[ratio_index]
                confirmed_rect = apply_aspect_ratio(
                    confirmed_rect, ratio_value, frame_w, frame_h
                )
                print(f"Format vidéo : {ratio_name}")

        was_three_pinch = three_pinch_now

        # --- Affichage du rectangle (en cours ou figé) ---
        if live_rect:
            x1, y1, x2, y2 = live_rect
            # appliquer filtre courant à la zone en cours
            frame = apply_color_filter_to_rect(frame, live_rect, FILTERS[filter_index], intensity=filter_intensity)
            # afficher la barre d'intensité à côté de la zone
            frame = draw_intensity_bar(frame, live_rect, filter_intensity)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            draw_text(frame, f"{x2 - x1} x {y2 - y1}", position=(x1, max(y1 - 15, 20)), scale=0.7)
            draw_text(frame, f"Filtre: {FILTERS[filter_index]} ({filter_intensity:.2f})", position=(x1, max(y1 - 35, 10)), color=(0, 255, 255), scale=0.6)
        elif confirmed_rect:
            x1, y1, x2, y2 = confirmed_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ratio_name, _ = ASPECT_RATIOS[ratio_index]
            draw_text(
                frame,
                f"Zone: {x2 - x1}x{y2 - y1} | Format: {ratio_name} (3 doigts = changer, pince 1 main = sauver)",
                position=(x1, max(y1 - 15, 20)), scale=0.6, color=(0, 255, 0)
            )

        if not confirmed_rect and not live_rect:
            draw_text(frame, "1 main: doigts/pincement | 2 mains: pince pour selectionner une zone",
                      position=(15, frame_h - 20), scale=0.6)
        elif confirmed_rect and not selecting:
            draw_text(frame, "Pince (1 main) pour sauvegarder",
                      position=(15, frame_h - 20), scale=0.6, color=(0, 255, 255))

        fps_counter.update_and_draw(frame)
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Échap
            break
        elif key in (ord('s'), ord('S')) and confirmed_rect:
            trigger_save = True

        # --- Sauvegarde (déclenchée par la touche S OU par un pincement 1 main) ---
        if trigger_save and confirmed_rect:
            x1, y1, x2, y2 = confirmed_rect
            if x2 > x1 and y2 > y1:
                # extraire le ROI original et y appliquer le filtre sélectionné
                roi = frame[y1:y2, x1:x2].copy()
                h_roi, w_roi = roi.shape[:2]
                # appliquer le filtre sur un petit frame dont le rect est (0,0,w,h)
                filtered_roi = roi.copy()
                filtered_roi = apply_color_filter_to_rect(filtered_roi, (0, 0, w_roi, h_roi), FILTERS[filter_index], intensity=filter_intensity)

                save_counter += 1
                filename = f"crop_{save_counter}.png"
                cv2.imwrite(filename, filtered_roi)
                print(f"Zone sauvegardee : {filename}")
                # Affiche un compte à rebours en gras pendant 3 secondes
                show_countdown_on_frame(frame, seconds=3, window_name="Hand Tracking")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()