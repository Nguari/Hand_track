"""
drawing.py
-----------
Fonctions d'affichage (FPS, texte, overlays) pour l'application.
"""

import cv2
import numpy as np
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


def show_countdown_on_frame(frame, seconds=3, window_name="Hand Tracking",
                            scale=4, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Affiche un compte à rebours en gras centré sur la frame.

    - Affiche les nombres `seconds`..1 avec un arrière-plan semi-transparent.
    - Utilise `cv2.imshow` et `cv2.waitKey(1000)` pour cadencer chaque seconde.
    - N'altère pas la frame originale (travaille sur une copie).
    """
    if seconds <= 0:
        return

    orig = frame.copy()
    h, w = orig.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(3, int(scale)) * 2  # épaisseur "gras"

    for sec in range(seconds, 0, -1):
        tmp = orig.copy()
        text = str(sec)

        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        x = w // 2 - text_w // 2
        y = h // 2 + text_h // 2

        pad = 30
        rx1, ry1 = max(0, x - pad), max(0, y - text_h - pad)
        rx2, ry2 = min(w, x + text_w + pad), min(h, y + pad)

        # rectangle d'arrière-plan semi-transparent
        overlay = tmp.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), bg_color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, tmp, 1 - alpha, 0, tmp)

        # texte en gras et anti-alias
        cv2.putText(tmp, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        cv2.imshow(window_name, tmp)
        key = cv2.waitKey(1000) & 0xFF
        if key == 27:
            break


def apply_color_filter_to_rect(frame, rect, mode, intensity=1.0):
    """Applique un filtre colorimétrique à la région rect (x1,y1,x2,y2) de la frame.

    - `mode` : 'normal', 'gray', 'invert', 'sepia', 'saturated', 'high_contrast'
    - `intensity` : float (approx 0.2 .. 2.0) qui module l'effet
    Retourne la frame modifiée en place.
    """
    try:
        x1, y1, x2, y2 = rect
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
    except Exception:
        return frame

    h_frame, w_frame = frame.shape[:2]
    # clamp coordinates to image bounds
    x1 = max(0, min(w_frame - 1, x1))
    x2 = max(0, min(w_frame, x2))
    y1 = max(0, min(h_frame - 1, y1))
    y2 = max(0, min(h_frame, y2))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    # clamp intensity
    intensity = float(max(0.2, min(2.0, intensity)))

    if mode == "normal":
        out = roi

    elif mode == "gray":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # blend original and gray according to intensity (0.0 = original, 1.0 = gray)
        alpha = max(0.0, min(1.0, intensity))
        out = cv2.addWeighted(gray_bgr, alpha, roi, 1 - alpha, 0)

    elif mode == "invert":
        inv = 255 - roi
        alpha = max(0.0, min(1.0, intensity))
        out = cv2.addWeighted(inv, alpha, roi, 1 - alpha, 0)

    elif mode == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sep = cv2.transform(roi, kernel)
        sep = np.clip(sep, 0, 255).astype('uint8')
        alpha = max(0.0, min(1.0, intensity))
        out = cv2.addWeighted(sep, alpha, roi, 1 - alpha, 0)

    elif mode == "saturated":
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype('float32')
        # base multiplier 1.6 scaled by intensity
        mult = 1.0 + (0.6 * intensity)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * mult, 0, 255)
        out = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

    elif mode == "high_contrast":
        # alpha controls contrast, beta small brightness shift
        alpha = 1.0 + 0.6 * intensity
        beta = int(10 * intensity)
        out = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)

    else:
        out = roi

    frame[y1:y2, x1:x2] = out
    return frame


def draw_intensity_bar(frame, rect, intensity, bar_color=(0, 255, 255), width=12, padding=8):
    """Dessine une barre verticale d'intensité à côté du rectangle `rect`.

    - `intensity` est attendu entre 0.2 et 2.0; on la normalise pour l'affichage.
    - La barre essaie de se placer à droite du rect; si elle dépasse de l'image,
      elle est dessinée à gauche.
    - Dessine un fond semi-transparent et la portion remplie proportionnelle.
    """
    x1, y1, x2, y2 = rect
    h_frame, w_frame = frame.shape[:2]

    # dimensions et position de la barre
    bar_h = int((y2 - y1) * 0.8)
    cy = (y1 + y2) // 2
    by1 = max(0, cy - bar_h // 2)
    by2 = min(h_frame, cy + bar_h // 2)

    bx1 = x2 + padding
    bx2 = bx1 + width
    # si la barre dépasse à droite, la placer à gauche du rectangle
    if bx2 >= w_frame:
        bx2 = x1 - padding
        bx1 = bx2 - width
        if bx1 < 0:
            # pas assez d'espace -> réduire la largeur
            bx1 = max(0, w_frame - width - padding)
            bx2 = bx1 + width

    # normaliser intensity [0.2..2.0] -> [0..1]
    norm = (float(intensity) - 0.2) / (2.0 - 0.2)
    norm = max(0.0, min(1.0, norm))

    filled_h = int((by2 - by1) * norm)
    filled_y1 = by2 - filled_h

    # dessiner sur overlay pour alpha
    overlay = frame.copy()
    # fond
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
    # portion remplie
    cv2.rectangle(overlay, (bx1, filled_y1), (bx2, by2), bar_color, -1)

    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # bordure et valeur textuelle
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
    val_text = f"{intensity:.2f}"
    txt_x = bx1 - 6 - (len(val_text) * 8)
    if txt_x < 0:
        txt_x = bx2 + 6
    txt_y = by2 + 20 if by2 + 20 < h_frame else by1 - 6
    cv2.putText(frame, val_text, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    return frame
