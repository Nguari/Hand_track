"""
smoothing.py
-------------
Lissage des positions détectées pour éviter les tremblements (jitter).
"""


class PositionSmoother:
    """Lisse une position (x, y) dans le temps avec une moyenne pondérée."""

    def __init__(self, smoothing_factor=0.5):
        """
        smoothing_factor : entre 0 et 1.
            - proche de 0 : très réactif mais tremblant
            - proche de 1 : très lisse mais avec du retard
        """
        self.smoothing_factor = smoothing_factor
        self.prev_x = None
        self.prev_y = None

    def update(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y

        smooth_x = self.prev_x + (x - self.prev_x) * (1 - self.smoothing_factor)
        smooth_y = self.prev_y + (y - self.prev_y) * (1 - self.smoothing_factor)

        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def reset(self):
        self.prev_x = None
        self.prev_y = None
