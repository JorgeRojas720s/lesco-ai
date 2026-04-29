"""
app/vision/preprocessor.py
==========================

Funcion
-------
Prepara los frames de la camara antes de pasarlos al detector de manos.

Notas
-----
Puede aplicar espejo horizontal, redimensionamiento y mejora de contraste
segun la configuracion usada al crear el preprocesador.
"""

import cv2
import numpy as np


class Preprocessor:
    """
    Preprocesa frames de cámara antes de pasarlos al detector.

    Aplica: flip horizontal, resize opcional y mejora de contraste.
    """

    def __init__(
        self,
        flip_horizontal: bool = True,
        target_size: tuple[int, int] | None = None,
        enhance_contrast: bool = False,
    ):
        """
        Args:
            flip_horizontal: Voltea la imagen como espejo (natural para el usuario).
            target_size: Si se provee (w, h), redimensiona el frame.
            enhance_contrast: Aplica CLAHE para mejorar la iluminación.
        """
        self.flip_horizontal = flip_horizontal
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance_contrast else None

    def process(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Aplica la cadena de preprocesamiento al frame.

        Args:
            frame: Frame BGR crudo de la cámara.

        Returns:
            Frame preprocesado listo para el detector.
        """
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        if self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.enhance_contrast and self._clahe:
            frame = self._apply_clahe(frame)

        return frame

    def _apply_clahe(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Aplica CLAHE canal por canal en el espacio LAB para mejorar contraste."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def crop_roi(
        self,
        frame: cv2.typing.MatLike,
        bbox: tuple[int, int, int, int],
        padding: int = 20,
    ) -> cv2.typing.MatLike:
        """
        Recorta la región de interés de una mano, con padding opcional.

        Args:
            frame: Frame completo.
            bbox: (x_min, y_min, x_max, y_max).
            padding: Píxeles extra alrededor del bbox.

        Returns:
            ROI recortada.
        """
        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = bbox

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        return frame[y_min:y_max, x_min:x_max]
