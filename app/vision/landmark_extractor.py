"""
Funcion
-------
Convierte una mano detectada en features estructuradas para visualizacion o
modelos: landmarks planos, estado de dedos, bounding box y centro.

Notas
-----
El vector `landmarks_flat` contiene 63 valores por mano: 21 puntos por 3
coordenadas normalizadas.
"""

import numpy as np
from dataclasses import dataclass
from app.vision.hand_detector import DetectedHand, LANDMARK_NAMES


@dataclass
class FingerState:
    """Estado de un dedo: extendido o doblado."""
    name: str
    is_extended: bool
    tip_px: tuple[int, int]


@dataclass
class HandFeatures:
    """Features extraídas de una mano para clasificación."""
    hand_label: str
    landmarks_flat: list[float]     # 63 valores: 21 puntos × (x, y, z) normalizados
    fingers: list[FingerState]
    bounding_box: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max) en píxeles
    center_px: tuple[int, int]


# Índices tip e inner-joint por dedo para detectar extensión
FINGER_TIP_INDICES = {
    "THUMB":  (4, 3),
    "INDEX":  (8, 6),
    "MIDDLE": (12, 10),
    "RING":   (16, 14),
    "PINKY":  (20, 18),
}


class LandmarkExtractor:
    """
    Extrae features estructuradas a partir de un DetectedHand.

    Uso:
        extractor = LandmarkExtractor()
        features = extractor.extract(hand)
    """

    def extract(self, hand: DetectedHand) -> HandFeatures:
        """
        Extrae landmarks normalizados, estado de dedos y bounding box.

        Args:
            hand: Objeto DetectedHand con landmarks_norm y landmarks_px.

        Returns:
            HandFeatures listo para pasar a un clasificador.
        """
        landmarks_flat = self._flatten_landmarks(hand.landmarks_norm)
        fingers = self._detect_finger_states(hand)
        bbox = self._bounding_box(hand.landmarks_px)
        center = self._center(bbox)

        return HandFeatures(
            hand_label=hand.hand_label,
            landmarks_flat=landmarks_flat,
            fingers=fingers,
            bounding_box=bbox,
            center_px=center,
        )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _flatten_landmarks(self, landmarks_norm: list[tuple[float, float, float]]) -> list[float]:
        """Aplana los 21 landmarks (x,y,z) en un vector de 63 floats."""
        flat = []
        for x, y, z in landmarks_norm:
            flat.extend([x, y, z])
        return flat

    def _detect_finger_states(self, hand: DetectedHand) -> list[FingerState]:
        """
        Determina si cada dedo está extendido comparando la posición del tip
        vs. el joint anterior (eje Y en coordenadas normalizadas).
        """
        fingers = []
        norm = hand.landmarks_norm
        px = hand.landmarks_px

        for finger_name, (tip_idx, base_idx) in FINGER_TIP_INDICES.items():
            tip_y = norm[tip_idx][1]
            base_y = norm[base_idx][1]

            # En imagen, Y crece hacia abajo → tip por encima = extendido
            if finger_name == "THUMB":
                # El pulgar se compara en X por su orientación lateral
                tip_x = norm[tip_idx][0]
                base_x = norm[base_idx][0]
                is_extended = abs(tip_x - base_x) > 0.04
            else:
                is_extended = tip_y < base_y

            fingers.append(FingerState(
                name=finger_name,
                is_extended=is_extended,
                tip_px=px[tip_idx],
            ))

        return fingers

    def _bounding_box(self, landmarks_px: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        """Calcula el bounding box mínimo que envuelve todos los landmarks."""
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        return (min(xs), min(ys), max(xs), max(ys))

    def _center(self, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        x_min, y_min, x_max, y_max = bbox
        return ((x_min + x_max) // 2, (y_min + y_max) // 2)

    def to_numpy(self, features: HandFeatures) -> np.ndarray:
        """Convierte landmarks_flat a un array numpy (útil para modelos ML)."""
        return np.array(features.landmarks_flat, dtype=np.float32)
