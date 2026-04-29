"""
app/utils/image.py
==================

Funcion
-------
Agrupa helpers de dibujo y conversion de imagen usados por la demo y las
herramientas de vision.

Notas
-----
Las funciones trabajan sobre frames BGR de OpenCV y devuelven el frame
modificado para permitir encadenar overlays.
"""

import cv2
import numpy as np
from app.vision.hand_detector import DetectionResult
from app.vision.landmark_extractor import HandFeatures, FingerState


def draw_fps(frame: cv2.typing.MatLike, fps: float) -> cv2.typing.MatLike:
    """Dibuja el FPS actual en la esquina superior izquierda."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame


def draw_finger_states(
    frame: cv2.typing.MatLike,
    fingers: list[FingerState],
    origin: tuple[int, int] = (10, 70),
) -> cv2.typing.MatLike:
    """
    Dibuja el estado de cada dedo (extendido / doblado) en el frame.

    Args:
        frame: Frame BGR.
        fingers: Lista de FingerState de una mano.
        origin: Esquina superior izquierda donde empieza el texto.
    """
    x, y = origin
    for i, finger in enumerate(fingers):
        color = (0, 255, 0) if finger.is_extended else (0, 0, 255)
        label = f"{finger.name}: {'UP' if finger.is_extended else 'DOWN'}"
        cv2.putText(frame, label, (x, y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def draw_bounding_box(
    frame: cv2.typing.MatLike,
    bbox: tuple[int, int, int, int],
    label: str = "",
    color: tuple[int, int, int] = (255, 165, 0),
) -> cv2.typing.MatLike:
    """Dibuja el bounding box de la mano con una etiqueta opcional."""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    if label:
        cv2.putText(frame, label, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def draw_landmark_index(
    frame: cv2.typing.MatLike,
    landmarks_px: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 255, 255),
) -> cv2.typing.MatLike:
    """Dibuja el índice numérico de cada landmark (útil para debug)."""
    for idx, (x, y) in enumerate(landmarks_px):
        cv2.putText(frame, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
    return frame


def frame_to_bytes(frame: cv2.typing.MatLike, ext: str = ".jpg") -> bytes:
    """Codifica un frame BGR a bytes (útil para streaming HTTP)."""
    _, buffer = cv2.imencode(ext, frame)
    return buffer.tobytes()
