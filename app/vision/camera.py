import cv2
from typing import Generator
import logging
import os

logger = logging.getLogger(__name__)


class Camera:
    """Maneja la captura de video desde la cámara."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Abre la cámara y configura resolución/FPS."""
        if os.name == "nt":
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {self.camera_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(f"Cámara abierta: {self.width}x{self.height} @ {self.fps}fps")

    def close(self) -> None:
        """Libera los recursos de la cámara."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Cámara cerrada.")

    def read_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Lee un solo frame. Retorna (éxito, frame)."""
        if not self._cap or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def stream(self) -> Generator[cv2.typing.MatLike, None, None]:
        """Generador que produce frames continuamente mientras la cámara esté abierta."""
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("La cámara no está abierta. Llama a open() primero.")

        while True:
            success, frame = self._cap.read()
            if not success:
                logger.warning("No se pudo leer el frame.")
                break
            yield frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
