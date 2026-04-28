"""
app/vision/camera.py
====================
Handles webcam capture with retry logic for dropped frames and graceful
degradation on camera lag.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Generator

import cv2

logger = logging.getLogger(__name__)

# Maximum consecutive failed reads before the stream is considered dead.
_MAX_CONSECUTIVE_FAILURES = 30
# Seconds to wait between retry attempts on a failed read.
_RETRY_DELAY_S = 0.01


class Camera:
    """
    Manages video capture from a webcam.

    Improvements over v1
    --------------------
    * stream() tolerates transient camera lag – up to _MAX_CONSECUTIVE_FAILURES
      consecutive failed reads before it gives up, instead of stopping on the
      first bad frame.
    * Logs a warning every time a frame is dropped so you can correlate
      dropped frames with gaps in the landmark data.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the camera and configure resolution / FPS."""
        if os.name == "nt":
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)

        # Fallback if the backend-specific open failed.
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {self.camera_index}. "
                "Verifica que no esté siendo usada por otra aplicación."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("Cámara abierta: %dx%d @ %dfps", self.width, self.height, self.fps)

    def close(self) -> None:
        """Release camera resources."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Cámara cerrada.")

    def read_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Read a single frame.  Returns (success, frame)."""
        if not self._cap or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def stream(self) -> Generator[cv2.typing.MatLike, None, None]:
        """
        Yield frames continuously while the camera is open.

        Tolerates transient failures (camera lag, USB hiccups) by retrying
        up to _MAX_CONSECUTIVE_FAILURES times before aborting.  Each failed
        read is logged so you can investigate hardware issues.
        """
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("La cámara no está abierta. Llama a open() primero.")

        consecutive_failures = 0

        while True:
            success, frame = self._cap.read()

            if not success or frame is None:
                consecutive_failures += 1
                logger.warning(
                    "Frame fallido (%d/%d consecutivos).",
                    consecutive_failures,
                    _MAX_CONSECUTIVE_FAILURES,
                )
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "Demasiados frames fallidos consecutivos. "
                        "Deteniendo el stream."
                    )
                    break
                time.sleep(_RETRY_DELAY_S)
                continue

            consecutive_failures = 0   # reset counter on a good read
            yield frame

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
