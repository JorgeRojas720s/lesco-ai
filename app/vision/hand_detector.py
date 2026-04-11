import mediapipe as mp
import cv2
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Índices de landmarks de MediaPipe (21 puntos por mano)
LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
}


@dataclass
class DetectedHand:
    """Representa una mano detectada con sus 21 landmarks."""
    hand_label: str                    # "Left" o "Right"
    score: float                       # Confianza de detección
    landmarks_px: list[tuple[int, int]]        # Coordenadas en píxeles (x, y)
    landmarks_norm: list[tuple[float, float, float]]  # Coordenadas normalizadas (x, y, z)


@dataclass
class DetectionResult:
    """Resultado completo de detección en un frame."""
    hands: list[DetectedHand] = field(default_factory=list)

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    @property
    def has_hands(self) -> bool:
        return len(self.hands) > 0


class HandDetector:
    """
    Detecta manos en un frame usando MediaPipe Hands.

    Uso:
        detector = HandDetector()
        with detector:
            result = detector.detect(frame)
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_complexity = model_complexity

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = None

    def start(self) -> None:
        """Inicializa el modelo de MediaPipe."""
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        logger.info("HandDetector iniciado.")

    def stop(self) -> None:
        """Libera recursos del modelo."""
        if self._hands:
            self._hands.close()
            logger.info("HandDetector detenido.")

    def detect(self, frame_bgr: cv2.typing.MatLike) -> DetectionResult:
        """
        Procesa un frame BGR y retorna las manos detectadas.

        Args:
            frame_bgr: Frame capturado por OpenCV (BGR).

        Returns:
            DetectionResult con la lista de manos encontradas.
        """
        if self._hands is None:
            raise RuntimeError("El detector no está iniciado. Llama a start() primero.")

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        mp_result = self._hands.process(frame_rgb)

        result = DetectionResult()

        if not mp_result.multi_hand_landmarks:
            return result

        for hand_landmarks, handedness in zip(
            mp_result.multi_hand_landmarks,
            mp_result.multi_handedness,
        ):
            label = handedness.classification[0].label       # "Left" / "Right"
            score = handedness.classification[0].score

            landmarks_px = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_landmarks.landmark
            ]
            landmarks_norm = [
                (lm.x, lm.y, lm.z)
                for lm in hand_landmarks.landmark
            ]

            result.hands.append(DetectedHand(
                hand_label=label,
                score=score,
                landmarks_px=landmarks_px,
                landmarks_norm=landmarks_norm,
            ))

        return result

    def draw(
        self,
        frame_bgr: cv2.typing.MatLike,
        result: DetectionResult,
        draw_connections: bool = True,
    ) -> cv2.typing.MatLike:
        """
        Dibuja los landmarks sobre el frame. Retorna el frame anotado (copia).

        Args:
            frame_bgr: Frame original.
            result: Resultado de detect().
            draw_connections: Si True, dibuja las líneas entre landmarks.
        """
        annotated = frame_bgr.copy()

        for hand in result.hands:
            # Dibujar puntos directamente desde píxeles
            for x, y in hand.landmarks_px:
                cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

            # Dibujar conexiones manualmente
            if draw_connections:
                for start_idx, end_idx in self._mp_hands.HAND_CONNECTIONS:
                    x1, y1 = hand.landmarks_px[start_idx]
                    x2, y2 = hand.landmarks_px[end_idx]
                    cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Etiqueta de la mano
            wrist_px = hand.landmarks_px[0]
            cv2.putText(
                annotated,
                f"{hand.hand_label} ({hand.score:.2f})",
                (wrist_px[0] - 30, wrist_px[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()