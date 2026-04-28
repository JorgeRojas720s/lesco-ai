"""
app/recognize.py
================
Reconocedor en tiempo real de señas LESCO usando DTW (Dynamic Time Warping).

Cómo funciona
-------------
1.  Al iniciar, carga el dataset HDF5 y calcula un *template* por etiqueta:
    el promedio de todas las muestras de esa etiqueta (ya normalizadas a 60
    frames).  Con pocas muestras, el promedio es una estimación razonable del
    "movimiento típico" de esa seña.

2.  La cámara corre en bucle.  Un **detector de movimiento** monitorea la
    velocidad de los landmarks frame a frame.  Cuando la velocidad supera
    un umbral → la seña está comenzando (estado SIGNING).

3.  Mientras está en SIGNING, los frames se acumulan en un buffer.  Cuando
    la velocidad cae por debajo del umbral durante N frames seguidos, la seña
    se considera terminada.

4.  El buffer capturado se remuestrea a 60 frames y se compara contra cada
    template usando **DTW** (Dynamic Time Warping).  DTW es la elección
    correcta porque:
    - Maneja que una persona firme más rápido o más lento que el template.
    - Encuentra la alineación temporal óptima entre dos secuencias.
    - No requiere entrenamiento previo.

5.  El label con menor distancia DTW gana.  Se calcula una confianza relativa
    entre el mejor y el segundo lugar.

Uso
---
    uv run python -m app.recognize --dataset data/signs_dataset.h5

    # Ajustar sensibilidad de detección de movimiento
    uv run python -m app.recognize --dataset data/signs_dataset.h5 --motion-threshold 0.04

    # Ver todos los candidatos en pantalla
    uv run python -m app.recognize --dataset data/signs_dataset.h5 --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from enum import Enum, auto
from pathlib import Path

import cv2
import h5py
import numpy as np

from app.vision.camera import Camera
from app.vision.hand_detector import DetectionResult, HandDetector
from app.vision.preprocessor import Preprocessor

# Reutilizamos las mismas funciones que usamos al grabar → garantiza que
# los features sean idénticos entre entrenamiento e inferencia.
from app.collect_data import (
    FEATURES_PER_FRAME,
    HAND_SLOTS,
    frame_to_features,
    normalize_hand,
    smart_resample,
)

LOGGER = logging.getLogger(__name__)

# ── Parámetros por defecto ─────────────────────────────────────────────────────
DEFAULT_MOTION_THRESHOLD = 0.03   # norma L2 del delta de features entre frames
DEFAULT_SILENCE_FRAMES   = 22     # frames quietos para confirmar fin de seña
DEFAULT_MIN_SIGN_FRAMES  = 14     # mínimo de frames para clasificar
DEFAULT_MAX_SIGN_FRAMES  = 200    # fuerza clasificación si la seña es muy larga
DEFAULT_RESULT_HOLD_S    = 2.5    # segundos que se muestra el resultado
DEFAULT_TOP_K            = 3      # cuántos candidatos mostrar
DTW_BAND                 = 12     # ancho de la banda Sakoe-Chiba (frames)
SEQUENCE_LENGTH          = 60     # debe coincidir con el dataset


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def load_templates(h5_path: Path) -> dict[str, np.ndarray]:
    """
    Lee el dataset HDF5 y calcula un template promedio por etiqueta.

    El template es el centroide de todas las muestras de esa etiqueta en el
    espacio de secuencias (T, F).  Con pocas muestras es una buena estimación
    del patrón temporal central.

    Returns
    -------
    dict  label → array float32 de shape (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {h5_path}")

    templates: dict[str, np.ndarray] = {}

    with h5py.File(h5_path, "r") as f:
        n = f["X"].shape[0]
        if n == 0:
            raise ValueError("El dataset está vacío. Graba muestras primero.")

        X = f["X"][:]                          # (N, T, F)
        raw_labels = f["labels"][:]
        labels = [
            lb.decode("utf-8") if isinstance(lb, bytes) else str(lb)
            for lb in raw_labels
        ]

    unique_labels = sorted(set(labels))
    labels_arr = np.array(labels)

    for label in unique_labels:
        mask = labels_arr == label
        samples = X[mask]                      # (k, T, F)
        templates[label] = samples.mean(axis=0).astype(np.float32)
        LOGGER.info("Template '%s': %d muestras promediadas.", label, mask.sum())

    return templates


# ══════════════════════════════════════════════════════════════════════════════
# 2. DTW (Dynamic Time Warping)
# ══════════════════════════════════════════════════════════════════════════════

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band: int = DTW_BAND) -> float:
    """
    Calcula la distancia DTW entre dos secuencias multivariadas.

    Usa la banda de Sakoe-Chiba para limitar la búsqueda a ±band frames
    alrededor de la diagonal.  Esto hace el cálculo O(N·band) en vez de
    O(N²) y, más importante, evita alineaciones temporales absurdas
    (ej. mapear el inicio de una seña con su final).

    Parameters
    ----------
    seq_a, seq_b:
        Arrays float32 de shape (T_a, F) y (T_b, F).
    band:
        Ancho de la banda Sakoe-Chiba en frames.

    Returns
    -------
    Distancia DTW normalizada (dividida por T_a + T_b para comparar
    secuencias de diferente longitud sin sesgo).
    """
    T_a, F = seq_a.shape
    T_b     = seq_b.shape[0]

    INF = np.inf
    # Tabla de costos acumulados
    dp = np.full((T_a, T_b), INF, dtype=np.float64)

    for i in range(T_a):
        j_lo = max(0, i - band)
        j_hi = min(T_b, i + band + 1)
        for j in range(j_lo, j_hi):
            # Distancia euclidiana entre frames i y j
            diff  = seq_a[i] - seq_b[j]
            cost  = float(np.dot(diff, diff) ** 0.5)   # norma L2

            if i == 0 and j == 0:
                dp[i, j] = cost
            else:
                prev = INF
                if i > 0 and j > 0:
                    prev = min(prev, dp[i-1, j-1])
                if i > 0 and j >= j_lo:
                    prev = min(prev, dp[i-1, j])
                if j > 0 and i <= T_a - 1:
                    prev = min(prev, dp[i, j-1])
                dp[i, j] = cost + (prev if prev < INF else INF)

    raw = dp[T_a - 1, T_b - 1]
    # Normalizar para que la distancia no dependa de la longitud
    return float(raw) / (T_a + T_b)


def classify(
    live_seq: np.ndarray,
    templates: dict[str, np.ndarray],
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, float, float]]:
    """
    Compara live_seq contra todos los templates con DTW.

    Returns
    -------
    Lista de (label, distancia, confianza) ordenada de mejor a peor.
    La confianza es un porcentaje relativo: el ganador obtiene 100% si
    supera ampliamente al segundo lugar.
    """
    results: list[tuple[str, float]] = []

    for label, template in templates.items():
        dist = dtw_distance(live_seq, template)
        results.append((label, dist))

    results.sort(key=lambda x: x[1])   # menor distancia = más parecido

    # Confianza relativa: 1 - (mejor / segundo_mejor)
    # Si el mejor duplica al segundo en distancia, confianza ≈ 50 %
    # Si el mejor gana por margen amplio, confianza → 100 %
    if len(results) >= 2:
        best_dist    = results[0][1]
        second_dist  = results[1][1]
        denom        = best_dist + second_dist
        confidence   = (1.0 - best_dist / denom) * 100.0 if denom > 0 else 100.0
    else:
        confidence = 100.0

    output = []
    for rank, (label, dist) in enumerate(results[:top_k]):
        c = confidence if rank == 0 else (100.0 - confidence) / max(len(results) - 1, 1)
        output.append((label, dist, c))

    return output


# ══════════════════════════════════════════════════════════════════════════════
# 3. DETECTOR DE MOVIMIENTO / SEGMENTADOR DE SEÑAS
# ══════════════════════════════════════════════════════════════════════════════

class SignerState(Enum):
    WAITING   = auto()   # esperando que empiece el movimiento
    SIGNING   = auto()   # capturando la seña activa
    DONE      = auto()   # seña terminada, lista para clasificar


class MotionSegmenter:
    """
    Detecta automáticamente el inicio y el fin de una seña basándose en
    la velocidad de los landmarks.

    Principio
    ---------
    La velocidad en el frame t es la norma L2 de (features[t] - features[t-1]).
    Se mantiene una ventana deslizante para suavizar picos de ruido.

    Transiciones
    ------------
    WAITING  → SIGNING   cuando la velocidad media supera motion_threshold
    SIGNING  → WAITING   si el buffer es demasiado corto (seña descartada)
    SIGNING  → DONE      cuando la velocidad cae por debajo del umbral
                          durante silence_frames frames consecutivos
    """

    def __init__(
        self,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        silence_frames: int     = DEFAULT_SILENCE_FRAMES,
        min_sign_frames: int    = DEFAULT_MIN_SIGN_FRAMES,
        max_sign_frames: int    = DEFAULT_MAX_SIGN_FRAMES,
        smooth_window: int      = 5,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.silence_frames   = silence_frames
        self.min_sign_frames  = min_sign_frames
        self.max_sign_frames  = max_sign_frames

        self.state: SignerState           = SignerState.WAITING
        self._buffer: list[np.ndarray]   = []
        self._detected: list[bool]        = []
        self._prev_feat: np.ndarray | None = None
        self._silence_count: int          = 0
        self._velocity_window: deque[float] = deque(maxlen=smooth_window)
        self.current_velocity: float      = 0.0

    # ── API pública ────────────────────────────────────────────────────────────

    def update(self, features: np.ndarray, hand_detected: bool) -> bool:
        """
        Procesa un frame.

        Returns True cuando una seña completa está lista en get_sequence().
        """
        # Calcula velocidad frame-a-frame
        if self._prev_feat is not None and hand_detected:
            delta = features - self._prev_feat
            velocity = float(np.linalg.norm(delta))
        else:
            velocity = 0.0

        self._prev_feat = features.copy()
        self._velocity_window.append(velocity)
        self.current_velocity = float(np.mean(self._velocity_window))

        moving = self.current_velocity > self.motion_threshold

        # ── Máquina de estados ─────────────────────────────────────────────────
        if self.state == SignerState.WAITING:
            if moving and hand_detected:
                self.state = SignerState.SIGNING
                self._buffer = [features]
                self._detected = [hand_detected]
                self._silence_count = 0

        elif self.state == SignerState.SIGNING:
            self._buffer.append(features)
            self._detected.append(hand_detected)

            if not moving:
                self._silence_count += 1
            else:
                self._silence_count = 0

            sign_done = (
                self._silence_count >= self.silence_frames
                or len(self._buffer) >= self.max_sign_frames
            )

            if sign_done:
                if len(self._buffer) >= self.min_sign_frames:
                    self.state = SignerState.DONE
                    return True          # ← seña lista
                else:
                    # Demasiado corta → descartar y volver a esperar
                    LOGGER.debug("Seña descartada: %d frames < mínimo %d",
                                 len(self._buffer), self.min_sign_frames)
                    self._reset()

        return False

    def get_sequence(self) -> tuple[list[np.ndarray], list[bool]]:
        """Devuelve el buffer acumulado (frames, detected). Llama después de update() == True."""
        return list(self._buffer), list(self._detected)

    def reset(self) -> None:
        """Reinicia el segmentador para la siguiente seña."""
        self._reset()

    # ── Internos ───────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self.state          = SignerState.WAITING
        self._buffer        = []
        self._detected      = []
        self._silence_count = 0
        self._velocity_window.clear()
        self.current_velocity = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Colores BGR
_C_GREEN    = (0,  210,  80)
_C_RED      = (0,   40, 220)
_C_YELLOW   = (0,  210, 230)
_C_WHITE    = (230, 230, 230)
_C_GRAY     = (140, 140, 140)
_C_CYAN     = (220, 200,   0)
_C_ORANGE   = (0,  140, 255)
_C_BLACK    = (0,    0,   0)


def _text(frame, text, pos, scale=0.65, color=_C_WHITE, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(
    frame: cv2.typing.MatLike,
    segmenter: MotionSegmenter,
    result: list[tuple[str, float, float]] | None,
    result_expires: float,
    top_k: int,
    frame_count: int,
) -> None:
    """Dibuja toda la interfaz sobre el frame."""
    h, w = frame.shape[:2]
    now = time.monotonic()

    # ── Panel de estado (esquina superior izquierda) ───────────────────────────
    state_map = {
        SignerState.WAITING: ("ESPERANDO SEÑA",  _C_GRAY),
        SignerState.SIGNING: ("CAPTURANDO...",   _C_RED),
        SignerState.DONE:    ("CLASIFICANDO",    _C_YELLOW),
    }
    state_text, state_color = state_map[segmenter.state]

    # Fondo semitransparente para el panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), _C_BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if segmenter.state == SignerState.SIGNING and (frame_count // 12) % 2 == 0:
        _text(frame, "● " + state_text, (12, 30), 0.75, _C_RED, 2)
    else:
        _text(frame, state_text, (12, 30), 0.75, state_color, 2)

    frames_info = f"Frames: {len(segmenter._buffer)}" if segmenter.state == SignerState.SIGNING else ""
    _text(frame, frames_info, (12, 58), 0.55, _C_GRAY)
    _text(frame, "Q: salir", (w - 90, 30), 0.5, _C_GRAY)

    # ── Barra de velocidad (movimiento) ───────────────────────────────────────
    bar_h   = 14
    bar_y   = 78
    bar_w   = w - 24
    ratio   = min(segmenter.current_velocity / (segmenter.motion_threshold * 3), 1.0)
    filled  = int(bar_w * ratio)

    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    bar_color = _C_RED if ratio > 0.33 else _C_GREEN
    if filled > 0:
        cv2.rectangle(frame, (12, bar_y), (12 + filled, bar_y + bar_h), bar_color, -1)
    # Línea de umbral
    threshold_x = 12 + int(bar_w * (segmenter.motion_threshold / (segmenter.motion_threshold * 3)))
    cv2.line(frame, (threshold_x, bar_y - 2), (threshold_x, bar_y + bar_h + 2), _C_YELLOW, 1)
    _text(frame, "vel", (12, bar_y - 3), 0.38, _C_GRAY)

    # ── Resultado de clasificación ─────────────────────────────────────────────
    if result is not None and now < result_expires:
        remaining = result_expires - now
        alpha_fade = min(1.0, remaining / 0.5)   # fade-out en el último 0.5 s

        # Fondo grande para el resultado principal
        box_y  = h // 2 - 55
        box_h  = 60
        panel  = frame.copy()
        cv2.rectangle(panel, (0, box_y), (w, box_y + box_h), _C_BLACK, -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

        best_label, best_dist, best_conf = result[0]

        # Etiqueta principal (grande)
        font_scale = 1.8
        (tw, th), _ = cv2.getTextSize(best_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
        cx = (w - tw) // 2
        # Sombra
        cv2.putText(frame, best_label, (cx + 2, box_y + 46), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, _C_BLACK, 4)
        cv2.putText(frame, best_label, (cx, box_y + 44), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, _C_GREEN, 3)

        # Barra de confianza
        conf_bar_y = box_y + box_h + 6
        conf_w     = int((w - 24) * best_conf / 100.0)
        cv2.rectangle(frame, (12, conf_bar_y), (w - 12, conf_bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (12, conf_bar_y), (12 + conf_w, conf_bar_y + 10), _C_GREEN, -1)
        _text(frame, f"Confianza: {best_conf:.0f}%", (12, conf_bar_y + 24), 0.52, _C_GREEN)

        # Candidatos alternativos (debajo)
        if len(result) > 1:
            alt_y = conf_bar_y + 42
            _text(frame, "Alternativas:", (12, alt_y), 0.48, _C_GRAY)
            for i, (lbl, dist, _) in enumerate(result[1:top_k]):
                _text(frame, f"  {lbl}  (dist {dist:.3f})",
                      (12, alt_y + 20 + i * 20), 0.48, _C_GRAY)

    elif segmenter.state == SignerState.WAITING and result is None:
        # Instrucción inicial
        msg = "Realiza una sena frente a la camara"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cx = (w - tw) // 2
        _text(frame, msg, (cx, h - 20), 0.6, _C_CYAN)


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconocedor en tiempo real de señas LESCO con DTW.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",          type=Path, default="data/signs_dataset.h5")
    p.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD,
                   help="Velocidad mínima para considerar que hay movimiento.")
    p.add_argument("--silence-frames",   type=int,   default=DEFAULT_SILENCE_FRAMES,
                   help="Frames quietos para confirmar fin de seña.")
    p.add_argument("--min-sign-frames",  type=int,   default=DEFAULT_MIN_SIGN_FRAMES)
    p.add_argument("--top-k",            type=int,   default=DEFAULT_TOP_K,
                   help="Cuántos candidatos mostrar en pantalla.")
    p.add_argument("--camera-index",     type=int,   default=0)
    p.add_argument("--result-hold",      type=float, default=DEFAULT_RESULT_HOLD_S,
                   help="Segundos que se mantiene el resultado en pantalla.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Cargar templates ───────────────────────────────────────────────────────
    LOGGER.info("Cargando templates desde %s …", args.dataset)
    templates = load_templates(args.dataset)
    LOGGER.info("Templates listos: %s", list(templates.keys()))

    if len(templates) < 2:
        LOGGER.warning(
            "Solo hay %d etiqueta(s). El reconocedor necesita al menos 2 para "
            "tener una comparación significativa.", len(templates)
        )

    preprocessor = Preprocessor(flip_horizontal=True)
    segmenter     = MotionSegmenter(
        motion_threshold=args.motion_threshold,
        silence_frames=args.silence_frames,
        min_sign_frames=args.min_sign_frames,
    )

    last_result:   list[tuple[str, float, float]] | None = None
    result_expires: float = 0.0
    frame_count:   int   = 0

    LOGGER.info("Listo. Presiona Q para salir.")

    try:
        with Camera(camera_index=args.camera_index) as camera, \
             HandDetector(max_hands=2) as detector:

            for raw_frame in camera.stream():
                frame_count += 1
                frame  = preprocessor.process(raw_frame)
                result_det = detector.detect(frame)
                annotated  = detector.draw(frame, result_det)

                # Extraer features del frame actual
                feat, hand_detected = frame_to_features(result_det)

                # Actualizar segmentador; devuelve True cuando la seña terminó
                sign_complete = segmenter.update(feat, hand_detected)

                if sign_complete:
                    frames_buf, detected_buf = segmenter.get_sequence()

                    # Remuestrear a la misma longitud que los templates
                    live_seq = smart_resample(frames_buf, detected_buf, SEQUENCE_LENGTH)

                    # Clasificar con DTW
                    ranking = classify(live_seq, templates, top_k=args.top_k)
                    last_result    = ranking
                    result_expires = time.monotonic() + args.result_hold

                    LOGGER.info(
                        "Clasificación: %s  (conf %.0f%%)  |  dist=%.4f",
                        ranking[0][0], ranking[0][2], ranking[0][1],
                    )

                    segmenter.reset()

                # Dibujar HUD
                draw_hud(
                    annotated,
                    segmenter=segmenter,
                    result=last_result,
                    result_expires=result_expires,
                    top_k=args.top_k,
                    frame_count=frame_count,
                )

                cv2.imshow("LESCO-AI Recognizer", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
