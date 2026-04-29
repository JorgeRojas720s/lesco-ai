"""
app/main.py
===========

Funcion
-------
Ejecuta el reconocedor principal en tiempo real usando la red neuronal
entrenada y la webcam.

Comandos
--------
uv run python -m app.main
    Inicia el reconocedor usando models/neural_sign_classifier.npz.

uv run python -m app.main --model models/prueba.npz
    Inicia el reconocedor usando un modelo entrenado diferente.

uv run python -m app.main --dataset data/signs_dataset.h5
    Usa el dataset indicado para rechazar movimientos que no parecen senas conocidas.

uv run python -m app.main --motion-threshold 0.04 --confidence-threshold 0.6
    Ajusta sensibilidad de movimiento y confianza minima para aceptar palabras.

uv run python -m app.main --confidence-threshold 0.9 --margin-threshold 0.35 --distance-scale 1.0 --min-class-distance-threshold 0.0
    Modo mas estricto: rechaza mas movimientos que no coinciden con el dataset.

uv run python -m app.main --min-class-distance-threshold 2.5
    Permite mas variacion entre la sena en vivo y las muestras guardadas.

uv run python -m app.main --bayes-smoothing 0.2 --bayes-min-evidence 3
    Ajusta la suavizacion bayesiana y la evidencia minima para aceptar una sena.

Notas
-----
Usa MediaPipe para obtener landmarks, detecta automaticamente una sena por
movimiento, la remuestrea al tamano esperado por el modelo y muestra las
probabilidades calculadas por la red y estabilizadas con un filtro bayesiano.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import h5py
import numpy as np

from app.cli.collect_data import FEATURES_PER_FRAME, frame_to_features, smart_resample
from app.ml.bayesian_filter import BayesianSignFilter
from app.ml.neural_sign_classifier import NeuralSignClassifier
from app.vision.camera import Camera
from app.vision.hand_detector import HandDetector
from app.vision.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/neural_sign_classifier.npz")
DEFAULT_DATASET_PATH = Path("data/signs_dataset.h5")
DEFAULT_MOTION_THRESHOLD = 0.03
DEFAULT_SILENCE_FRAMES = 22
DEFAULT_MIN_SIGN_FRAMES = 14
DEFAULT_MAX_SIGN_FRAMES = 200
DEFAULT_RESULT_HOLD_S = 2.5
DEFAULT_TOP_K = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MARGIN_THRESHOLD = 0.20
DEFAULT_MIN_DETECTED_FRAMES = 12
DEFAULT_MIN_DETECTED_RATIO = 0.45
DEFAULT_DISTANCE_SCALE = 1.5
DEFAULT_MIN_CLASS_DISTANCE_THRESHOLD = 1.20
DEFAULT_BAYES_SMOOTHING = 0.15
DEFAULT_BAYES_MIN_EVIDENCE = 3

_C_GREEN = (0, 210, 80)
_C_RED = (0, 40, 220)
_C_YELLOW = (0, 210, 230)
_C_WHITE = (230, 230, 230)
_C_GRAY = (150, 150, 150)
_C_CYAN = (220, 200, 0)
_C_BLACK = (0, 0, 0)


class SignerState(Enum):
    WAITING = auto()
    SIGNING = auto()


@dataclass
class ClassReference:
    label: str
    samples: np.ndarray
    threshold: float


@dataclass
class RecognitionResult:
    candidates: list[tuple[str, float]]
    accepted: bool
    reject_reason: str | None = None
    distance: float | None = None
    distance_threshold: float | None = None
    evidence_count: int = 0

    @property
    def best_label(self) -> str:
        return self.candidates[0][0] if self.candidates else ""

    @property
    def best_probability(self) -> float:
        return self.candidates[0][1] if self.candidates else 0.0


class MotionSegmenter:
    """Detecta inicio y fin de una sena usando velocidad de landmarks."""

    def __init__(
        self,
        motion_threshold: float,
        silence_frames: int,
        min_sign_frames: int,
        max_sign_frames: int,
        smooth_window: int = 5,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.silence_frames = silence_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames

        self.state = SignerState.WAITING
        self.current_velocity = 0.0
        self._buffer: list[np.ndarray] = []
        self._detected: list[bool] = []
        self._prev_features: np.ndarray | None = None
        self._silence_count = 0
        self._velocity_window: deque[float] = deque(maxlen=smooth_window)

    def update(self, features: np.ndarray, hand_detected: bool) -> bool:
        """Procesa un frame y devuelve True cuando una sena esta lista."""
        if self._prev_features is not None and hand_detected:
            velocity = float(np.linalg.norm(features - self._prev_features))
        else:
            velocity = 0.0

        self._prev_features = features.copy()
        self._velocity_window.append(velocity)
        self.current_velocity = float(np.mean(self._velocity_window))
        moving = self.current_velocity > self.motion_threshold

        if self.state == SignerState.WAITING:
            if moving and hand_detected:
                self.state = SignerState.SIGNING
                self._buffer = [features]
                self._detected = [hand_detected]
                self._silence_count = 0
            return False

        self._buffer.append(features)
        self._detected.append(hand_detected)

        if moving:
            self._silence_count = 0
        else:
            self._silence_count += 1

        sign_finished = (
            self._silence_count >= self.silence_frames
            or len(self._buffer) >= self.max_sign_frames
        )
        if not sign_finished:
            return False

        if len(self._buffer) < self.min_sign_frames:
            self.reset()
            return False

        return True

    def get_sequence(self) -> tuple[list[np.ndarray], list[bool]]:
        return list(self._buffer), list(self._detected)

    def reset(self) -> None:
        self.state = SignerState.WAITING
        self.current_velocity = 0.0
        self._buffer = []
        self._detected = []
        self._silence_count = 0
        self._velocity_window.clear()

    @property
    def buffered_frames(self) -> int:
        return len(self._buffer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconocedor LESCO en tiempo real usando red neuronal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Dataset usado para validar si la sena se parece a una clase conocida.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    parser.add_argument("--silence-frames", type=int, default=DEFAULT_SILENCE_FRAMES)
    parser.add_argument("--min-sign-frames", type=int, default=DEFAULT_MIN_SIGN_FRAMES)
    parser.add_argument("--max-sign-frames", type=int, default=DEFAULT_MAX_SIGN_FRAMES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--result-hold", type=float, default=DEFAULT_RESULT_HOLD_S)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Probabilidad minima para aceptar una palabra como segura.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
        help="Diferencia minima entre la primera y segunda probabilidad.",
    )
    parser.add_argument(
        "--min-detected-frames",
        type=int,
        default=DEFAULT_MIN_DETECTED_FRAMES,
        help="Frames minimos con mano detectada antes de intentar reconocer.",
    )
    parser.add_argument(
        "--min-detected-ratio",
        type=float,
        default=DEFAULT_MIN_DETECTED_RATIO,
        help="Proporcion minima de frames de la toma con mano detectada.",
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=DEFAULT_DISTANCE_SCALE,
        help="Tolerancia para validar distancia contra ejemplos del dataset.",
    )
    parser.add_argument(
        "--min-class-distance-threshold",
        type=float,
        default=DEFAULT_MIN_CLASS_DISTANCE_THRESHOLD,
        help="Umbral minimo de distancia por clase para no rechazar variaciones normales.",
    )
    parser.add_argument(
        "--bayes-smoothing",
        type=float,
        default=DEFAULT_BAYES_SMOOTHING,
        help="Suaviza la evidencia de la red para que una prediccion aislada no domine.",
    )
    parser.add_argument(
        "--bayes-min-evidence",
        type=int,
        default=DEFAULT_BAYES_MIN_EVIDENCE,
        help="Cantidad minima de predicciones temporales antes de aceptar una sena.",
    )
    return parser.parse_args()


def load_model(path: Path) -> NeuralSignClassifier:
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {path}. "
            "Entrenalo con: uv run python -m app.cli.train_neural_network "
            "--dataset data/signs_dataset.h5"
        )

    model = NeuralSignClassifier.load(path)
    if model.features_per_frame != FEATURES_PER_FRAME:
        raise ValueError(
            "El modelo espera "
            f"{model.features_per_frame} features por frame, pero el pipeline produce "
            f"{FEATURES_PER_FRAME}."
        )
    return model


def sequence_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Distancia promedio por frame entre dos secuencias ya remuestreadas."""
    return float(np.linalg.norm(seq_a - seq_b, axis=1).mean())


def min_sequence_distance(sequence: np.ndarray, samples: np.ndarray) -> float:
    """Distancia contra la muestra mas parecida de una clase."""
    distances = np.linalg.norm(samples - sequence[None, :, :], axis=2).mean(axis=1)
    return float(distances.min())


def build_temporal_evidence(
    frames: list[np.ndarray],
    detected: list[bool],
    target_length: int,
) -> list[tuple[np.ndarray, float]]:
    """
    Genera varias vistas temporales de una misma sena.

    Esto simula varias predicciones seguidas sobre la toma completa y reduce
    el impacto de frames iniciales/finales ruidosos.
    """
    total = len(frames)
    if total == 0:
        return []

    windows: list[tuple[int, int, float]] = [(0, total, 1.2)]
    trim_10 = max(1, int(total * 0.10))
    trim_15 = max(1, int(total * 0.15))

    if total - trim_10 >= 12:
        windows.append((trim_10, total, 0.8))
        windows.append((0, total - trim_10, 0.8))

    if total - (2 * trim_15) >= 12:
        windows.append((trim_15, total - trim_15, 0.7))

    evidence: list[np.ndarray] = []
    for start, end, weight in windows:
        sequence = smart_resample(frames[start:end], detected[start:end], target_length)
        evidence.append((sequence, weight))
    return evidence


def _decode_labels(raw_labels: np.ndarray) -> np.ndarray:
    return np.array(
        [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in raw_labels
        ],
        dtype=str,
    )


def load_class_references(
    dataset_path: Path,
    model: NeuralSignClassifier,
    distance_scale: float,
    min_class_distance_threshold: float,
) -> dict[str, ClassReference]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {dataset_path}. "
            "Se necesita para rechazar movimientos que no coinciden con senas conocidas."
        )

    with h5py.File(dataset_path, "r") as h5_file:
        X = h5_file["X"][:].astype(np.float32)
        labels = _decode_labels(h5_file["labels"][:])

    if X.shape[1:] != (model.sequence_length, model.features_per_frame):
        raise ValueError(
            "El dataset y el modelo no tienen la misma forma: "
            f"dataset={X.shape[1:]}, modelo={(model.sequence_length, model.features_per_frame)}."
        )

    samples_by_label: dict[str, np.ndarray] = {}
    references: dict[str, ClassReference] = {}

    for label in model.labels:
        samples = X[labels == label]
        if len(samples) == 0:
            LOGGER.warning("La etiqueta %s existe en el modelo pero no en el dataset.", label)
            continue

        samples_by_label[label] = samples.astype(np.float32)

    for label, samples in samples_by_label.items():
        intra_distances: list[float] = []
        if len(samples) >= 2:
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    intra_distances.append(sequence_distance(samples[i], samples[j]))

        distances_to_other_classes = []
        for other_label, other_samples in samples_by_label.items():
            if other_label == label:
                continue
            for sample in samples:
                distances_to_other_classes.append(min_sequence_distance(sample, other_samples))

        nearest_other = min(distances_to_other_classes) if distances_to_other_classes else np.inf

        if intra_distances:
            intra = np.array(intra_distances, dtype=np.float32)
            learned_radius = float(np.median(intra) + distance_scale * intra.std())
        else:
            learned_radius = min_class_distance_threshold

        if np.isfinite(nearest_other):
            threshold = max(learned_radius, min_class_distance_threshold)
            threshold = min(threshold, max(min_class_distance_threshold, float(nearest_other * 0.90)))
        else:
            threshold = max(learned_radius, min_class_distance_threshold)

        references[label] = ClassReference(
            label=label,
            samples=samples,
            threshold=threshold,
        )
        LOGGER.info(
            "Referencia %s: %d muestra(s), umbral_distancia=%.4f",
            label,
            len(samples),
            threshold,
        )

    return references


def classify_sequence(
    model: NeuralSignClassifier,
    references: dict[str, ClassReference],
    frames: list[np.ndarray],
    detected: list[bool],
    top_k: int,
    confidence_threshold: float,
    margin_threshold: float,
    min_detected_frames: int,
    min_detected_ratio: float,
    bayes_smoothing: float,
    bayes_min_evidence: int,
) -> RecognitionResult:
    detected_count = sum(detected)
    detected_ratio = detected_count / max(len(detected), 1)
    if detected_count < min_detected_frames or detected_ratio < min_detected_ratio:
        return RecognitionResult(
            candidates=[],
            accepted=False,
            reject_reason=(
                f"mano insuficiente ({detected_count} frames, "
                f"{detected_ratio:.0%} de la toma)"
            ),
        )

    evidence_sequences = build_temporal_evidence(frames, detected, model.sequence_length)
    if len(evidence_sequences) < bayes_min_evidence:
        return RecognitionResult(
            candidates=[],
            accepted=False,
            reject_reason=f"evidencia insuficiente ({len(evidence_sequences)} predicciones)",
            evidence_count=len(evidence_sequences),
        )

    bayes_filter = BayesianSignFilter(model.labels, smoothing=bayes_smoothing)
    for sequence_view, weight in evidence_sequences:
        probabilities = model.predict_proba(sequence_view)[0]
        bayes_filter.update(probabilities, weight=weight)

    candidates = bayes_filter.top_k(top_k)
    sequence = evidence_sequences[0][0]

    best_label, best_probability = candidates[0]
    second_probability = candidates[1][1] if len(candidates) > 1 else 0.0
    margin = best_probability - second_probability

    if best_probability < confidence_threshold:
        return RecognitionResult(
            candidates=candidates,
            accepted=False,
            reject_reason=f"confianza baja ({best_probability:.0%})",
            evidence_count=bayes_filter.update_count,
        )

    if margin < margin_threshold:
        return RecognitionResult(
            candidates=candidates,
            accepted=False,
            reject_reason=f"margen bajo ({margin:.0%})",
            evidence_count=bayes_filter.update_count,
        )

    reference = references.get(best_label)
    if reference is None:
        return RecognitionResult(
            candidates=candidates,
            accepted=False,
            reject_reason=f"sin referencia para {best_label}",
            evidence_count=bayes_filter.update_count,
        )

    class_distances = {
        label: min_sequence_distance(sequence, ref.samples)
        for label, ref in references.items()
    }
    nearest_label = min(class_distances, key=class_distances.get)
    if nearest_label != best_label:
        return RecognitionResult(
            candidates=candidates,
            accepted=False,
            reject_reason=(
                f"red dice {best_label}, pero el movimiento se parece mas a {nearest_label}"
            ),
            distance=class_distances[best_label],
            distance_threshold=reference.threshold,
            evidence_count=bayes_filter.update_count,
        )

    distance = class_distances[best_label]
    if distance > reference.threshold:
        return RecognitionResult(
            candidates=candidates,
            accepted=False,
            reject_reason=(
                f"movimiento fuera de la clase {best_label} "
                f"(dist {distance:.3f} > {reference.threshold:.3f})"
            ),
            distance=distance,
            distance_threshold=reference.threshold,
            evidence_count=bayes_filter.update_count,
        )

    return RecognitionResult(
        candidates=candidates,
        accepted=True,
        distance=distance,
        distance_threshold=reference.threshold,
        evidence_count=bayes_filter.update_count,
    )


def _text(
    frame: cv2.typing.MatLike,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = _C_WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(
    frame: cv2.typing.MatLike,
    segmenter: MotionSegmenter,
    result: RecognitionResult | None,
    result_expires: float,
    frame_counter: int,
) -> None:
    h, w = frame.shape[:2]
    now = time.monotonic()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), _C_BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if segmenter.state == SignerState.SIGNING:
        state_text = "CAPTURANDO..."
        if (frame_counter // 12) % 2 == 0:
            state_text = "REC " + state_text
        _text(frame, state_text, (12, 30), 0.75, _C_RED, 2)
        _text(frame, f"Frames: {segmenter.buffered_frames}", (12, 58), 0.55, _C_GRAY)
    else:
        _text(frame, "ESPERANDO SENA", (12, 30), 0.75, _C_GRAY, 2)

    _text(frame, "Modelo: red neuronal + Bayes", (12, 84), 0.5, _C_CYAN)
    _text(frame, "Q: salir", (w - 90, 30), 0.5, _C_GRAY)

    bar_y = 90
    bar_w = w - 24
    ratio = min(segmenter.current_velocity / (segmenter.motion_threshold * 3), 1.0)
    filled = int(bar_w * ratio)
    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + 8), (50, 50, 50), -1)
    if filled > 0:
        cv2.rectangle(frame, (12, bar_y), (12 + filled, bar_y + 8), _C_RED, -1)

    if result is not None and now < result_expires and result.accepted:
        shown_label = result.best_label
        best_probability = result.best_probability
        box_y = h // 2 - 60
        panel = frame.copy()
        cv2.rectangle(panel, (0, box_y), (w, box_y + 82), _C_BLACK, -1)
        cv2.addWeighted(panel, 0.58, frame, 0.42, 0, frame)

        font_scale = 1.7 if len(shown_label) <= 12 else 1.25
        (tw, _), _ = cv2.getTextSize(shown_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
        cv2.putText(
            frame,
            shown_label,
            ((w - tw) // 2, box_y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _C_GREEN,
            3,
        )

        confidence_text = f"{shown_label}: {best_probability * 100:.0f}%"
        _text(frame, confidence_text, (12, box_y + 76), 0.55, _C_GREEN, 2)

        alt_y = box_y + 110
        for idx, (label, probability) in enumerate(result.candidates[1:], start=2):
            _text(
                frame,
                f"{idx}. {label}: {probability * 100:.0f}%",
                (12, alt_y + (idx - 2) * 22),
                0.52,
                _C_GRAY,
            )
    elif segmenter.state == SignerState.WAITING:
        msg = "Realiza una sena frente a la camara"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        _text(frame, msg, ((w - tw) // 2, h - 20), 0.6, _C_CYAN)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = load_model(args.model)
    LOGGER.info("Modelo cargado: %s", args.model)
    LOGGER.info("Etiquetas: %s", ", ".join(model.labels))
    references = load_class_references(
        args.dataset,
        model,
        args.distance_scale,
        args.min_class_distance_threshold,
    )

    preprocessor = Preprocessor(flip_horizontal=True)
    segmenter = MotionSegmenter(
        motion_threshold=args.motion_threshold,
        silence_frames=args.silence_frames,
        min_sign_frames=args.min_sign_frames,
        max_sign_frames=args.max_sign_frames,
    )

    current_result: RecognitionResult | None = None
    result_expires = 0.0
    frame_counter = 0

    try:
        with Camera(camera_index=args.camera_index) as camera, HandDetector(max_hands=2) as detector:
            LOGGER.info("Reconocedor iniciado. Presiona Q en la ventana para salir.")
            for frame_bgr in camera.stream():
                frame_counter += 1
                frame_bgr = preprocessor.process(frame_bgr)
                detection = detector.detect(frame_bgr)
                annotated = detector.draw(frame_bgr, detection)

                features, hand_detected = frame_to_features(detection)
                if segmenter.update(features, hand_detected):
                    frames, detected = segmenter.get_sequence()
                    result = classify_sequence(
                        model,
                        references,
                        frames,
                        detected,
                        args.top_k,
                        args.confidence_threshold,
                        args.margin_threshold,
                        args.min_detected_frames,
                        args.min_detected_ratio,
                        args.bayes_smoothing,
                        args.bayes_min_evidence,
                    )
                    segmenter.reset()

                    if result.accepted:
                        current_result = result
                        result_expires = time.monotonic() + args.result_hold
                        LOGGER.info(
                            "Prediccion aceptada: %s (%.1f%%, dist %.3f/%.3f, evidencias=%d)",
                            result.best_label,
                            result.best_probability * 100,
                            result.distance,
                            result.distance_threshold,
                            result.evidence_count,
                        )
                    else:
                        current_result = None
                        result_expires = 0.0
                        candidate_text = (
                            f"{result.best_label} {result.best_probability * 100:.1f}%"
                            if result.candidates
                            else "sin candidato"
                        )
                        LOGGER.info(
                            "Movimiento rechazado: %s (%s, evidencias=%d).",
                            result.reject_reason,
                            candidate_text,
                            result.evidence_count,
                        )

                if current_result is not None and time.monotonic() >= result_expires:
                    current_result = None

                draw_hud(
                    annotated,
                    segmenter,
                    current_result,
                    result_expires,
                    frame_counter,
                )

                cv2.imshow("LESCO-AI Neural Recognizer", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("Salida solicitada con Q.")
                    break
            if frame_counter == 0:
                LOGGER.warning(
                    "La camara abrio, pero no entrego frames. "
                    "Prueba con --camera-index 1 o verifica permisos/uso de la camara."
                )
    except KeyboardInterrupt:
        LOGGER.info("Ejecucion interrumpida por el usuario.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
