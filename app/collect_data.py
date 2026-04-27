import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from app.vision.camera import Camera
from app.vision.hand_detector import DetectedHand, DetectionResult, HandDetector
from app.vision.preprocessor import Preprocessor


LOGGER = logging.getLogger(__name__)
FEATURES_PER_HAND = 63
MAX_HANDS = 2
FEATURES_PER_FRAME = FEATURES_PER_HAND * MAX_HANDS


def read_console_key() -> str | None:
    """Read one key from the terminal without blocking, when supported."""
    if os.name != "nt":
        return None

    import msvcrt

    if not msvcrt.kbhit():
        return None

    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        msvcrt.getwch()
        return None

    return key.lower()


def normalize_hand(hand: DetectedHand) -> np.ndarray:
    """Return 21 hand landmarks normalized around the wrist and hand size."""
    landmarks = np.array(hand.landmarks_norm, dtype=np.float32)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    x_range = np.ptp(landmarks[:, 0])
    y_range = np.ptp(landmarks[:, 1])
    scale = max(float(x_range), float(y_range), 1e-6)
    landmarks = landmarks / scale

    return landmarks.reshape(-1)


def frame_to_features(result: DetectionResult) -> np.ndarray:
    """Encode up to two detected hands into one fixed-length feature vector."""
    features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

    hands = sorted(
        result.hands,
        key=lambda hand: sum(point[0] for point in hand.landmarks_px) / len(hand.landmarks_px),
    )[:MAX_HANDS]

    for index, hand in enumerate(hands):
        start = index * FEATURES_PER_HAND
        end = start + FEATURES_PER_HAND
        features[start:end] = normalize_hand(hand)

    return features


def resample_sequence(sequence: list[np.ndarray], target_length: int) -> np.ndarray:
    """Convert a variable-length recording into a fixed number of frames."""
    data = np.array(sequence, dtype=np.float32)
    if len(data) == target_length:
        return data

    old_positions = np.linspace(0, len(data) - 1, num=len(data))
    new_positions = np.linspace(0, len(data) - 1, num=target_length)

    resampled = np.empty((target_length, data.shape[1]), dtype=np.float32)
    for feature_index in range(data.shape[1]):
        resampled[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            data[:, feature_index],
        )

    return resampled


def append_sample(
    output_path: Path,
    label: str,
    sample: np.ndarray,
    original_length: int,
) -> int:
    """Append one labeled sample to the dataset file and return total samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = np.load(output_path, allow_pickle=False)
        x_data = existing["X"]
        y_data = existing["y"]
        lengths = existing["original_lengths"]

        x_data = np.concatenate([x_data, sample[np.newaxis, ...]], axis=0)
        y_data = np.concatenate([y_data, np.array([label])], axis=0)
        lengths = np.concatenate([lengths, np.array([original_length], dtype=np.int32)])
    else:
        x_data = sample[np.newaxis, ...]
        y_data = np.array([label])
        lengths = np.array([original_length], dtype=np.int32)

    np.savez_compressed(
        output_path,
        X=x_data,
        y=y_data,
        original_lengths=lengths,
        sequence_length=np.array([sample.shape[0]], dtype=np.int32),
        features_per_frame=np.array([sample.shape[1]], dtype=np.int32),
    )

    return len(y_data)


def draw_status(
    frame: cv2.typing.MatLike,
    label: str,
    is_recording: bool,
    recorded_frames: int,
    total_samples: int | None,
) -> None:
    status = "GRABANDO" if is_recording else "LISTO"
    color = (0, 0, 255) if is_recording else (0, 255, 0)

    lines = [
        f"Etiqueta: {label}",
        f"Estado: {status}",
        f"Frames muestra: {recorded_frames}",
        "R: iniciar | S: finalizar y guardar | Q: salir",
        "Usa la ventana o la terminal mientras el script corre",
    ]

    if total_samples is not None:
        lines.append(f"Muestras guardadas: {total_samples}")

    for index, line in enumerate(lines):
        y = 30 + index * 28
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura muestras de landmarks para entrenar senas LESCO.",
    )
    parser.add_argument("--label", required=True, help="Palabra o etiqueta a registrar. Ej: HOLA")
    parser.add_argument(
        "--output",
        default="data/signs_dataset.npz",
        help="Archivo donde se acumulan las muestras.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Cantidad fija de frames guardados por muestra despues del remuestreo.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=12,
        help="Minimo de frames requeridos para aceptar una muestra.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    label = args.label.upper().strip().replace(" ", "_")
    output_path = Path(args.output)
    preprocessor = Preprocessor(flip_horizontal=True)

    is_recording = False
    current_sequence: list[np.ndarray] = []
    total_samples: int | None = None

    LOGGER.info("Registrando etiqueta: %s", label)
    LOGGER.info("Presiona R para iniciar, S para finalizar y guardar, Q para salir.")
    LOGGER.info("Las teclas funcionan en la ventana de camara o en esta terminal mientras el script este corriendo.")
    LOGGER.info("Salida: %s", output_path)

    with Camera(camera_index=args.camera_index) as camera, HandDetector(max_hands=2) as detector:
        for frame in camera.stream():
            frame = preprocessor.process(frame)
            result = detector.detect(frame)
            annotated = detector.draw(frame, result)

            if is_recording:
                current_sequence.append(frame_to_features(result))

            draw_status(
                annotated,
                label=label,
                is_recording=is_recording,
                recorded_frames=len(current_sequence),
                total_samples=total_samples,
            )

            cv2.imshow("LESCO-AI Data Collector", annotated)
            window_key = cv2.waitKey(1) & 0xFF
            key = chr(window_key).lower() if window_key != 255 else None
            key = key or read_console_key()

            if key == "r" and not is_recording:
                current_sequence = []
                is_recording = True
                LOGGER.info("Grabacion iniciada para %s.", label)

            elif key == "s" and is_recording:
                is_recording = False

                if len(current_sequence) < args.min_frames:
                    LOGGER.warning(
                        "Muestra descartada: %s frames. Minimo requerido: %s.",
                        len(current_sequence),
                        args.min_frames,
                    )
                    current_sequence = []
                    continue

                sample = resample_sequence(current_sequence, args.sequence_length)
                total_samples = append_sample(
                    output_path=output_path,
                    label=label,
                    sample=sample,
                    original_length=len(current_sequence),
                )
                LOGGER.info(
                    "Muestra guardada: label=%s frames_originales=%s frames_guardados=%s total=%s",
                    label,
                    len(current_sequence),
                    args.sequence_length,
                    total_samples,
                )
                current_sequence = []

            elif key == "q":
                LOGGER.info("Saliendo del capturador.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
