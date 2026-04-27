import argparse
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from app.collect_data import frame_to_features
from app.vision.camera import Camera
from app.vision.hand_detector import HandDetector
from app.vision.preprocessor import Preprocessor


LOGGER = logging.getLogger(__name__)


def load_dataset(path: Path, labels_filter: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset: {path}")

    dataset = np.load(path, allow_pickle=False)
    x_data = dataset["X"].astype(np.float32)
    y_data = dataset["y"].astype(str)

    if labels_filter:
        mask = np.array([label in labels_filter for label in y_data])
        missing = labels_filter - set(y_data)
        if missing:
            raise ValueError(f"Estas etiquetas no existen en el dataset: {sorted(missing)}")
        x_data = x_data[mask]
        y_data = y_data[mask]

    if len(x_data) == 0:
        raise ValueError("El dataset no tiene muestras para comparar.")

    return x_data, y_data


def distance_to_samples(sequence: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Return one RMS distance per saved sample."""
    diff = samples - sequence[np.newaxis, :, :]
    return np.sqrt(np.mean(diff * diff, axis=(1, 2)))


def compare_by_label(
    sequence: np.ndarray,
    samples: np.ndarray,
    labels: np.ndarray,
) -> list[tuple[str, float, float, int]]:
    """Compare current sequence against dataset and return label, probability, distance, count."""
    sample_distances = distance_to_samples(sequence, samples)
    unique_labels = sorted(set(labels))

    label_distances = []
    for label in unique_labels:
        distances = sample_distances[labels == label]
        label_distances.append((label, float(np.min(distances)), int(len(distances))))

    distances = np.array([item[1] for item in label_distances], dtype=np.float32)
    if len(distances) == 1:
        probabilities = np.array([1.0], dtype=np.float32)
    else:
        temperature = max(float(np.std(distances)), 1e-6)
        logits = -(distances - float(np.min(distances))) / temperature
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

    results = [
        (label, float(prob), distance, count)
        for (label, distance, count), prob in zip(label_distances, probabilities)
    ]
    return sorted(results, key=lambda item: item[1], reverse=True)


def sequence_motion_score(sequence: np.ndarray) -> float:
    """Return average frame-to-frame movement for the current sequence."""
    if len(sequence) < 2:
        return 0.0

    deltas = np.diff(sequence, axis=0)
    frame_motion = np.sqrt(np.mean(deltas * deltas, axis=1))
    return float(np.mean(frame_motion))


def draw_results(
    frame: cv2.typing.MatLike,
    results: list[tuple[str, float, float, int]] | None,
    frames_ready: int,
    sequence_length: int,
    min_probability: float,
    top_k: int,
    hand_ratio: float,
    min_hand_ratio: float,
    motion_score: float,
    min_motion: float,
) -> None:
    cv2.rectangle(frame, (0, 0), (520, 180), (0, 0, 0), -1)

    if results is None:
        text = f"Recolectando ventana: {frames_ready}/{sequence_length}"
        cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Mano visible: {hand_ratio * 100:.0f}% / minimo {min_hand_ratio * 100:.0f}%",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            frame,
            f"Movimiento: {motion_score:.4f} / minimo {min_motion:.4f}",
            (12, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
        )
        cv2.putText(frame, "Q: salir", (12, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        return

    best_label, best_prob, _, _ = results[0]
    prediction = best_label if best_prob >= min_probability else "INCIERTO"
    color = (0, 255, 0) if prediction != "INCIERTO" else (0, 255, 255)

    cv2.putText(
        frame,
        f"Prediccion: {prediction} ({best_prob * 100:.1f}%)",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    y = 68
    for label, probability, distance, count in results[:top_k]:
        bar_width = int(250 * probability)
        cv2.rectangle(frame, (150, y - 16), (400, y - 2), (60, 60, 60), -1)
        cv2.rectangle(frame, (150, y - 16), (150 + bar_width, y - 2), (0, 180, 255), -1)
        line = f"{label}: {probability * 100:5.1f}% d={distance:.3f} n={count}"
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        y += 28

    cv2.putText(frame, "Q: salir", (12, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara senas en tiempo real contra el dataset de landmarks.",
    )
    parser.add_argument("--dataset", default="data/signs_dataset.npz")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--labels", help="Etiquetas separadas por coma. Ej: HOLA,GRACIAS,AGUA")
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument(
        "--min-hand-ratio",
        type=float,
        default=0.45,
        help="Proporcion minima de frames de la ventana donde debe verse una mano.",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.012,
        help="Movimiento minimo promedio requerido para comparar una sena dinamica.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    labels_filter = None
    if args.labels:
        labels_filter = {label.strip().upper().replace(" ", "_") for label in args.labels.split(",")}

    dataset_path = Path(args.dataset)
    samples, labels = load_dataset(dataset_path, labels_filter=labels_filter)
    sequence_length = samples.shape[1]
    feature_count = samples.shape[2]

    LOGGER.info("Dataset: %s", dataset_path)
    LOGGER.info("Muestras: %s | Etiquetas: %s", len(samples), sorted(set(labels)))
    LOGGER.info("Ventana temporal: %s frames | Features por frame: %s", sequence_length, feature_count)

    preprocessor = Preprocessor(flip_horizontal=True)
    sequence_window: deque[np.ndarray] = deque(maxlen=sequence_length)
    hand_window: deque[bool] = deque(maxlen=sequence_length)

    with Camera(camera_index=args.camera_index) as camera, HandDetector(max_hands=2) as detector:
        for frame in camera.stream():
            frame = preprocessor.process(frame)
            result = detector.detect(frame)
            annotated = detector.draw(frame, result)

            sequence_window.append(frame_to_features(result))
            hand_window.append(result.has_hands)

            comparison = None
            hand_ratio = sum(hand_window) / len(hand_window)
            motion_score = 0.0
            if len(sequence_window) == sequence_length:
                current_sequence = np.array(sequence_window, dtype=np.float32)
                motion_score = sequence_motion_score(current_sequence)
                if hand_ratio >= args.min_hand_ratio and motion_score >= args.min_motion:
                    comparison = compare_by_label(current_sequence, samples, labels)

            draw_results(
                annotated,
                results=comparison,
                frames_ready=len(sequence_window),
                sequence_length=sequence_length,
                min_probability=args.min_probability,
                top_k=args.top_k,
                hand_ratio=hand_ratio,
                min_hand_ratio=args.min_hand_ratio,
                motion_score=motion_score,
                min_motion=args.min_motion,
            )

            cv2.imshow("LESCO-AI Realtime Compare", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                LOGGER.info("Saliendo del comparador.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
