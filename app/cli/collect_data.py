"""

Funcion
-------
Captura landmarks de manos con la webcam, normaliza cada frame y guarda
muestras de senas en un dataset HDF5.

Comandos
--------
uv run python -m app.cli.collect_data --label HOLA
    Graba muestras para la etiqueta HOLA en data/signs_dataset.h5.

uv run python -m app.cli.collect_data --label BUENOS_DIAS --sequence-length 60
    Graba BUENOS_DIAS y remuestrea cada toma a 60 frames.

uv run python -m app.cli.collect_data --label HOLA --output data/prueba.h5
    Guarda las muestras en un archivo HDF5 diferente.

Notas
-----
Teclas durante la captura: R inicia, S guarda la toma actual y Q sale.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from app.storage.dataset import LESCODataset
from app.vision.camera import Camera
from app.vision.hand_detector import DetectedHand, DetectionResult, HandDetector
from app.vision.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

# ── Feature layout ─────────────────────────────────────────────────────────────
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3          # x, y, z from MediaPipe
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63

# Semantic slot assignment: the model always sees these at fixed indices.
HAND_SLOTS: dict[str, int] = {"Right": 0, "Left": 1}
MAX_HANDS = len(HAND_SLOTS)
FEATURES_PER_FRAME = FEATURES_PER_HAND * MAX_HANDS  # 126


# ── State machine ──────────────────────────────────────────────────────────────
class State(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    SAVING = auto()


# ── Landmark normalisation ─────────────────────────────────────────────────────

def normalize_hand(hand: DetectedHand) -> np.ndarray:
    """
    Return the 21 landmarks of *hand* as a 63-element float32 vector,
    translated to the wrist and scaled by the bounding range.

    This makes the features invariant to hand size and screen position,
    which is critical for generalisation across different people and
    camera distances.
    """
    # Shape (21, 3): each row is (x_norm, y_norm, z_norm) from MediaPipe.
    pts = np.array(hand.landmarks_norm, dtype=np.float32)   # (21, 3)

    # Translate: wrist (landmark 0) becomes the origin.
    pts -= pts[0]

    # Scale: normalise by the largest spatial range so the hand fits in [-1, 1].
    scale = float(np.ptp(pts[:, :2]))  # peak-to-peak of x and y only
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    return pts.reshape(-1)  # (63,)


def frame_to_features(result: DetectionResult) -> tuple[np.ndarray, bool]:
    """
    Convert a DetectionResult into a fixed-length feature vector (126 floats)
    and a boolean indicating whether at least one hand was actually detected.

    Slot assignment
    ---------------
    Slot 0 (indices   0–62 ) → Right hand (or zeros if absent)
    Slot 1 (indices  63–125) → Left  hand (or zeros if absent)

    Using label-based slots instead of sorting by x-position guarantees that
    the same feature indices always represent the same semantic hand across
    all frames and all signs, which is a hard requirement for LSTM training.
    """
    features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    any_detected = False

    for hand in result.hands:
        slot = HAND_SLOTS.get(hand.hand_label)
        if slot is None:
            continue  # unexpected label; skip
        start = slot * FEATURES_PER_HAND
        features[start : start + FEATURES_PER_HAND] = normalize_hand(hand)
        any_detected = True

    return features, any_detected


# ── Gap-aware resampling ───────────────────────────────────────────────────────

def smart_resample(
    frames: list[np.ndarray],
    detected: list[bool],
    target_length: int,
) -> np.ndarray:
    """
    Resample a variable-length recording to *target_length* frames.

    Unlike simple linear interpolation over all frames, this function uses
    only the frames where a hand was actually detected as "anchor points".
    Gaps (dropped frames / momentary occlusions) are therefore bridged by
    interpolating across the *real* motion data on either side, not by
    dragging zeros into the interpolated sequence.

    Args:
        frames:        List of feature vectors (one per raw captured frame).
        detected:      Parallel list; True if at least one hand was seen.
        target_length: Number of output frames (e.g. 60).

    Returns:
        Float32 array of shape (target_length, FEATURES_PER_FRAME).
    """
    anchor_indices = [i for i, d in enumerate(detected) if d]
    anchor_data = [frames[i] for i in anchor_indices]

    if not anchor_indices:
        LOGGER.warning("smart_resample: no detected frames – returning zeros.")
        return np.zeros((target_length, FEATURES_PER_FRAME), dtype=np.float32)

    if len(anchor_indices) == 1:
        # Single detected frame: tile it across all target frames.
        return np.tile(anchor_data[0], (target_length, 1)).astype(np.float32)

    total = len(frames)
    # Map each anchor to its relative position in [0, 1].
    old_pos = np.array(anchor_indices, dtype=np.float64) / max(total - 1, 1)
    new_pos = np.linspace(0.0, 1.0, target_length)

    data = np.array(anchor_data, dtype=np.float32)  # (n_anchors, features)
    resampled = np.empty((target_length, data.shape[1]), dtype=np.float32)

    for feat_idx in range(data.shape[1]):
        resampled[:, feat_idx] = np.interp(new_pos, old_pos, data[:, feat_idx])

    return resampled


# ── On-screen drawing helpers ──────────────────────────────────────────────────

def _put_text_centered(
    frame: cv2.typing.MatLike,
    text: str,
    center_y: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw *text* horizontally centred on *frame* at vertical position *center_y*."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cx = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (cx, center_y + th // 2), font, font_scale, color, thickness)


def draw_countdown(frame: cv2.typing.MatLike, seconds_left: float) -> None:
    """Render a large semi-transparent countdown overlay."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    number = str(max(1, int(np.ceil(seconds_left))))
    _put_text_centered(frame, number, h // 2 - 20, 6.0, (0, 255, 255), 8)
    _put_text_centered(frame, "Preparate...", h // 2 + 80, 1.0, (255, 255, 255))


def draw_status(
    frame: cv2.typing.MatLike,
    label: str,
    state: State,
    recorded_frames: int,
    detected_frames: int,
    total_samples: int | None,
    frame_counter: int,
) -> None:
    """
    Render the HUD overlay on the camera frame.

    The "● REC" indicator blinks every 15 screen frames so it is obvious
    even at a glance that the script is actively capturing.
    """
    h, w = frame.shape[:2]

    # ── Status badge (top-right corner) ────────────────────────────────────────
    if state == State.RECORDING:
        blink_on = (frame_counter // 15) % 2 == 0
        badge_text = "● REC" if blink_on else "  REC"
        badge_color = (0, 0, 255) if blink_on else (0, 0, 180)
        cv2.putText(frame, badge_text, (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, badge_color, 2)

    # ── Info panel (top-left corner) ───────────────────────────────────────────
    state_labels = {
        State.IDLE: ("LISTO", (0, 200, 0)),
        State.COUNTDOWN: ("CUENTA...", (0, 200, 255)),
        State.RECORDING: ("GRABANDO", (0, 0, 255)),
        State.SAVING: ("GUARDANDO", (200, 200, 0)),
    }
    state_text, state_color = state_labels[state]

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"Etiqueta : {label}", (220, 220, 220)),
        (f"Estado   : {state_text}", state_color),
    ]

    if state == State.RECORDING:
        gap_frames = recorded_frames - detected_frames
        lines.append((f"Frames   : {recorded_frames}  (gaps: {gap_frames})", (200, 200, 200)))

    if total_samples is not None:
        lines.append((f"Muestras : {total_samples}", (180, 255, 180)))

    lines.append(("R: grabar  S: guardar  Q: salir", (160, 160, 160)))

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # ── Hand slot indicators (bottom bar) ──────────────────────────────────────
    bar_y = h - 20
    cv2.putText(frame, "Slot 0: Derecha", (10, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    cv2.putText(frame, "Slot 1: Izquierda", (w // 2, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura muestras de landmarks LESCO y las guarda en HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label", required=True,
        help="Palabra o etiqueta. Ej: HOLA, BUENOS_DIAS",
    )
    parser.add_argument(
        "--output", default="data/signs_dataset.h5",
        help="Archivo HDF5 donde se acumulan las muestras.",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=60,
        help="Frames por muestra después del remuestreo.",
    )
    parser.add_argument(
        "--min-frames", type=int, default=12,
        help="Mínimo de frames capturados para aceptar la muestra.",
    )
    parser.add_argument(
        "--countdown-seconds", type=float, default=3.0,
        help="Segundos de cuenta regresiva antes de grabar.",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
    )
    return parser.parse_args()


# ── Console key helper (Windows only) ─────────────────────────────────────────

def _read_console_key() -> str | None:
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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    label = args.label.upper().strip().replace(" ", "_")
    output_path = Path(args.output)
    preprocessor = Preprocessor(flip_horizontal=True)

    LOGGER.info("Etiqueta: %s", label)
    LOGGER.info("Salida  : %s", output_path)
    LOGGER.info("R=grabar  S=guardar  Q=salir")

    state = State.IDLE
    current_frames: list[np.ndarray] = []
    current_detected: list[bool] = []
    countdown_end: float = 0.0
    total_samples: int | None = None
    frame_counter: int = 0

    dataset = LESCODataset(
        path=output_path,
        sequence_length=args.sequence_length,
        features_per_frame=FEATURES_PER_FRAME,
    )

    # ── Main capture loop ──────────────────────────────────────────────────────
    # The outer try/finally guarantees that the HDF5 file is always closed
    # cleanly, even on Ctrl-C or unexpected camera failure.
    try:
        with Camera(camera_index=args.camera_index) as camera, \
             HandDetector(max_hands=2) as detector:

            for frame in camera.stream():
                frame_counter += 1
                frame = preprocessor.process(frame)
                result = detector.detect(frame)
                annotated = detector.draw(frame, result)

                # ── State transitions ──────────────────────────────────────────
                if state == State.COUNTDOWN:
                    remaining = countdown_end - time.monotonic()
                    if remaining <= 0.0:
                        state = State.RECORDING
                        current_frames = []
                        current_detected = []
                        LOGGER.info("Grabación iniciada.")
                    else:
                        draw_countdown(annotated, remaining)

                if state == State.RECORDING:
                    feat, detected = frame_to_features(result)
                    current_frames.append(feat)
                    current_detected.append(detected)

                # ── Draw HUD ──────────────────────────────────────────────────
                draw_status(
                    annotated,
                    label=label,
                    state=state,
                    recorded_frames=len(current_frames),
                    detected_frames=sum(current_detected),
                    total_samples=total_samples,
                    frame_counter=frame_counter,
                )

                cv2.imshow("LESCO-AI Data Collector", annotated)

                # ── Key reading ───────────────────────────────────────────────
                window_key = cv2.waitKey(1) & 0xFF
                key = chr(window_key).lower() if window_key != 255 else None
                key = key or _read_console_key()

                if key == "r" and state == State.IDLE:
                    state = State.COUNTDOWN
                    countdown_end = time.monotonic() + args.countdown_seconds
                    LOGGER.info("Cuenta regresiva de %.0fs iniciada.", args.countdown_seconds)

                elif key == "s" and state == State.RECORDING:
                    state = State.SAVING
                    n_frames = len(current_frames)
                    n_detected = sum(current_detected)
                    gap_count = n_frames - n_detected

                    if n_frames < args.min_frames:
                        LOGGER.warning(
                            "Muestra descartada: %d frames (mínimo %d).",
                            n_frames, args.min_frames,
                        )
                        state = State.IDLE
                        current_frames = []
                        current_detected = []
                        continue

                    if n_detected < args.min_frames // 2:
                        LOGGER.warning(
                            "Muestra descartada: sólo %d/%d frames con detección real.",
                            n_detected, n_frames,
                        )
                        state = State.IDLE
                        current_frames = []
                        current_detected = []
                        continue

                    sample = smart_resample(
                        current_frames, current_detected, args.sequence_length
                    )
                    total_samples = dataset.append(
                        label=label,
                        sequence=sample,
                        original_length=n_frames,
                    )
                    LOGGER.info(
                        "Guardado: label=%s  frames_raw=%d  gaps=%d  total_muestras=%d",
                        label, n_frames, gap_count, total_samples,
                    )
                    current_frames = []
                    current_detected = []
                    state = State.IDLE

                elif key == "q":
                    LOGGER.info("Saliendo.")
                    break

    finally:
        dataset.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
