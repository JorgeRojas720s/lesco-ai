"""
app/api/server.py
=================

Servidor web del reconocedor LESCO.

Expone el MISMO pipeline de deteccion que ``app/main.py`` (MediaPipe + red
neuronal + filtro Bayesiano) a traves de HTTP, para que el frontend
(``app/frontend``) lo use en lugar de la ventana ``cv2.imshow``.

IMPORTANTE
----------
Este modulo NO implementa logica de deteccion. Solo *orquesta*, por cada frame
recibido, las MISMAS funciones/clases que usa el loop de ``app.main.main()``:
``load_model``, ``load_class_references``, ``Preprocessor``, ``HandDetector``,
``frame_to_features``, ``MotionSegmenter`` y ``classify_sequence``. La logica de
reconocimiento sigue viviendo intacta en sus modulos originales.

Arranque
--------
    uv run uvicorn app.api.server:app --port 8000

Luego abrir http://localhost:8000/

Variables de entorno opcionales:
    LESCO_MODEL    Ruta al modelo .npz   (default: models/neural_sign_classifier.npz)
    LESCO_DATASET  Ruta al dataset .h5   (default: data/signs_dataset.h5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.cli.collect_data import frame_to_features
from app.main import (
    DEFAULT_BAYES_MIN_EVIDENCE,
    DEFAULT_BAYES_SMOOTHING,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DATASET_PATH,
    DEFAULT_DISTANCE_SCALE,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MAX_SIGN_FRAMES,
    DEFAULT_MIN_CLASS_DISTANCE_THRESHOLD,
    DEFAULT_MIN_DETECTED_FRAMES,
    DEFAULT_MIN_DETECTED_RATIO,
    DEFAULT_MIN_SIGN_FRAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_RESULT_HOLD_S,
    DEFAULT_SILENCE_FRAMES,
    DEFAULT_TOP_K,
    MotionSegmenter,
    RecognitionResult,
    classify_sequence,
    load_class_references,
    load_model,
)
from app.vision.hand_detector import HandDetector
from app.vision.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_EMPTY_RESULT = {"accepted": False, "label": "", "confidence": 0, "candidates": []}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


class RecognizerSession:
    """Mantiene el estado del reconocedor entre frames HTTP.

    Reproduce, por frame, exactamente el cuerpo del loop de ``app.main.main()``:
    preprocesa, detecta manos, extrae features, segmenta el movimiento y, cuando
    una sena termina, la clasifica. Un ``Lock`` serializa el acceso porque el
    estado (buffer del segmentador, resultado vigente) es compartido.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        dataset_path: Path = DEFAULT_DATASET_PATH,
    ) -> None:
        self.model = load_model(model_path)
        LOGGER.info("Modelo cargado: %s", model_path)
        LOGGER.info("Etiquetas: %s", ", ".join(self.model.labels))
        self.references = load_class_references(
            dataset_path,
            self.model,
            DEFAULT_DISTANCE_SCALE,
            DEFAULT_MIN_CLASS_DISTANCE_THRESHOLD,
        )

        self.preprocessor = Preprocessor(flip_horizontal=True)
        self.detector = HandDetector(max_hands=2)
        self.detector.start()
        # Mismos defaults que el CLI (app.main). Ajustables por entorno si el
        # navegador no alcanza ~30 FPS: subí LESCO_MOTION_THRESHOLD o bajá
        # LESCO_SILENCE_FRAMES / LESCO_MIN_SIGN_FRAMES para que la seña cierre.
        self.segmenter = MotionSegmenter(
            motion_threshold=_env_float("LESCO_MOTION_THRESHOLD", DEFAULT_MOTION_THRESHOLD),
            silence_frames=_env_int("LESCO_SILENCE_FRAMES", DEFAULT_SILENCE_FRAMES),
            min_sign_frames=_env_int("LESCO_MIN_SIGN_FRAMES", DEFAULT_MIN_SIGN_FRAMES),
            max_sign_frames=_env_int("LESCO_MAX_SIGN_FRAMES", DEFAULT_MAX_SIGN_FRAMES),
        )
        self.confidence_threshold = _env_float(
            "LESCO_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD
        )
        # Cuánta "mano detectada" se exige para intentar clasificar. Si en el
        # navegador se pierde mano por recompresión/desenfoque, bajá estos:
        # LESCO_MIN_DETECTED_RATIO (p. ej. 0.25) y LESCO_MIN_DETECTED_FRAMES.
        self.min_detected_frames = _env_int(
            "LESCO_MIN_DETECTED_FRAMES", DEFAULT_MIN_DETECTED_FRAMES
        )
        self.min_detected_ratio = _env_float(
            "LESCO_MIN_DETECTED_RATIO", DEFAULT_MIN_DETECTED_RATIO
        )

        self.current_result: RecognitionResult | None = None
        self.result_expires = 0.0
        self.frame_counter = 0
        self.last_event: dict | None = None
        self._lock = threading.Lock()

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """Procesa un frame BGR y devuelve el estado serializado para el HUD."""
        with self._lock:
            self.frame_counter += 1
            frame_bgr = self.preprocessor.process(frame_bgr)
            detection = self.detector.detect(frame_bgr)
            features, hand_detected = frame_to_features(detection)

            if self.segmenter.update(features, hand_detected):
                frames, detected = self.segmenter.get_sequence()
                result = classify_sequence(
                    self.model,
                    self.references,
                    frames,
                    detected,
                    DEFAULT_TOP_K,
                    self.confidence_threshold,
                    DEFAULT_MARGIN_THRESHOLD,
                    self.min_detected_frames,
                    self.min_detected_ratio,
                    DEFAULT_BAYES_SMOOTHING,
                    DEFAULT_BAYES_MIN_EVIDENCE,
                )
                self.segmenter.reset()
                self._record_event(result)

                if result.accepted:
                    self.current_result = result
                    self.result_expires = time.monotonic() + DEFAULT_RESULT_HOLD_S
                else:
                    self.current_result = None
                    self.result_expires = 0.0

            if self.current_result is not None and time.monotonic() >= self.result_expires:
                self.current_result = None

            return self._serialize()

    def _record_event(self, result: RecognitionResult) -> None:
        """Registra el resultado de una seña cerrada (en log y para el HUD)."""
        best = result.best_label
        prob = result.best_probability
        if result.accepted:
            LOGGER.info("Predicción aceptada: %s (%.0f%%)", best, prob * 100)
        else:
            candidate = f"{best} {prob * 100:.0f}%" if result.candidates else "sin candidato"
            LOGGER.info(
                "Movimiento rechazado: %s (%s, evidencias=%d)",
                result.reject_reason,
                candidate,
                result.evidence_count,
            )
        self.last_event = {
            "accepted": result.accepted,
            "reason": result.reject_reason,
            "label": best,
            "confidence": round(prob * 100),
            "ts": time.monotonic(),
        }

    def _serialize(self) -> dict:
        now = time.monotonic()
        result_payload = dict(_EMPTY_RESULT)

        if (
            self.current_result is not None
            and self.current_result.accepted
            and now < self.result_expires
        ):
            r = self.current_result
            result_payload = {
                "accepted": True,
                "label": r.best_label,
                "confidence": round(r.best_probability * 100),
                "candidates": [
                    [label, round(float(prob), 4)] for label, prob in r.candidates
                ],
            }

        # Feedback de la última seña cerrada (útil cuando fue rechazada)
        info = None
        if self.last_event is not None and now - self.last_event["ts"] < 3.0:
            info = {
                "accepted": self.last_event["accepted"],
                "reason": self.last_event["reason"],
                "label": self.last_event["label"],
                "confidence": self.last_event["confidence"],
            }

        return {
            "state": self.segmenter.state.name,  # "WAITING" | "SIGNING"
            "velocity": round(self.segmenter.current_velocity, 5),
            "motion_threshold": self.segmenter.motion_threshold,
            "result": result_payload,
            "info": info,
        }


# ── Sesion unica (un solo cliente / camara) ─────────────────────────────────────

_session: RecognizerSession | None = None
_session_error: str | None = None
_session_lock = threading.Lock()


def get_session() -> RecognizerSession | None:
    """Devuelve la sesion del reconocedor, creandola de forma perezosa.

    Si el modelo/dataset no existen, no rompe el servidor: registra el error y
    devuelve ``None`` para que el HUD siga sirviendose y muestre el problema.
    """
    global _session, _session_error
    with _session_lock:
        if _session is not None:
            return _session
        if _session_error is not None:
            return None
        try:
            model_path = Path(os.environ.get("LESCO_MODEL", str(DEFAULT_MODEL_PATH)))
            dataset_path = Path(os.environ.get("LESCO_DATASET", str(DEFAULT_DATASET_PATH)))
            _session = RecognizerSession(model_path, dataset_path)
        except Exception as exc:  # noqa: BLE001 - se reporta al frontend
            _session_error = str(exc)
            LOGGER.error("No se pudo iniciar el reconocedor: %s", exc)
            return None
        return _session


def _decode_and_process(session: RecognizerSession, data: bytes) -> dict:
    buffer = np.frombuffer(data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {
            "state": "WAITING",
            "velocity": 0.0,
            "motion_threshold": 0.0,
            "result": dict(_EMPTY_RESULT),
        }
    return session.process_frame(frame_bgr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    get_session()  # warm-up; si falla, queda registrado y el HUD lo informa
    yield


app = FastAPI(title="LESCO-AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile) -> dict:
    session = get_session()
    if session is None:
        return {
            "state": "WAITING",
            "velocity": 0.0,
            "motion_threshold": 0.0,
            "error": _session_error,
            "result": dict(_EMPTY_RESULT),
        }

    data = await file.read()
    # imdecode + MediaPipe son sincronos y pesados: fuera del event loop.
    return await asyncio.to_thread(_decode_and_process, session, data)


# El frontend se sirve desde el mismo origen (sin CORS; getUserMedia OK en
# localhost). Se monta al final para no ensombrecer la ruta /predict.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
