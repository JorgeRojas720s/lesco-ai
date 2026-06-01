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
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.cli.collect_data import (
    FEATURES_PER_FRAME,
    frame_to_features,
    smart_resample,
)
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
from app.ml.neural_sign_classifier import train_classifier
from app.storage.dataset import LESCODataset
from app.vision.hand_detector import HandDetector
from app.vision.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_EMPTY_RESULT = {"accepted": False, "label": "", "confidence": 0, "candidates": []}

SEQUENCE_LENGTH = 60       # frames por muestra tras remuestrear (igual que el CLI)
MIN_SAMPLE_FRAMES = 12     # mínimo de frames crudos para aceptar una toma


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _model_path() -> Path:
    return Path(os.environ.get("LESCO_MODEL", str(DEFAULT_MODEL_PATH)))


def _dataset_path() -> Path:
    return Path(os.environ.get("LESCO_DATASET", str(DEFAULT_DATASET_PATH)))


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
        # Validación de distancia contra el dataset. El dataset se grabó con el
        # CLI; el navegador manda frames recomprimidos, así que las distancias
        # quedan infladas. Subí LESCO_MIN_CLASS_DISTANCE_THRESHOLD (p. ej. 4.0)
        # para que una seña correcta no se rechace por "fuera de la clase".
        self.references = load_class_references(
            dataset_path,
            self.model,
            _env_float("LESCO_DISTANCE_SCALE", DEFAULT_DISTANCE_SCALE),
            _env_float(
                "LESCO_MIN_CLASS_DISTANCE_THRESHOLD",
                DEFAULT_MIN_CLASS_DISTANCE_THRESHOLD,
            ),
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
            _session = RecognizerSession(_model_path(), _dataset_path())
        except Exception as exc:  # noqa: BLE001 - se reporta al frontend
            _session_error = str(exc)
            LOGGER.error("No se pudo iniciar el reconocedor: %s", exc)
            return None
        return _session


def reset_session() -> None:
    """Fuerza recrear el reconocedor (p. ej. tras entrenar un modelo nuevo).

    El próximo /predict reconstruye el modelo y las referencias del dataset.
    """
    global _session, _session_error
    with _session_lock:
        _session = None
        _session_error = None


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


# ─────────────────────────────────────────────────────────────────────────────
# GRABACIÓN DE MUESTRAS (dataset web)
#
# Usa el MISMO preprocesamiento + detección + frame_to_features que /predict, y
# guarda con smart_resample + LESCODataset.append (idéntico al CLI collect_data).
# Así el dataset queda grabado por el pipeline web y coincide con lo que ve el
# reconocedor → las distancias dejan de estar infladas.
# ─────────────────────────────────────────────────────────────────────────────


class CollectorSession:
    """Acumula los frames de una toma y graba una muestra al dataset."""

    def __init__(self) -> None:
        self.preprocessor = Preprocessor(flip_horizontal=True)
        self.detector = HandDetector(max_hands=2)
        self.detector.start()
        self.frames: list[np.ndarray] = []
        self.detected: list[bool] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.frames = []
            self.detected = []

    def add_frame(self, frame_bgr: np.ndarray) -> dict:
        with self._lock:
            frame_bgr = self.preprocessor.process(frame_bgr)
            detection = self.detector.detect(frame_bgr)
            features, hand_detected = frame_to_features(detection)
            self.frames.append(features)
            self.detected.append(hand_detected)
            return {
                "frames": len(self.frames),
                "detected": int(sum(self.detected)),
                "hand_detected": bool(hand_detected),
            }

    def save(self, label: str) -> dict:
        with self._lock:
            n_frames = len(self.frames)
            n_detected = int(sum(self.detected))
            if n_frames < MIN_SAMPLE_FRAMES:
                return {"ok": False, "reason": f"toma muy corta ({n_frames} frames, mínimo {MIN_SAMPLE_FRAMES})"}
            if n_detected < MIN_SAMPLE_FRAMES // 2:
                return {"ok": False, "reason": f"poca mano detectada ({n_detected}/{n_frames} frames)"}
            sample = smart_resample(self.frames, self.detected, SEQUENCE_LENGTH)
            self.frames = []
            self.detected = []

        clean = label.upper().strip().replace(" ", "_")
        with LESCODataset(_dataset_path(), SEQUENCE_LENGTH, FEATURES_PER_FRAME) as dataset:
            total = dataset.append(clean, sample, n_frames)
            info = dataset.info()
        LOGGER.info("Muestra guardada: %s (raw=%d, total=%d)", clean, n_frames, total)
        return {"ok": True, "label": clean, "total": total, "per_label": info["per_label"]}


_collector: CollectorSession | None = None
_collector_lock = threading.Lock()


def get_collector() -> CollectorSession:
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = CollectorSession()
        return _collector


def _dataset_info() -> dict:
    path = _dataset_path()
    if not path.exists():
        return {"path": str(path), "total_samples": 0, "per_label": {}, "exists": False}
    with LESCODataset(path, SEQUENCE_LENGTH, FEATURES_PER_FRAME) as dataset:
        info = dataset.info()
    info["exists"] = True
    return info


@app.get("/api/dataset")
async def dataset_info() -> dict:
    return await asyncio.to_thread(_dataset_info)


@app.post("/collect/start")
async def collect_start() -> dict:
    get_collector().reset()
    return {"ok": True}


@app.post("/collect/frame")
async def collect_frame(file: UploadFile) -> dict:
    data = await file.read()

    def _work() -> dict:
        buffer = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {"frames": 0, "detected": 0, "hand_detected": False}
        return get_collector().add_frame(frame_bgr)

    return await asyncio.to_thread(_work)


@app.post("/collect/save")
async def collect_save(label: str = Form(...)) -> dict:
    return await asyncio.to_thread(get_collector().save, label)


@app.post("/collect/discard")
async def collect_discard() -> dict:
    get_collector().reset()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO (reutiliza train_classifier; recarga el modelo en caliente)
# ─────────────────────────────────────────────────────────────────────────────

_train_state: dict = {"status": "idle", "message": "", "metrics": None}
_train_lock = threading.Lock()


def _run_training() -> None:
    try:
        with LESCODataset(_dataset_path(), SEQUENCE_LENGTH, FEATURES_PER_FRAME) as dataset:
            X, y = dataset.load_all()
        if X.shape[0] == 0:
            raise ValueError("El dataset está vacío. Grabá muestras primero.")

        model, history, split = train_classifier(X, y)
        model.save(_model_path())

        has_val = len(split["val_idx"]) > 0
        metrics = {
            "labels": list(model.labels),
            "train_samples": int(len(split["train_idx"])),
            "val_samples": int(len(split["val_idx"])),
            "train_acc": round(float(history.train_accuracy[-1]), 4),
            "val_acc": round(float(history.val_accuracy[-1]), 4) if has_val else None,
        }
        reset_session()  # el próximo /predict usa el modelo + dataset nuevos
        with _train_lock:
            _train_state.update(
                status="done",
                message="Entrenamiento completo. Modelo recargado en caliente.",
                metrics=metrics,
            )
        LOGGER.info("Entrenamiento completo: %s", metrics)
    except Exception as exc:  # noqa: BLE001 - se reporta al frontend
        with _train_lock:
            _train_state.update(status="error", message=str(exc), metrics=None)
        LOGGER.error("Error entrenando: %s", exc)


@app.post("/train")
async def train() -> dict:
    with _train_lock:
        if _train_state["status"] == "running":
            return {"status": "running", "message": "Ya hay un entrenamiento en curso."}
        _train_state.update(status="running", message="Entrenando…", metrics=None)
    threading.Thread(target=_run_training, daemon=True).start()
    return {"status": "running", "message": "Entrenamiento iniciado."}


@app.get("/train/status")
async def train_status() -> dict:
    with _train_lock:
        return dict(_train_state)


# El frontend se sirve desde el mismo origen (sin CORS; getUserMedia OK en
# localhost). Se monta al final para no ensombrecer las rutas de la API.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
