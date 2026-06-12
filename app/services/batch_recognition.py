"""Batch recognition service for web-captured sign sequences."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from app.cli.collect_data import frame_to_features
from app.main import (
    DEFAULT_BAYES_MIN_EVIDENCE,
    DEFAULT_BAYES_SMOOTHING,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MIN_DETECTED_FRAMES,
    DEFAULT_MIN_DETECTED_RATIO,
    DEFAULT_RESULT_HOLD_S,
    DEFAULT_RULE_CONFIDENCE_THRESHOLD,
    DEFAULT_RULE_MIN_STABLE_FRAMES,
    DEFAULT_RULE_NO_HAND_TIMEOUT_S,
    DEFAULT_TOP_K,
    RecognitionResult,
    classify_sequence,
    load_class_references,
    load_model,
)
from app.ml.rule_based_translator import RuleBasedTranslator, RuleDecision
from app.vision.hand_detector import HandDetector
from app.vision.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)

EMPTY_RESULT = {"accepted": False, "label": "", "confidence": 0, "candidates": []}


class BatchRecognitionService:
    """Classifies complete sign batches captured by the browser.

    The frontend is responsible for collecting the sequence without waiting for
    the backend. This service receives the encoded frames, extracts landmarks in
    order, and runs the same classifier/rule stack used by the native pipeline.
    """

    def __init__(
        self,
        model_path: Path,
        dataset_path: Path,
        *,
        distance_scale: float,
        min_class_distance_threshold: float,
        detection_confidence: float,
        tracking_confidence: float,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_detected_frames: int = DEFAULT_MIN_DETECTED_FRAMES,
        min_detected_ratio: float = DEFAULT_MIN_DETECTED_RATIO,
        rule_confidence_threshold: float = DEFAULT_RULE_CONFIDENCE_THRESHOLD,
        rule_min_stable_frames: int = DEFAULT_RULE_MIN_STABLE_FRAMES,
        rule_no_hand_timeout_s: float = DEFAULT_RULE_NO_HAND_TIMEOUT_S,
    ) -> None:
        self.model = load_model(model_path)
        LOGGER.info("Modelo batch cargado: %s", model_path)
        self.references = load_class_references(
            dataset_path,
            self.model,
            distance_scale,
            min_class_distance_threshold,
        )
        self.preprocessor = Preprocessor(flip_horizontal=True)
        self.detector = HandDetector(
            max_hands=2,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
        )
        self.detector.start()
        self.translator = RuleBasedTranslator(
            confidence_threshold=rule_confidence_threshold,
            min_stable_frames=rule_min_stable_frames,
            no_hand_timeout_s=rule_no_hand_timeout_s,
        )
        self.confidence_threshold = confidence_threshold
        self.min_detected_frames = min_detected_frames
        self.min_detected_ratio = min_detected_ratio
        self.last_event: dict | None = None
        self._lock = threading.Lock()

    def process_encoded_frames(self, encoded_frames: list[bytes]) -> dict:
        """Decode and classify a complete sign sequence."""
        with self._lock:
            features: list[np.ndarray] = []
            detected: list[bool] = []
            decoded_count = 0

            for data in encoded_frames:
                frame_bgr = self._decode_frame(data)
                if frame_bgr is None:
                    continue
                decoded_count += 1
                frame_bgr = self.preprocessor.process(frame_bgr)
                detection = self.detector.detect(frame_bgr)
                frame_features, hand_detected = frame_to_features(detection)
                self.translator.update_hand_presence(hand_detected)
                features.append(frame_features)
                detected.append(hand_detected)

            if not features:
                result = RecognitionResult(
                    candidates=[],
                    accepted=False,
                    reject_reason="lote sin frames validos",
                )
            else:
                result = classify_sequence(
                    self.model,
                    self.references,
                    features,
                    detected,
                    DEFAULT_TOP_K,
                    self.confidence_threshold,
                    DEFAULT_MARGIN_THRESHOLD,
                    self.min_detected_frames,
                    self.min_detected_ratio,
                    DEFAULT_BAYES_SMOOTHING,
                    DEFAULT_BAYES_MIN_EVIDENCE,
                )

            rule_decision = None
            if result.accepted:
                rule_decision = self.translator.apply_prediction(
                    result.best_label,
                    result.best_probability,
                    stable_frames=len(features),
                )

            self._record_event(result, rule_decision, decoded_count)
            return self._serialize(result, rule_decision, decoded_count, len(encoded_frames))

    @staticmethod
    def _decode_frame(data: bytes) -> cv2.typing.MatLike | None:
        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _record_event(
        self,
        result: RecognitionResult,
        rule_decision: RuleDecision | None,
        decoded_count: int,
    ) -> None:
        best = result.best_label
        prob = result.best_probability
        final_accepted = (
            result.accepted
            and rule_decision is not None
            and rule_decision.accepted
        )
        reject_reason = result.reject_reason
        if result.accepted and rule_decision is not None and not rule_decision.accepted:
            reject_reason = rule_decision.reason

        if final_accepted:
            LOGGER.info(
                "Batch aceptado: %s (%.0f%%, frames=%d, texto=%r)",
                best,
                prob * 100,
                decoded_count,
                self.translator.text,
            )
        else:
            candidate = f"{best} {prob * 100:.0f}%" if result.candidates else "sin candidato"
            LOGGER.info(
                "Batch rechazado: %s (%s, frames=%d, evidencias=%d)",
                reject_reason,
                candidate,
                decoded_count,
                result.evidence_count,
            )

        self.last_event = {
            "accepted": final_accepted,
            "reason": reject_reason,
            "label": best,
            "confidence": round(prob * 100),
            "candidates": [
                [label, round(float(candidate_prob), 4)]
                for label, candidate_prob in result.candidates
            ],
            "rule_action": rule_decision.action.value if rule_decision is not None else None,
            "rule_reason": rule_decision.reason if rule_decision is not None else None,
        }

    def _serialize(
        self,
        result: RecognitionResult,
        rule_decision: RuleDecision | None,
        decoded_count: int,
        received_count: int,
    ) -> dict:
        final_accepted = (
            result.accepted
            and rule_decision is not None
            and rule_decision.accepted
        )
        result_payload = dict(EMPTY_RESULT)
        if final_accepted:
            result_payload = {
                "accepted": True,
                "label": result.best_label,
                "confidence": round(result.best_probability * 100),
                "candidates": [
                    [label, round(float(prob), 4)]
                    for label, prob in result.candidates
                ],
            }

        return {
            "state": "WAITING",
            "velocity": 0.0,
            "motion_threshold": 0.0,
            "result": result_payload,
            "translation": {
                "text": self.translator.text,
                "action": (
                    self.translator.last_decision.action.value
                    if self.translator.last_decision is not None
                    else None
                ),
                "label": (
                    self.translator.last_decision.label
                    if self.translator.last_decision is not None
                    else ""
                ),
                "reason": (
                    self.translator.last_decision.reason
                    if self.translator.last_decision is not None
                    else None
                ),
            },
            "info": None if final_accepted else dict(self.last_event or {}),
            "batch": {
                "received_frames": received_count,
                "decoded_frames": decoded_count,
                "hold_seconds": DEFAULT_RESULT_HOLD_S,
            },
        }
