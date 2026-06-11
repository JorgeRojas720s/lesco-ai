"""
app/ml/rule_based_translator.py
===============================

Sistema basado en reglas para convertir predicciones estabilizadas en texto.

La red neuronal y el filtro Bayes estiman la seña mas probable. Esta capa no
reemplaza al modelo: decide si la prediccion se acepta y como se agregan
palabras al texto final.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from app.vision.landmark_extractor import FingerState


class RuleAction(str, Enum):
    ACCEPT = "accept"
    IGNORE = "ignore"
    CLOSE_WORD = "close_word"


@dataclass(frozen=True)
class FingerSnapshot:
    """Estado booleano de los dedos usado por reglas simbolicas simples."""

    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False


@dataclass
class RuleDecision:
    """Resultado de evaluar una prediccion contra las reglas."""

    action: RuleAction
    accepted: bool
    label: str = ""
    reason: str | None = None
    text: str = ""


@dataclass
class RuleBasedTranslator:
    """
    Mantiene el texto traducido aplicando reglas sobre predicciones aceptadas.

    Parameters
    ----------
    confidence_threshold:
        Confianza minima para permitir que una seña llegue al texto.
    min_stable_frames:
        Cantidad minima de frames de la toma para considerar estable la seña.
    no_hand_timeout_s:
        Tiempo sin mano para cerrar la palabra actual.
    """

    confidence_threshold: float = 0.70
    min_stable_frames: int = 10
    no_hand_timeout_s: float = 1.0
    text: str = ""
    last_decision: RuleDecision | None = None
    _last_hand_seen_at: float = field(default_factory=time.monotonic)
    _word_open: bool = False

    def reset(self) -> None:
        self.text = ""
        self.last_decision = None
        self._last_hand_seen_at = time.monotonic()
        self._word_open = False

    def update_hand_presence(
        self,
        hand_detected: bool,
        *,
        now: float | None = None,
    ) -> RuleDecision | None:
        """
        Aplica la regla: si la mano desaparece por mas de 1 segundo, cerrar
        la palabra actual.
        """
        now = time.monotonic() if now is None else now
        if hand_detected:
            self._last_hand_seen_at = now
            return None

        if self._word_open and now - self._last_hand_seen_at >= self.no_hand_timeout_s:
            decision = self._append_space(reason="mano ausente")
            self.last_decision = decision
            self._last_hand_seen_at = now
            return decision

        return None

    def apply_prediction(
        self,
        label: str,
        confidence: float,
        *,
        stable_frames: int,
        finger_snapshot: FingerSnapshot | None = None,
    ) -> RuleDecision:
        """Evalua y aplica una prediccion ya estabilizada por el filtro Bayes."""
        label = normalize_label(label)
        confidence = float(confidence)

        if confidence < self.confidence_threshold:
            return self._store(
                RuleDecision(
                    action=RuleAction.IGNORE,
                    accepted=False,
                    label=label,
                    reason=f"confianza menor a {self.confidence_threshold:.0%}",
                    text=self.text,
                )
            )

        if stable_frames < self.min_stable_frames:
            return self._store(
                RuleDecision(
                    action=RuleAction.IGNORE,
                    accepted=False,
                    label=label,
                    reason=f"seña estable por solo {stable_frames} frames",
                    text=self.text,
                )
            )

        corrected_label = self._correct_with_fingers(label, finger_snapshot)

        self.text = append_token(self.text, corrected_label)
        self._word_open = True
        return self._store(
            RuleDecision(
                action=RuleAction.ACCEPT,
                accepted=True,
                label=corrected_label,
                text=self.text,
            )
        )

    def _append_space(self, *, reason: str) -> RuleDecision:
        self.text = self.text.rstrip()
        if self.text:
            self.text += " "
        self._word_open = False
        return RuleDecision(
            action=RuleAction.CLOSE_WORD,
            accepted=True,
            label="",
            reason=reason,
            text=self.text,
        )

    def _correct_with_fingers(
        self,
        label: str,
        finger_snapshot: FingerSnapshot | None,
    ) -> str:
        if finger_snapshot is None:
            return label

        rule_label = infer_label_from_fingers(finger_snapshot)
        if rule_label is None:
            return label

        # Solo corrige cuando la regla simbolica coincide con ejemplos claros.
        if rule_label == "BIEN" and label in {"OK", "PULGAR", "LIKE"}:
            return "BIEN"
        if rule_label == "ALTO" and label in {"PARE", "STOP"}:
            return "ALTO"
        return label

    def _store(self, decision: RuleDecision) -> RuleDecision:
        self.last_decision = decision
        return decision


def normalize_label(label: str) -> str:
    """Normaliza etiquetas del modelo para compararlas con reglas."""
    return label.strip().upper().replace(" ", "_")


def append_token(text: str, token: str) -> str:
    """Agrega una seña como unidad textual evitando espacios duplicados."""
    clean_text = text.rstrip()
    if not clean_text:
        return token
    return f"{clean_text} {token}"


def snapshot_from_fingers(fingers: list[FingerState]) -> FingerSnapshot:
    """Convierte la salida de LandmarkExtractor a un snapshot compacto."""
    by_name = {finger.name.upper(): finger.is_extended for finger in fingers}
    return FingerSnapshot(
        thumb=by_name.get("THUMB", False),
        index=by_name.get("INDEX", False),
        middle=by_name.get("MIDDLE", False),
        ring=by_name.get("RING", False),
        pinky=by_name.get("PINKY", False),
    )


def infer_label_from_fingers(snapshot: FingerSnapshot) -> str | None:
    """
    Reglas simbolicas simples basadas en dedos extendidos.

    Estas reglas son auxiliares: dan pistas para validar/corregir etiquetas
    claras, no sustituyen al clasificador entrenado.
    """
    other_fingers_folded = not any(
        [snapshot.index, snapshot.middle, snapshot.ring, snapshot.pinky]
    )
    if snapshot.thumb and other_fingers_folded:
        return "BIEN"

    if all([snapshot.thumb, snapshot.index, snapshot.middle, snapshot.ring, snapshot.pinky]):
        return "ALTO"

    return None
