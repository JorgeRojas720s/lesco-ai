from app.ml.rule_based_translator import (
    FingerSnapshot,
    RuleAction,
    RuleBasedTranslator,
    infer_label_from_fingers,
)


def test_ignores_low_confidence_prediction() -> None:
    translator = RuleBasedTranslator(confidence_threshold=0.70, min_stable_frames=10)

    decision = translator.apply_prediction("HOLA", 0.69, stable_frames=20)

    assert decision.action == RuleAction.IGNORE
    assert translator.text == ""


def test_ignores_unstable_prediction() -> None:
    translator = RuleBasedTranslator(confidence_threshold=0.70, min_stable_frames=10)

    decision = translator.apply_prediction("HOLA", 0.95, stable_frames=9)

    assert decision.action == RuleAction.IGNORE
    assert "9 frames" in (decision.reason or "")


def test_accepts_signs_as_text_tokens() -> None:
    translator = RuleBasedTranslator(confidence_threshold=0.70, min_stable_frames=10)

    translator.apply_prediction("HOLA", 0.95, stable_frames=15)
    decision = translator.apply_prediction("GRACIAS", 0.90, stable_frames=15)

    assert decision.action == RuleAction.ACCEPT
    assert translator.text == "HOLA GRACIAS"


def test_closes_word_when_hand_disappears() -> None:
    translator = RuleBasedTranslator(no_hand_timeout_s=1.0)
    translator.apply_prediction("HOLA", 0.95, stable_frames=15)
    translator.update_hand_presence(True, now=10.0)

    decision = translator.update_hand_presence(False, now=11.1)

    assert decision is not None
    assert decision.action == RuleAction.CLOSE_WORD
    assert translator.text == "HOLA "


def test_finger_rules_infer_clear_labels() -> None:
    assert infer_label_from_fingers(
        FingerSnapshot(thumb=True, index=False, middle=False, ring=False, pinky=False)
    ) == "BIEN"
    assert infer_label_from_fingers(
        FingerSnapshot(thumb=True, index=True, middle=True, ring=True, pinky=True)
    ) == "ALTO"
