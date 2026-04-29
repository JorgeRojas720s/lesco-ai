"""
app/ml/neural_sign_classifier.py
================================

Funcion
-------
Define y entrena una red neuronal simple para clasificar secuencias de senas
a partir de landmarks guardados en el dataset.

Notas
-----
El modelo aplana cada secuencia completa, normaliza entradas y devuelve
probabilidades por etiqueta para poder combinarlas luego con otros metodos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass
class TrainingHistory:
    """Metrics collected while training."""

    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]


@dataclass
class NeuralSignClassifier:
    """
    Feed-forward neural network with softmax output.

    The output probabilities are intentionally exposed because a later
    uncertainty layer can combine them with Bayesian classification and rules.
    """

    labels: list[str]
    sequence_length: int
    features_per_frame: int
    input_mean: Array
    input_std: Array
    weights: list[Array]
    biases: list[Array]

    @property
    def input_size(self) -> int:
        return self.sequence_length * self.features_per_frame

    def predict_proba(self, sequences: Array) -> Array:
        """
        Return class probabilities for one or more sign sequences.

        Parameters
        ----------
        sequences:
            Shape (T, F) for one sample or (N, T, F) for a batch.
        """
        X = _prepare_sequences(
            sequences,
            sequence_length=self.sequence_length,
            features_per_frame=self.features_per_frame,
        )
        X = (X - self.input_mean) / self.input_std
        return _forward(X, self.weights, self.biases)[-1]

    def predict(self, sequences: Array) -> list[str]:
        """Return the most likely label for each sequence."""
        probabilities = self.predict_proba(sequences)
        return [self.labels[int(i)] for i in probabilities.argmax(axis=1)]

    def save(self, path: Path | str) -> None:
        """Persist model parameters in a compressed NumPy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Array] = {
            "labels": np.array(self.labels, dtype=object),
            "sequence_length": np.array(self.sequence_length, dtype=np.int32),
            "features_per_frame": np.array(self.features_per_frame, dtype=np.int32),
            "input_mean": self.input_mean.astype(np.float32),
            "input_std": self.input_std.astype(np.float32),
            "num_layers": np.array(len(self.weights), dtype=np.int32),
        }
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{idx}"] = weight.astype(np.float32)
            payload[f"b{idx}"] = bias.astype(np.float32)

        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path | str) -> "NeuralSignClassifier":
        """Load a model saved with `save`."""
        with np.load(Path(path), allow_pickle=True) as data:
            num_layers = int(data["num_layers"])
            weights = [data[f"W{i}"].astype(np.float32) for i in range(num_layers)]
            biases = [data[f"b{i}"].astype(np.float32) for i in range(num_layers)]
            labels = [str(label) for label in data["labels"].tolist()]

            return cls(
                labels=labels,
                sequence_length=int(data["sequence_length"]),
                features_per_frame=int(data["features_per_frame"]),
                input_mean=data["input_mean"].astype(np.float32),
                input_std=data["input_std"].astype(np.float32),
                weights=weights,
                biases=biases,
            )


def train_classifier(
    X: Array,
    y: Array,
    *,
    hidden_sizes: tuple[int, ...] = (128, 64),
    epochs: int = 800,
    learning_rate: float = 0.001,
    l2: float = 0.0001,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[NeuralSignClassifier, TrainingHistory, dict[str, Array]]:
    """
    Train a neural sign classifier from HDF5 arrays.

    Returns the model, metric history, and the split indices used for training.
    If there are not enough samples per class, validation is skipped and all
    samples are used for training.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=str)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X.shape[0] == 0:
        raise ValueError("Cannot train with an empty dataset.")

    labels = sorted(np.unique(y).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_idx = np.array([label_to_idx[label] for label in y], dtype=np.int64)

    train_idx, val_idx = _make_stratified_split(y_idx, validation_split, seed)

    N, sequence_length, features_per_frame = X.shape
    X_flat = X.reshape(N, sequence_length * features_per_frame)

    mean = X_flat[train_idx].mean(axis=0).astype(np.float32)
    std = X_flat[train_idx].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    X_scaled = ((X_flat - mean) / std).astype(np.float32)
    y_one_hot = np.eye(len(labels), dtype=np.float32)[y_idx]

    rng = np.random.default_rng(seed)
    layer_sizes = (X_scaled.shape[1], *hidden_sizes, len(labels))
    weights, biases = _init_parameters(layer_sizes, rng)

    history = TrainingHistory([], [], [], [])
    adam = _AdamState.from_parameters(weights, biases)

    for epoch in range(1, epochs + 1):
        activations = _forward(X_scaled[train_idx], weights, biases)
        grads_w, grads_b = _backward(
            activations,
            y_one_hot[train_idx],
            weights,
            l2=l2,
        )
        _adam_step(weights, biases, grads_w, grads_b, adam, learning_rate, epoch)

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 20) == 0:
            train_proba = _forward(X_scaled[train_idx], weights, biases)[-1]
            history.train_loss.append(_cross_entropy(train_proba, y_one_hot[train_idx], weights, l2))
            history.train_accuracy.append(_accuracy(train_proba, y_idx[train_idx]))

            if len(val_idx):
                val_proba = _forward(X_scaled[val_idx], weights, biases)[-1]
                history.val_loss.append(_cross_entropy(val_proba, y_one_hot[val_idx], weights, l2))
                history.val_accuracy.append(_accuracy(val_proba, y_idx[val_idx]))

    model = NeuralSignClassifier(
        labels=labels,
        sequence_length=sequence_length,
        features_per_frame=features_per_frame,
        input_mean=mean,
        input_std=std,
        weights=weights,
        biases=biases,
    )
    split = {"train_idx": train_idx, "val_idx": val_idx}
    return model, history, split


def _prepare_sequences(
    sequences: Array,
    *,
    sequence_length: int,
    features_per_frame: int,
) -> Array:
    X = np.asarray(sequences, dtype=np.float32)
    if X.ndim == 2:
        X = X[None, :, :]
    expected = (sequence_length, features_per_frame)
    if X.ndim != 3 or X.shape[1:] != expected:
        raise ValueError(f"Expected shape (N, {expected[0]}, {expected[1]}), got {X.shape}.")
    return X.reshape(X.shape[0], sequence_length * features_per_frame)


def _init_parameters(layer_sizes: tuple[int, ...], rng: np.random.Generator) -> tuple[list[Array], list[Array]]:
    weights: list[Array] = []
    biases: list[Array] = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = np.sqrt(2.0 / max(fan_in, 1))
        weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32))
        biases.append(np.zeros(fan_out, dtype=np.float32))
    return weights, biases


def _forward(X: Array, weights: list[Array], biases: list[Array]) -> list[Array]:
    activations = [X]
    current = X
    for weight, bias in zip(weights[:-1], biases[:-1]):
        current = np.maximum(current @ weight + bias, 0.0)
        activations.append(current)
    logits = current @ weights[-1] + biases[-1]
    activations.append(_softmax(logits))
    return activations


def _backward(
    activations: list[Array],
    y_one_hot: Array,
    weights: list[Array],
    *,
    l2: float,
) -> tuple[list[Array], list[Array]]:
    m = max(y_one_hot.shape[0], 1)
    delta = (activations[-1] - y_one_hot) / m

    grads_w: list[Array] = [np.empty_like(w) for w in weights]
    grads_b: list[Array] = [np.empty(w.shape[1], dtype=np.float32) for w in weights]

    for layer in range(len(weights) - 1, -1, -1):
        prev_activation = activations[layer]
        grads_w[layer] = (prev_activation.T @ delta + l2 * weights[layer]).astype(np.float32)
        grads_b[layer] = delta.sum(axis=0).astype(np.float32)

        if layer > 0:
            delta = delta @ weights[layer].T
            delta = delta * (activations[layer] > 0.0)

    return grads_w, grads_b


def _softmax(logits: Array) -> Array:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _cross_entropy(proba: Array, y_one_hot: Array, weights: list[Array], l2: float) -> float:
    clipped = np.clip(proba, 1e-8, 1.0)
    data_loss = -np.sum(y_one_hot * np.log(clipped)) / max(y_one_hot.shape[0], 1)
    reg_loss = 0.5 * l2 * sum(float(np.sum(w * w)) for w in weights)
    return float(data_loss + reg_loss)


def _accuracy(proba: Array, y_idx: Array) -> float:
    return float(np.mean(proba.argmax(axis=1) == y_idx))


def _make_stratified_split(
    y_idx: Array,
    validation_split: float,
    seed: int,
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    validation_split = float(np.clip(validation_split, 0.0, 0.8))
    train_indices: list[int] = []
    val_indices: list[int] = []

    if validation_split <= 0.0:
        return np.arange(len(y_idx), dtype=np.int64), np.array([], dtype=np.int64)

    labels, counts = np.unique(y_idx, return_counts=True)
    if np.any(counts < 2):
        return np.arange(len(y_idx), dtype=np.int64), np.array([], dtype=np.int64)

    for label in labels:
        indices = np.where(y_idx == label)[0]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * validation_split)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


@dataclass
class _AdamState:
    mw: list[Array]
    vw: list[Array]
    mb: list[Array]
    vb: list[Array]

    @classmethod
    def from_parameters(cls, weights: list[Array], biases: list[Array]) -> "_AdamState":
        return cls(
            mw=[np.zeros_like(w) for w in weights],
            vw=[np.zeros_like(w) for w in weights],
            mb=[np.zeros_like(b) for b in biases],
            vb=[np.zeros_like(b) for b in biases],
        )


def _adam_step(
    weights: list[Array],
    biases: list[Array],
    grads_w: list[Array],
    grads_b: list[Array],
    state: _AdamState,
    learning_rate: float,
    step: int,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for idx in range(len(weights)):
        state.mw[idx] = beta1 * state.mw[idx] + (1.0 - beta1) * grads_w[idx]
        state.vw[idx] = beta2 * state.vw[idx] + (1.0 - beta2) * (grads_w[idx] * grads_w[idx])
        state.mb[idx] = beta1 * state.mb[idx] + (1.0 - beta1) * grads_b[idx]
        state.vb[idx] = beta2 * state.vb[idx] + (1.0 - beta2) * (grads_b[idx] * grads_b[idx])

        mw_hat = state.mw[idx] / (1.0 - beta1**step)
        vw_hat = state.vw[idx] / (1.0 - beta2**step)
        mb_hat = state.mb[idx] / (1.0 - beta1**step)
        vb_hat = state.vb[idx] / (1.0 - beta2**step)

        weights[idx] -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
        biases[idx] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
