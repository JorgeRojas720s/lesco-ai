"""
app/ml/bayesian_filter.py
=========================

Funcion
-------
Acumula evidencia probabilistica de varias predicciones de la red neuronal
para obtener una decision mas estable.

Notas
-----
Usa una actualizacion bayesiana simple: la posterior actual se multiplica por
la nueva evidencia y se normaliza. Trabaja en logaritmos para evitar problemas
numericos cuando se combinan varias probabilidades pequenas.
"""

from __future__ import annotations

import numpy as np


class BayesianSignFilter:
    """Filtro bayesiano discreto sobre las etiquetas conocidas por el modelo."""

    def __init__(
        self,
        labels: list[str],
        *,
        smoothing: float = 0.15,
    ) -> None:
        if not labels:
            raise ValueError("BayesianSignFilter necesita al menos una etiqueta.")
        self.labels = labels
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self._log_posterior = np.full(len(labels), -np.log(len(labels)), dtype=np.float64)
        self.update_count = 0

    def reset(self) -> None:
        """Reinicia la posterior a una distribucion uniforme."""
        self._log_posterior[:] = -np.log(len(self.labels))
        self.update_count = 0

    def update(self, probabilities: np.ndarray, weight: float = 1.0) -> None:
        """
        Incorpora una nueva prediccion de la red como evidencia.

        `weight` permite que evidencias parciales tengan menos fuerza que la
        prediccion completa de la seña.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (len(self.labels),):
            raise ValueError(f"Expected {len(self.labels)} probabilities, got {probs.shape}.")

        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()

        uniform = np.full_like(probs, 1.0 / len(probs))
        likelihood = (1.0 - self.smoothing) * probs + self.smoothing * uniform

        self._log_posterior += float(weight) * np.log(likelihood)
        self._normalize()
        self.update_count += 1

    def posterior(self) -> np.ndarray:
        """Devuelve la distribucion posterior actual."""
        return np.exp(self._log_posterior).astype(np.float32)

    def top_k(self, k: int) -> list[tuple[str, float]]:
        """Devuelve las etiquetas mas probables de la posterior."""
        posterior = self.posterior()
        indices = np.argsort(posterior)[::-1][:k]
        return [(self.labels[int(idx)], float(posterior[idx])) for idx in indices]

    def _normalize(self) -> None:
        max_log = float(np.max(self._log_posterior))
        shifted = self._log_posterior - max_log
        log_sum = max_log + float(np.log(np.exp(shifted).sum()))
        self._log_posterior -= log_sum
