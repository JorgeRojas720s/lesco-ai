"""
app/storage/dataset.py
======================

Funcion
-------
Administra el dataset HDF5 donde se guardan las secuencias de landmarks,
sus etiquetas y la duracion original de cada toma.

Notas
-----
Crea y valida el esquema del archivo. Los datasets principales son `/X`,
`/labels` y `/original_lengths`; `append()` agrega muestras sin reescribir
todo el archivo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

LOGGER = logging.getLogger(__name__)

_DATASET_VERSION = "2.0"
_CHUNK_SIZE = 32   # samples per HDF5 chunk – tune for your cache size


class LESCODataset:
    """
    HDF5-backed store for LESCO landmark sequences.

    The file is kept open while this object is alive; call close() (or use
    it as a context manager) when you are done.

    Parameters
    ----------
    path:
        Path to the .h5 file (created if it does not exist).
    sequence_length:
        Number of frames per sample (e.g. 60).
    features_per_frame:
        Width of each frame vector (e.g. 126 = 2 hands × 63).
    """

    def __init__(
        self,
        path: Path | str,
        sequence_length: int,
        features_per_frame: int,
    ) -> None:
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.features_per_frame = features_per_frame
        self._file: h5py.File | None = None
        self._open()

    # ── Public API ─────────────────────────────────────────────────────────────

    def append(
        self,
        label: str,
        sequence: np.ndarray,
        original_length: int,
    ) -> int:
        """
        Append one sample to the dataset.

        Parameters
        ----------
        label:
            Class label string (e.g. "HOLA").
        sequence:
            Float32 array of shape (sequence_length, features_per_frame).
        original_length:
            Number of raw frames captured before resampling.

        Returns
        -------
        Total number of samples in the dataset after the append.
        """
        if self._file is None:
            raise RuntimeError("Dataset is closed.")

        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape != (self.sequence_length, self.features_per_frame):
            raise ValueError(
                f"Expected shape ({self.sequence_length}, {self.features_per_frame}), "
                f"got {sequence.shape}."
            )

        n = self._file["X"].shape[0]

        # Resize each resizable dataset by one row.
        self._file["X"].resize(n + 1, axis=0)
        self._file["labels"].resize(n + 1, axis=0)
        self._file["original_lengths"].resize(n + 1, axis=0)

        self._file["X"][n] = sequence
        self._file["labels"][n] = label.encode("utf-8")
        self._file["original_lengths"][n] = original_length

        self._file.flush()   # fsync-level safety on every write
        return n + 1

    def info(self) -> dict:
        """Return a summary dictionary about the dataset (no large data loaded)."""
        if self._file is None:
            raise RuntimeError("Dataset is closed.")

        n = self._file["X"].shape[0]
        labels_raw = self._file["labels"][:] if n > 0 else []
        labels = [lb.decode("utf-8") if isinstance(lb, bytes) else lb for lb in labels_raw]
        unique, counts = np.unique(labels, return_counts=True) if labels else ([], [])

        return {
            "path": str(self.path),
            "total_samples": n,
            "sequence_length": self.sequence_length,
            "features_per_frame": self.features_per_frame,
            "shape_X": (n, self.sequence_length, self.features_per_frame),
            "per_label": dict(zip(unique.tolist(), counts.tolist())) if len(unique) else {},
            "version": self._file.attrs.get("version", "unknown"),
            "created_at": self._file.attrs.get("created_at", "unknown"),
        }

    def load_all(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the full dataset into memory as numpy arrays.

        Returns
        -------
        X : float32 array of shape (N, sequence_length, features_per_frame)
        y : str array of shape (N,)
        """
        if self._file is None:
            raise RuntimeError("Dataset is closed.")

        X = self._file["X"][:]
        raw_labels = self._file["labels"][:]
        y = np.array(
            [lb.decode("utf-8") if isinstance(lb, bytes) else lb for lb in raw_labels],
            dtype=object,
        )
        return X, y

    def close(self) -> None:
        """Flush and close the underlying HDF5 file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            LOGGER.debug("HDF5 dataset closed: %s", self.path)

    # ── Context manager support ────────────────────────────────────────────────

    def __enter__(self) -> "LESCODataset":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _open(self) -> None:
        """Open (or create) the HDF5 file and initialise its schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "a", libver="latest")

        # Treat the file as new if the primary dataset does not exist yet.
        if "X" not in self._file:
            self._create_schema()
            LOGGER.info("Nuevo dataset HDF5 creado: %s", self.path)
        else:
            self._validate_schema()
            n = self._file["X"].shape[0]
            LOGGER.info("Dataset HDF5 abierto: %s  (%d muestras)", self.path, n)

    def _create_schema(self) -> None:
        """Initialise empty resizable datasets and file-level metadata."""
        f = self._file

        # Primary data: resizable on axis 0, chunked, gzip-compressed.
        f.create_dataset(
            "X",
            shape=(0, self.sequence_length, self.features_per_frame),
            maxshape=(None, self.sequence_length, self.features_per_frame),
            dtype="float32",
            chunks=(_CHUNK_SIZE, self.sequence_length, self.features_per_frame),
            compression="gzip",
            compression_opts=4,
        )

        # Variable-length UTF-8 strings for labels.
        str_dtype = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(_CHUNK_SIZE,),
        )

        f.create_dataset(
            "original_lengths",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=(_CHUNK_SIZE,),
        )

        # File-level metadata as HDF5 attributes.
        f.attrs["version"] = _DATASET_VERSION
        f.attrs["sequence_length"] = self.sequence_length
        f.attrs["features_per_frame"] = self.features_per_frame
        f.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        f.attrs["hand_slot_right"] = 0    # indices 0–62
        f.attrs["hand_slot_left"] = 1     # indices 63–125
        f.attrs["landmarks_per_hand"] = 21
        f.attrs["coords_per_landmark"] = 3

    def _validate_schema(self) -> None:
        """Warn if the opened file has mismatched parameters."""
        f = self._file
        saved_sl = int(f.attrs.get("sequence_length", -1))
        saved_fp = int(f.attrs.get("features_per_frame", -1))

        if saved_sl != self.sequence_length or saved_fp != self.features_per_frame:
            LOGGER.warning(
                "Parámetros del dataset no coinciden: "
                "archivo=(%d, %d)  código=(%d, %d). "
                "Considera usar un archivo separado.",
                saved_sl, saved_fp, self.sequence_length, self.features_per_frame,
            )
