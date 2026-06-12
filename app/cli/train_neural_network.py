"""

Funcion
-------
Entrena una red neuronal usando las secuencias guardadas en el dataset HDF5 y
guarda el modelo entrenado en un archivo .npz.

Comandos
--------
uv run python -m app.cli.train_neural_network --dataset data/signs_dataset.h5
    Entrena con el dataset principal y guarda el modelo en models/.

uv run python -m app.cli.train_neural_network --dataset data/signs_dataset.h5 --epochs 300
    Cambia la cantidad de epocas de entrenamiento.

uv run python -m app.cli.train_neural_network --dataset data/signs_dataset.h5 --output models/prueba.npz
    Guarda el modelo entrenado en una ruta diferente.

Notas
-----
El recolector ya remuestrea cada toma a una forma fija, normalmente (60, 126).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from app.ml.neural_sign_classifier import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una red neuronal para clasificar senas LESCO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/signs_dataset.h5"),
        help="Archivo HDF5 generado por app.cli.collect_data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/neural_sign_classifier.npz"),
        help="Ruta donde se guardara el modelo entrenado.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Neuronas por capa oculta.",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_h5_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    with h5py.File(path, "r") as f:
        X = f["X"][:].astype(np.float32)
        raw_labels = f["labels"][:]
        y = np.array(
            [label.decode("utf-8") if isinstance(label, bytes) else str(label) for label in raw_labels],
            dtype=str,
        )
        original_lengths = f["original_lengths"][:].astype(np.int32)

    return X, y, original_lengths


def print_dataset_report(X: np.ndarray, y: np.ndarray, original_lengths: np.ndarray) -> None:
    print(f"Dataset: X={X.shape}, clases={len(np.unique(y))}, muestras={len(y)}")
    print("Distribucion por etiqueta:")
    for label in sorted(np.unique(y)):
        mask = y == label
        lengths = original_lengths[mask]
        print(
            f"  {label:<20} muestras={int(mask.sum()):>3} "
            f"frames_raw={int(lengths.min())}/{int(np.median(lengths))}/{int(lengths.max())}"
        )

    counts = {label: int(np.sum(y == label)) for label in np.unique(y)}
    weak = [label for label, count in counts.items() if count < 5]
    if weak:
        print()
        print("Aviso: hay clases con pocas muestras.")
        print("Para una evaluacion real conviene tener al menos 20-30 muestras por sena.")
        print("Con menos datos, el entrenamiento sirve como prueba tecnica del flujo.")


def main() -> None:
    args = parse_args()
    X, y, original_lengths = load_h5_dataset(args.dataset)
    print_dataset_report(X, y, original_lengths)

    model, history, split = train_classifier(
        X,
        y,
        hidden_sizes=tuple(args.hidden_sizes),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    model.save(args.output)

    print()
    print(f"Modelo guardado en: {args.output}")
    print(f"Etiquetas: {', '.join(model.labels)}")
    print(f"Train samples: {len(split['train_idx'])}")

    if len(split["val_idx"]):
        print(f"Validation samples: {len(split['val_idx'])}")
        print(
            "Ultima metrica: "
            f"train_acc={history.train_accuracy[-1]:.3f}, "
            f"val_acc={history.val_accuracy[-1]:.3f}, "
            f"train_loss={history.train_loss[-1]:.4f}, "
            f"val_loss={history.val_loss[-1]:.4f}"
        )
    else:
        print("Validation samples: 0")
        print("Validacion omitida porque alguna clase tiene menos de 2 muestras.")
        print(
            "Ultima metrica: "
            f"train_acc={history.train_accuracy[-1]:.3f}, "
            f"train_loss={history.train_loss[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
