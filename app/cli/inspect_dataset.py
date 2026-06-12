"""

Funcion
-------
Inspecciona un dataset HDF5 de senas LESCO para revisar muestras,
etiquetas, calidad de datos y ejemplos guardados.

Comandos
--------
uv run python -m app.cli.inspect_dataset data/signs_dataset.h5
    Muestra un resumen general del dataset.

uv run python -m app.cli.inspect_dataset data/signs_dataset.h5 --label HOLA
    Muestra estadisticas de una etiqueta especifica.

uv run python -m app.cli.inspect_dataset data/signs_dataset.h5 --plot 3
    Grafica la trayectoria de landmarks de la muestra indicada.

uv run python -m app.cli.inspect_dataset data/signs_dataset.h5 --export 0
    Exporta una muestra a CSV para inspeccion externa.

uv run python -m app.cli.inspect_dataset data/signs_dataset.h5 --delete-label HOLA
    Elimina todas las muestras de una etiqueta.

Notas
-----
Usalo para verificar el dataset antes de entrenar o reconocer senas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_labels(raw: np.ndarray) -> list[str]:
    return [lb.decode("utf-8") if isinstance(lb, bytes) else str(lb) for lb in raw]


def _load_file(path: Path) -> h5py.File:
    if not path.exists():
        print(f"[ERROR] Archivo no encontrado: {path}", file=sys.stderr)
        sys.exit(1)
    return h5py.File(path, "r")


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_summary(path: Path) -> None:
    """Print a human-readable summary of the dataset."""
    with _load_file(path) as f:
        n = f["X"].shape[0]
        sl = int(f.attrs.get("sequence_length", "?"))
        fp = int(f.attrs.get("features_per_frame", "?"))
        version = f.attrs.get("version", "?")
        created = f.attrs.get("created_at", "?")

        print("=" * 58)
        print(f"  LESCO Dataset  –  {path.name}")
        print("=" * 58)
        print(f"  Versión           : {version}")
        print(f"  Creado            : {created}")
        print(f"  Total de muestras : {n}")
        print(f"  Forma de X        : ({n}, {sl}, {fp})")
        print(f"    • Frames/muestra: {sl}")
        print(f"    • Features/frame: {fp}  (2 manos × 21 puntos × 3 coords)")
        print(f"  Tamaño en disco   : {path.stat().st_size / 1024:.1f} KB")
        print()

        if n == 0:
            print("  (dataset vacío)")
            return

        labels = _decode_labels(f["labels"][:])
        unique, counts = np.unique(labels, return_counts=True)
        orig_lengths = f["original_lengths"][:]

        print(f"  {'Etiqueta':<20} {'Muestras':>8} {'Frames_raw (min/med/max)':>26}")
        print(f"  {'-'*20} {'-'*8} {'-'*26}")
        for lbl, cnt in zip(unique, counts):
            mask = np.array(labels) == lbl
            lo = orig_lengths[mask].min()
            med = int(np.median(orig_lengths[mask]))
            hi = orig_lengths[mask].max()
            print(f"  {lbl:<20} {cnt:>8}    {lo:>4} / {med:>4} / {hi:>4}")
        print()

        # Basic data quality check
        X = f["X"][:]
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        zero_rows = int((X.reshape(n, -1) == 0).all(axis=1).sum())
        print(f"  Calidad de datos:")
        print(f"    NaN       : {nan_count}")
        print(f"    Inf       : {inf_count}")
        print(f"    Muestras todo-cero : {zero_rows}")
        if nan_count == 0 and inf_count == 0:
            print("    OK: Sin valores invalidos detectados")
        print()


def cmd_label_detail(path: Path, label: str) -> None:
    """Show detailed statistics for one label."""
    with _load_file(path) as f:
        labels = _decode_labels(f["labels"][:])
        indices = [i for i, lb in enumerate(labels) if lb == label.upper()]

        if not indices:
            print(f"[AVISO] Etiqueta '{label}' no encontrada en el dataset.")
            return

        X_sub = f["X"][indices]            # (k, seq_len, features)
        orig = f["original_lengths"][indices]

        print(f"\n  Etiqueta: {label.upper()}  ({len(indices)} muestras)")
        print(f"  {'Idx':>5}  {'frames_raw':>10}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for k, (ds_idx, raw_len) in enumerate(zip(indices, orig)):
            seq = X_sub[k].reshape(-1)
            print(
                f"  {ds_idx:>5}  {raw_len:>10}  {seq.mean():>8.4f}"
                f"  {seq.std():>8.4f}  {seq.min():>8.4f}  {seq.max():>8.4f}"
            )
        print()


def cmd_plot(path: Path, sample_index: int) -> None:
    """
    Plot the landmark trajectory for one sample.

    Requires matplotlib (not a core dependency; installed separately).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib no está instalado.  Ejecuta: uv add matplotlib")
        sys.exit(1)

    with _load_file(path) as f:
        n = f["X"].shape[0]
        if sample_index >= n:
            print(f"[ERROR] Índice {sample_index} fuera de rango (total: {n})")
            sys.exit(1)

        seq = f["X"][sample_index]        # (seq_len, features)
        label_raw = f["labels"][sample_index]
        label = label_raw.decode("utf-8") if isinstance(label_raw, bytes) else str(label_raw)
        orig_len = int(f["original_lengths"][sample_index])

    seq_len, features = seq.shape
    # Split by hand slot (63 features each)
    right = seq[:, :63].reshape(seq_len, 21, 3)   # (T, 21, 3)
    left  = seq[:, 63:].reshape(seq_len, 21, 3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(
        f"Muestra #{sample_index}  –  {label}  "
        f"(frames raw: {orig_len}  →  resampled: {seq_len})",
        fontsize=13,
    )

    hand_data = [("Derecha (slot 0)", right), ("Izquierda (slot 1)", left)]
    coord_names = ["x", "y", "z"]
    colors = plt.cm.tab10(np.linspace(0, 1, 21))  # one color per landmark

    for row, (hand_name, hand_seq) in enumerate(hand_data):
        for col, coord in enumerate(coord_names):
            ax = axes[row][col]
            for lm_idx in range(21):
                ax.plot(hand_seq[:, lm_idx, col], color=colors[lm_idx], linewidth=0.8)
            ax.set_title(f"{hand_name} – {coord}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Valor normalizado")
            ax.grid(True, linewidth=0.4)

            # Shade zero regions (where the hand was absent / padding)
            flat = hand_seq[:, :, col]
            is_zero_frame = (flat == 0).all(axis=1)
            for t in range(seq_len):
                if is_zero_frame[t]:
                    ax.axvspan(t - 0.5, t + 0.5, color="red", alpha=0.08)

    plt.tight_layout()
    plt.show()


def cmd_export(path: Path, sample_index: int) -> None:
    """Export one sample to a CSV file for inspection in a spreadsheet."""
    import csv

    with _load_file(path) as f:
        n = f["X"].shape[0]
        if sample_index >= n:
            print(f"[ERROR] Índice {sample_index} fuera de rango (total: {n})")
            sys.exit(1)

        seq = f["X"][sample_index]
        label_raw = f["labels"][sample_index]
        label = label_raw.decode("utf-8") if isinstance(label_raw, bytes) else str(label_raw)

    out_path = path.parent / f"sample_{sample_index}_{label}.csv"

    # Build header: frame, then one column per feature grouped as
    # R_lm00_x, R_lm00_y, R_lm00_z, ..., L_lm20_z
    header = ["frame"]
    for side in ["R", "L"]:
        for lm in range(21):
            for coord in ["x", "y", "z"]:
                header.append(f"{side}_lm{lm:02d}_{coord}")

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for t, row in enumerate(seq):
            writer.writerow([t] + row.tolist())

    print(f"  Exportado: {out_path}  ({seq.shape[0]} frames × {seq.shape[1]} features)")


def cmd_delete_label(path: Path, label: str) -> None:
    """
    Remove all samples for *label* and rewrite the dataset.

    This rewrites the file because HDF5 cannot shrink a dataset in-place
    without a full copy (h5repack / h5copy).  For large datasets consider
    marking samples as invalid instead.
    """
    label = label.upper()

    with h5py.File(path, "r") as f:
        labels = _decode_labels(f["labels"][:])
        keep = np.array([lb != label for lb in labels])
        n_remove = int((~keep).sum())

        if n_remove == 0:
            print(f"  [AVISO] Etiqueta '{label}' no encontrada.")
            return

        X_keep = f["X"][:][keep]
        labels_keep = np.array(labels)[keep]
        orig_keep = f["original_lengths"][:][keep]
        attrs = dict(f.attrs)

    print(f"  Eliminando {n_remove} muestras de '{label}' …")
    tmp = path.with_suffix(".tmp.h5")

    with h5py.File(tmp, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v

        str_dtype = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("X", data=X_keep, compression="gzip", compression_opts=4)
        f.create_dataset(
            "labels",
            data=np.array([lb.encode("utf-8") for lb in labels_keep], dtype=object),
            dtype=str_dtype,
        )
        f.create_dataset("original_lengths", data=orig_keep)

    tmp.replace(path)
    print(f"  Listo. Quedaron {len(labels_keep)} muestras en {path.name}.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspecciona y verifica un dataset HDF5 de LESCO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", type=Path, help="Ruta al archivo .h5")
    parser.add_argument("--label", help="Mostrar detalle de una etiqueta específica")
    parser.add_argument("--plot", type=int, metavar="IDX",
                        help="Graficar la trayectoria del sample en el índice IDX")
    parser.add_argument("--export", type=int, metavar="IDX",
                        help="Exportar el sample IDX a CSV")
    parser.add_argument("--delete-label", metavar="LABEL",
                        help="Eliminar todas las muestras de una etiqueta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.delete_label:
        cmd_delete_label(args.dataset, args.delete_label)
        return

    cmd_summary(args.dataset)

    if args.label:
        cmd_label_detail(args.dataset, args.label)

    if args.plot is not None:
        cmd_plot(args.dataset, args.plot)

    if args.export is not None:
        cmd_export(args.dataset, args.export)


if __name__ == "__main__":
    main()
