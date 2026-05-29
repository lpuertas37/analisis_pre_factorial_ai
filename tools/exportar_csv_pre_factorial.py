from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["item_id", "construct", "text"]


def export_pre_factorial(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    exported = data[REQUIRED_COLUMNS].copy()
    exported["item_id"] = exported["item_id"].astype(str).str.strip()
    exported["construct"] = exported["construct"].astype(str).str.strip()
    exported["text"] = exported["text"].astype(str).str.strip()

    empty_rows = exported.eq("").any(axis=1)
    if empty_rows.any():
        rows = ", ".join(str(index + 2) for index in exported.index[empty_rows])
        raise ValueError(f"Hay celdas vacias en filas CSV: {rows}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta una matriz maestra de items al formato minimo del analisis pre-factorial."
    )
    parser.add_argument("--input", required=True, help="CSV maestro con item_id, construct y text.")
    parser.add_argument("--output", required=True, help="CSV de salida compatible con analisis_pre_factorial.py.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_pre_factorial(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
