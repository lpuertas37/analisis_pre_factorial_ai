from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "source_id",
    "constructo",
    "instrumento",
    "referencia",
    "doi_o_url",
}

ITEM_COLUMNS = {
    "item_id_original",
    "item_text",
}

INSTRUMENT_COLUMNS = [
    "source_id",
    "constructo",
    "dimension",
    "instrumento",
    "referencia",
    "doi_o_url",
    "poblacion_original",
    "contexto_original",
    "idioma",
    "adaptacion_contexto",
    "escala_respuesta",
    "n_items_reportados",
    "nivel_trazabilidad",
    "utilidad_como_punto_partida",
    "notas",
]

ITEM_OUTPUT_COLUMNS = [
    "source_id",
    "instrumento",
    "constructo",
    "dimension",
    "item_id_original",
    "item_text",
    "idioma",
    "contexto_original",
    "adaptacion_contexto",
    "escala_respuesta",
    "decision_inicial",
    "motivo_decision",
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def traceability_level(row: pd.Series) -> str:
    doi = clean_text(row.get("doi_o_url")).lower()
    item_text = clean_text(row.get("item_text"))
    instrument = clean_text(row.get("instrumento"))
    if item_text and ("10." in doi or "doi" in doi or "osf.io" in doi or "arxiv" in doi):
        return "alta"
    if instrument and ("10." in doi or "doi" in doi or "osf.io" in doi or "arxiv" in doi):
        return "media"
    return "pendiente"


def context_utility(row: pd.Series, target_context: str, target_population: str) -> str:
    context = clean_text(row.get("contexto_original")).lower()
    population = clean_text(row.get("poblacion_original")).lower()
    adaptation = clean_text(row.get("adaptacion_contexto")).lower()

    context_match = target_context and target_context.lower() in context
    population_match = target_population and target_population.lower() in population
    adapted = adaptation not in {"", "no", "no reportada", "none", "n/a"}

    if context_match and population_match:
        return "muy_alta"
    if context_match or population_match or adapted:
        return "alta"
    return "media"


def initial_item_decision(text: str) -> tuple[str, str]:
    if not text:
        return "buscar_item", "La fuente registra instrumento pero no item textual."
    word_count = len(text.split())
    if word_count < 4:
        return "revisar", "Item demasiado corto o incompleto."
    if word_count > 35:
        return "revisar", "Item largo; conviene revisar doble contenido."
    if re.search(r"\b(siempre|nunca|todos|nadie)\b", text, flags=re.IGNORECASE):
        return "revisar", "Contiene absoluto que puede sesgar la respuesta."
    return "usar_como_base", "Item textual disponible para adaptacion o comparacion."


def prepare_review(
    input_path: Path,
    output_dir: Path,
    target_context: str,
    target_population: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(sorted(missing))}")

    data = data.copy()
    for column in data.columns:
        data[column] = data[column].map(clean_text)

    for column in [
        "dimension",
        "poblacion_original",
        "contexto_original",
        "idioma",
        "adaptacion_contexto",
        "item_id_original",
        "item_text",
        "escala_respuesta",
        "notas",
    ]:
        if column not in data.columns:
            data[column] = ""

    instrument_rows = []
    for keys, group in data.groupby(["source_id", "instrumento"], dropna=False):
        first = group.iloc[0]
        item_count = int(group["item_text"].ne("").sum())
        instrument_rows.append(
            {
                "source_id": keys[0],
                "constructo": first["constructo"],
                "dimension": first["dimension"],
                "instrumento": first["instrumento"],
                "referencia": first["referencia"],
                "doi_o_url": first["doi_o_url"],
                "poblacion_original": first["poblacion_original"],
                "contexto_original": first["contexto_original"],
                "idioma": first["idioma"],
                "adaptacion_contexto": first["adaptacion_contexto"],
                "escala_respuesta": first["escala_respuesta"],
                "n_items_reportados": item_count,
                "nivel_trazabilidad": traceability_level(first),
                "utilidad_como_punto_partida": context_utility(first, target_context, target_population),
                "notas": first["notas"],
            }
        )

    item_rows = []
    if ITEM_COLUMNS <= set(data.columns):
        for _, row in data.iterrows():
            decision, motive = initial_item_decision(row["item_text"])
            item_rows.append(
                {
                    "source_id": row["source_id"],
                    "instrumento": row["instrumento"],
                    "constructo": row["constructo"],
                    "dimension": row["dimension"],
                    "item_id_original": row["item_id_original"],
                    "item_text": row["item_text"],
                    "idioma": row["idioma"],
                    "contexto_original": row["contexto_original"],
                    "adaptacion_contexto": row["adaptacion_contexto"],
                    "escala_respuesta": row["escala_respuesta"],
                    "decision_inicial": decision,
                    "motivo_decision": motive,
                }
            )

    instruments = pd.DataFrame(instrument_rows).reindex(columns=INSTRUMENT_COLUMNS)
    items = pd.DataFrame(item_rows).reindex(columns=ITEM_OUTPUT_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    instruments.to_csv(output_dir / "instrumentos_existentes.csv", index=False, encoding="utf-8")
    items.to_csv(output_dir / "items_existentes.csv", index=False, encoding="utf-8")

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "instrumentos": int(len(instruments)),
        "items_textuales": int(items["item_text"].ne("").sum()) if not items.empty else 0,
        "trazabilidad": instruments["nivel_trazabilidad"].value_counts().to_dict(),
        "utilidad": instruments["utilidad_como_punto_partida"].value_counts().to_dict(),
    }
    (output_dir / "revision_instrumentos_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return instruments, items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estructura papers, instrumentos e items existentes como punto de partida para adaptar bancos de items."
    )
    parser.add_argument("--input", required=True, help="CSV de fuentes e instrumentos existentes.")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida.")
    parser.add_argument("--contexto-objetivo", default="", help="Contexto donde se usara el instrumento.")
    parser.add_argument("--poblacion-objetivo", default="", help="Poblacion objetivo del nuevo instrumento.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    instruments, items = prepare_review(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        target_context=args.contexto_objetivo,
        target_population=args.poblacion_objetivo,
    )
    print(
        json.dumps(
            {
                "instrumentos": int(len(instruments)),
                "items": int(len(items)),
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
