from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "constructo",
    "dimension",
    "definicion_conceptual",
    "definicion_operacional",
    "poblacion",
    "contexto",
    "fuente_teorica",
}

OUTPUT_COLUMNS = [
    "spec_id",
    "constructo",
    "dimension",
    "subdimension",
    "definicion_conceptual",
    "definicion_operacional",
    "indicador",
    "conducta_observable",
    "poblacion",
    "contexto",
    "formato_recomendado",
    "escala_respuesta",
    "reglas_item",
    "fuente_teorica",
    "nivel_evidencia",
    "riesgo_solapamiento",
    "notas",
]


def slugify(value: str, max_length: int = 36) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value.lower()).strip("_")
    return slug[:max_length].strip("_") or "sin_nombre"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def sentence_case(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    return value[0].upper() + value[1:]


def derive_indicator(row: pd.Series) -> str:
    indicator = clean_text(row.get("indicador"))
    if indicator:
        return slugify(indicator)

    subdimension = clean_text(row.get("subdimension"))
    dimension = clean_text(row.get("dimension"))
    operational = clean_text(row.get("definicion_operacional"))
    basis = subdimension or dimension or operational.split(".")[0]
    return slugify(basis)


def derive_observable(row: pd.Series) -> str:
    observable = clean_text(row.get("conducta_observable"))
    if observable:
        return sentence_case(observable)

    operational = clean_text(row.get("definicion_operacional"))
    if operational:
        return sentence_case(operational)

    dimension = clean_text(row.get("dimension"))
    return f"Manifiesta conductas observables asociadas con {dimension}."


def evidence_level(source: str) -> str:
    source_lower = source.lower()
    if source_lower in {"autor/anio/doi", "autor/ano/doi"}:
        return "pendiente_verificacion"
    if "doi" in source_lower or "10." in source_lower:
        return "fuente_con_doi"
    if "osf.io" in source_lower or "arxiv" in source_lower:
        return "preprint_o_repositorio"
    if source:
        return "fuente_declarada"
    return "pendiente_verificacion"


def overlap_risk(group: pd.DataFrame, row: pd.Series) -> str:
    operational = clean_text(row.get("definicion_operacional")).lower()
    if not operational:
        return "alto"

    same_construct = group[group["constructo"] == row["constructo"]]
    repeated = same_construct["definicion_operacional"].str.lower().eq(operational).sum()
    if repeated > 1:
        return "medio"
    if len(operational.split()) < 8:
        return "medio"
    return "bajo"


def item_rules(formato: str) -> str:
    if formato == "sjt":
        return (
            "Crear una situacion laboral concreta; incluir opciones plausibles; "
            "mapear cada opcion a nivel de efectividad o rasgo; evitar pistas obvias."
        )
    return (
        "Redactar en primera persona; medir una sola idea; evitar absolutos; "
        "mantener lenguaje claro para la poblacion objetivo; preservar indicador."
    )


def build_matrix(
    input_path: Path,
    output_path: Path,
    default_format: str,
    escala_respuesta: str,
) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(sorted(missing))}")

    data = data.copy()
    for column in data.columns:
        data[column] = data[column].map(clean_text)

    empty_required = data[list(REQUIRED_COLUMNS)].eq("").any(axis=1)
    if empty_required.any():
        rows = ", ".join(str(index + 2) for index in data.index[empty_required])
        raise ValueError(f"Hay celdas obligatorias vacias en filas CSV: {rows}")

    data["indicador"] = data.apply(derive_indicator, axis=1)
    data["conducta_observable"] = data.apply(derive_observable, axis=1)
    if "subdimension" not in data.columns:
        data["subdimension"] = ""
    if "notas" not in data.columns:
        data["notas"] = ""
    data["subdimension"] = data["subdimension"].map(clean_text)
    data["notas"] = data["notas"].map(clean_text)
    data["formato_recomendado"] = default_format
    data["escala_respuesta"] = escala_respuesta
    data["reglas_item"] = item_rules(default_format)
    data["nivel_evidencia"] = data["fuente_teorica"].map(evidence_level)
    data["riesgo_solapamiento"] = data.apply(lambda row: overlap_risk(data, row), axis=1)

    counters: dict[str, int] = {}
    spec_ids = []
    for _, row in data.iterrows():
        prefix = slugify(f"{row['constructo']}_{row['dimension']}", max_length=24).upper()
        counters[prefix] = counters.get(prefix, 0) + 1
        spec_ids.append(f"{prefix}-{counters[prefix]:03d}")
    data["spec_id"] = spec_ids

    output = data.reindex(columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(output)),
        "constructos": int(output["constructo"].nunique()),
        "dimensiones": int(output[["constructo", "dimension"]].drop_duplicates().shape[0]),
        "indicadores": int(output["indicador"].nunique()),
        "evidencia": output["nivel_evidencia"].value_counts().to_dict(),
        "riesgo_solapamiento": output["riesgo_solapamiento"].value_counts().to_dict(),
    }
    output_path.with_suffix(".resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normaliza constructos y dimensiones en una matriz teorica trazable para generar items."
    )
    parser.add_argument("--input", required=True, help="CSV con constructos, dimensiones, definiciones y fuentes.")
    parser.add_argument("--output", required=True, help="CSV de matriz teorica/especificacion.")
    parser.add_argument(
        "--formato",
        choices=["likert", "sjt"],
        default="likert",
        help="Formato de item que se generara despues.",
    )
    parser.add_argument(
        "--escala-respuesta",
        default="Likert 1-5: totalmente en desacuerdo a totalmente de acuerdo",
        help="Descripcion de la escala de respuesta prevista.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = build_matrix(
        input_path=Path(args.input),
        output_path=Path(args.output),
        default_format=args.formato,
        escala_respuesta=args.escala_respuesta,
    )
    print(
        json.dumps(
            {
                "archivo_generado": str(Path(args.output).resolve()),
                "filas": int(len(output)),
                "constructos": int(output["constructo"].nunique()),
                "indicadores": int(output["indicador"].nunique()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
