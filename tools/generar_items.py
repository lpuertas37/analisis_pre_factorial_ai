from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "spec_id",
    "constructo",
    "dimension",
    "indicador",
    "conducta_observable",
    "poblacion",
    "contexto",
    "fuente_teorica",
}

MASTER_COLUMNS = [
    "item_id",
    "construct",
    "dimension",
    "indicator",
    "text",
    "source",
    "polarity",
    "status",
    "risk_level",
    "review_notes",
]

SJT_COLUMNS = [
    "item_id",
    "construct",
    "dimension",
    "indicator",
    "situation",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "rubric",
    "source",
    "status",
    "risk_level",
    "review_notes",
]


LIKERT_TEMPLATES = [
    "En mi trabajo puedo {observable_infinitive}.",
    "Cuento con condiciones para {observable_infinitive}.",
    "Puedo {observable_infinitive} cuando la situacion lo requiere.",
    "Tengo claridad para {observable_infinitive} en mi contexto laboral.",
    "Me resulta habitual {observable_infinitive}.",
    "Dispongo de la informacion necesaria para {observable_infinitive}.",
]

LIKERT_REVERSE_TEMPLATES = [
    "Me cuesta {observable_infinitive} aun cuando la situacion lo requiere.",
    "En mi trabajo, rara vez logro {observable_infinitive}.",
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def slug_code(value: str, length: int = 3) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value.upper())
    letters = "".join(part[0] for part in parts if part)
    if len(letters) < length:
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        letters = (letters + compact)[:length]
    return (letters[:length] or "ITM").ljust(length, "X")


def normalize_observable(value: str) -> str:
    value = clean_text(value).rstrip(".")
    if not value:
        return "actuar de acuerdo con el indicador definido"
    return value[0].lower() + value[1:]


def observable_as_infinitive(value: str) -> str:
    text = normalize_observable(value)
    text = re.sub(r"\bsus\b", "mis", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsu\b", "mi", text, flags=re.IGNORECASE)
    replacements = [
        (r"^recibe\b", "recibir"),
        (r"^identifica\b", "identificar"),
        (r"^reconoce\b", "reconocer"),
        (r"^aplica\b", "aplicar"),
        (r"^utiliza\b", "utilizar"),
        (r"^usa\b", "usar"),
        (r"^comunica\b", "comunicar"),
        (r"^evalua\b", "evaluar"),
        (r"^evalua\b", "evaluar"),
        (r"^resuelve\b", "resolver"),
        (r"^colabora\b", "colaborar"),
        (r"^gestiona\b", "gestionar"),
        (r"^protege\b", "proteger"),
        (r"^contrasta\b", "contrastar"),
        (r"^verifica\b", "verificar"),
        (r"^manifiesta\b", "manifestar"),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, text):
            return re.sub(pattern, replacement, text, count=1)
    return text


def trim_item(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("..", ".")
    if text and text[-1] not in ".?!":
        text += "."
    return text


def second_person_phrase(value: str) -> str:
    text = value
    text = re.sub(r"\bmis\b", "tus", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmi\b", "tu", text, flags=re.IGNORECASE)
    return text


def context_phrase(value: str) -> str:
    context = clean_text(value) or "la organizacion"
    lowered = context.lower()
    if lowered.startswith(("la ", "el ", "los ", "las ", "una ", "un ")):
        return context
    return f"la {context}"


def risk_level(text: str) -> str:
    words = text.split()
    absolutes = {"siempre", "nunca", "todos", "nadie", "totalmente", "completamente"}
    if any(word.strip(".,;:").lower() in absolutes for word in words):
        return "medio"
    if len(words) > 28:
        return "medio"
    return "bajo"


def review_notes(text: str, polarity: str) -> str:
    notes = []
    if risk_level(text) == "medio":
        notes.append("Revisar longitud o absolutos.")
    if polarity == "inversa":
        notes.append("Confirmar que la inversion no cambie el constructo.")
    return " ".join(notes) or "Candidato generado desde matriz teorica."


def generate_likert_items(row: pd.Series, count: int, include_reverse: bool) -> list[dict[str, str]]:
    observable_lower = normalize_observable(row["conducta_observable"])
    observable_infinitive = observable_as_infinitive(row["conducta_observable"])
    direct_templates = LIKERT_TEMPLATES[: max(1, count)]

    rows = []
    for index, template in enumerate(direct_templates, start=1):
        text = trim_item(
            template.format(
                observable_lower=observable_lower,
                observable_infinitive=observable_infinitive,
            )
        )
        rows.append(build_master_row(row, index, text, "directa"))

    if include_reverse:
        for offset, template in enumerate(LIKERT_REVERSE_TEMPLATES, start=len(rows) + 1):
            text = trim_item(
                template.format(
                    observable_lower=observable_lower,
                    observable_infinitive=observable_infinitive,
                )
            )
            rows.append(build_master_row(row, offset, text, "inversa"))

    return rows


def build_master_row(row: pd.Series, index: int, text: str, polarity: str) -> dict[str, str]:
    code = f"{slug_code(row['constructo'], 2)}-{slug_code(row['dimension'], 3)}-{index:03d}"
    risk = risk_level(text)
    return {
        "item_id": code,
        "construct": row["constructo"],
        "dimension": row["dimension"],
        "indicator": row["indicador"],
        "text": text,
        "source": row["fuente_teorica"],
        "polarity": polarity,
        "status": "candidato",
        "risk_level": risk,
        "review_notes": review_notes(text, polarity),
    }


def generate_sjt_item(row: pd.Series, index: int) -> dict[str, str]:
    observable_first_person = observable_as_infinitive(row["conducta_observable"])
    observable_second_person = second_person_phrase(observable_first_person)
    context = context_phrase(row["contexto"])
    code = f"{slug_code(row['constructo'], 2)}-{slug_code(row['dimension'], 3)}-SJT-{index:03d}"
    situation = trim_item(
        f"Estas en {context} y debes {observable_second_person} mientras atiendes una situacion laboral ambigua."
    )
    return {
        "item_id": code,
        "construct": row["constructo"],
        "dimension": row["dimension"],
        "indicator": row["indicador"],
        "situation": situation,
        "option_a": trim_item(f"Actuo de forma consistente para {observable_first_person}, verificando el impacto de mi decision"),
        "option_b": trim_item(f"Intento {observable_first_person}, pero decido solo con la informacion disponible inicialmente"),
        "option_c": trim_item("Pospongo la decision hasta que otra persona defina que hacer"),
        "option_d": trim_item("Respondo de manera rapida sin revisar si mi accion se ajusta al objetivo"),
        "rubric": "A=alto ajuste; B=ajuste medio; C=bajo ajuste por evitacion; D=bajo ajuste por impulsividad.",
        "source": row["fuente_teorica"],
        "status": "candidato",
        "risk_level": "medio",
        "review_notes": "SJT generado como borrador: requiere revision experta de realismo, opciones y rubrica.",
    }


def ensure_unique_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, int] = {}
    for row in rows:
        item_id = row["item_id"]
        seen[item_id] = seen.get(item_id, 0) + 1
        if seen[item_id] > 1:
            row["item_id"] = f"{item_id}-{seen[item_id]}"
    return rows


def load_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en matriz teorica: {', '.join(sorted(missing))}")

    data = data.copy()
    for column in data.columns:
        data[column] = data[column].map(clean_text)

    empty_required = data[list(REQUIRED_COLUMNS)].eq("").any(axis=1)
    if empty_required.any():
        rows = ", ".join(str(index + 2) for index in data.index[empty_required])
        raise ValueError(f"Hay celdas obligatorias vacias en filas CSV: {rows}")

    return data


def generate_items(
    input_path: Path,
    output_path: Path,
    formato: str,
    per_indicator: int,
    include_reverse: bool,
    minimal_output: Path | None,
) -> pd.DataFrame:
    matrix = load_matrix(input_path)
    generated: list[dict[str, str]] = []

    for _, row in matrix.iterrows():
        if formato == "sjt":
            generated.append(generate_sjt_item(row, len(generated) + 1))
        else:
            generated.extend(generate_likert_items(row, per_indicator, include_reverse))

    generated = ensure_unique_ids(generated)
    columns = SJT_COLUMNS if formato == "sjt" else MASTER_COLUMNS
    output = pd.DataFrame(generated).reindex(columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")

    if formato == "likert":
        target = minimal_output or output_path.with_name(output_path.stem + "_minimo.csv")
        minimal = output[["item_id", "construct", "text"]].rename(columns={"construct": "construct"})
        minimal.to_csv(target, index=False, encoding="utf-8")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "formato": formato,
        "items": int(len(output)),
        "constructos": int(output["construct"].nunique()),
        "dimensiones": int(output[["construct", "dimension"]].drop_duplicates().shape[0]),
        "riesgo": output["risk_level"].value_counts().to_dict(),
    }
    output_path.with_suffix(".resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera items candidatos desde una matriz teorica trazable."
    )
    parser.add_argument("--input", required=True, help="CSV generado por generar_matriz_teorica.py.")
    parser.add_argument("--output", required=True, help="CSV de items candidatos.")
    parser.add_argument("--formato", choices=["likert", "sjt"], default="likert")
    parser.add_argument(
        "--por-indicador",
        type=int,
        default=4,
        help="Numero de items Likert directos por indicador.",
    )
    parser.add_argument(
        "--incluir-inversos",
        action="store_true",
        help="Agrega borradores inversos para cada indicador. Requieren revision experta.",
    )
    parser.add_argument(
        "--minimo-output",
        default="",
        help="Ruta opcional para CSV minimo item_id,construct,text compatible con analisis_pre_factorial.py.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = generate_items(
        input_path=Path(args.input),
        output_path=Path(args.output),
        formato=args.formato,
        per_indicator=max(1, args.por_indicador),
        include_reverse=args.incluir_inversos,
        minimal_output=Path(args.minimo_output) if args.minimo_output else None,
    )
    print(
        json.dumps(
            {
                "archivo_generado": str(Path(args.output).resolve()),
                "items": int(len(output)),
                "formato": args.formato,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
