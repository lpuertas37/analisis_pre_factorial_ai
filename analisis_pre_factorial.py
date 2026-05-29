"""Analisis pre-factorial para bancos de items generados o revisados con IA.

Este script operacionaliza la idea de "embedding-based factor analysis" descrita
por Suarez-Alvarez et al. (2026): convertir textos de items en representaciones
numericas, construir una matriz de similitud, extraer una estructura factorial
exploratoria y revisar residuales de forma model-free.

Entrada minima:
    CSV con columnas: item_id,text

Entrada recomendada:
    CSV con columnas: item_id,text,construct

Ejemplo:
    python analisis_pre_factorial.py --items ejemplo_items.csv --n-factors 3
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MPL_CACHE = Path(__file__).resolve().parent / ".matplotlib-cache" / f"run-{os.getpid()}"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {"item_id", "text"}


@dataclass
class FactorConfig:
    n_factors: int | None
    min_primary_loading: float
    max_cross_loading: float
    max_cross_loading_ratio: float
    min_loading_gap: float
    rotation: str
    backend: str
    sentence_model: str


@dataclass
class AnalysisResult:
    items: pd.DataFrame
    report: pd.DataFrame
    summary: dict
    similarity: np.ndarray
    residual: np.ndarray
    eigenvalues: np.ndarray
    loadings: np.ndarray


def read_items(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {path}: {sorted(missing)}")

    df = df.copy()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("")]

    if df["item_id"].duplicated().any():
        duplicated = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"Hay item_id duplicados: {duplicated[:10]}")

    if len(df) < 4:
        raise ValueError("Se necesitan al menos 4 items para una revision factorial minima.")

    return df.reset_index(drop=True)


def make_embeddings(texts: list[str], config: FactorConfig) -> tuple[np.ndarray, str]:
    backend = config.backend
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            backend = "sentence-transformers"
        except ImportError:
            backend = "tfidf"

    if backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Instala sentence-transformers o usa --backend tfidf."
            ) from exc

        model = SentenceTransformer(config.sentence_model)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=float), f"sentence-transformers:{config.sentence_model}"

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray().astype(float), "tfidf:char_wb_3_5"


def nearest_positive_semidefinite(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(symmetric)
    vals = np.clip(vals, eps, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    repaired = (repaired + repaired.T) / 2
    np.fill_diagonal(repaired, 1.0)
    return repaired


def choose_n_factors(eigenvalues: np.ndarray, max_factors: int = 8) -> int:
    positive = eigenvalues[eigenvalues > 1.0]
    if len(positive) == 0:
        return 1
    return int(min(len(positive), max_factors))


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    p, k = loadings.shape
    rotation = np.eye(k)
    previous = 0.0

    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, s, vh = np.linalg.svd(
            loadings.T
            @ (rotated**3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        )
        rotation = u @ vh
        current = float(np.sum(s))
        if previous and current < previous * (1 + tol):
            break
        previous = current

    return loadings @ rotation


def rotate_loadings(loadings: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "none" or loadings.shape[1] <= 1:
        return loadings
    if rotation == "varimax":
        return varimax(loadings)
    if rotation == "oblimin":
        try:
            from factor_analyzer.rotator import Rotator
        except ImportError as exc:
            raise RuntimeError(
                "Para usar --rotation oblimin instala factor-analyzer: pip install factor-analyzer"
            ) from exc

        rotator = Rotator(method="oblimin")
        return np.asarray(rotator.fit_transform(loadings), dtype=float)
    raise ValueError(f"Rotacion no soportada: {rotation}")


def factor_from_similarity(similarity: np.ndarray, n_factors: int | None, rotation: str) -> tuple[np.ndarray, np.ndarray, int]:
    repaired = nearest_positive_semidefinite(similarity)
    eigenvalues, eigenvectors = np.linalg.eigh(repaired)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    selected = n_factors or choose_n_factors(eigenvalues)
    selected = max(1, min(selected, repaired.shape[0] - 1))
    raw_loadings = eigenvectors[:, :selected] * np.sqrt(np.clip(eigenvalues[:selected], 0, None))
    rotated = rotate_loadings(raw_loadings, rotation)
    return rotated, eigenvalues, selected


def kmo_from_correlation(corr: np.ndarray) -> tuple[float, np.ndarray]:
    inv_corr = np.linalg.pinv(corr)
    partial = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial, 0)

    corr_no_diag = corr.copy()
    np.fill_diagonal(corr_no_diag, 0)

    corr_sq = corr_no_diag**2
    partial_sq = partial**2
    overall = float(corr_sq.sum() / (corr_sq.sum() + partial_sq.sum()))
    per_item = corr_sq.sum(axis=0) / (corr_sq.sum(axis=0) + partial_sq.sum(axis=0))
    return overall, per_item


def factor_decisions(loadings: np.ndarray, config: FactorConfig) -> pd.DataFrame:
    abs_loadings = np.abs(loadings)
    primary_idx = abs_loadings.argmax(axis=1)
    primary = abs_loadings[np.arange(loadings.shape[0]), primary_idx]

    sorted_abs = np.sort(abs_loadings, axis=1)
    secondary = sorted_abs[:, -2] if loadings.shape[1] > 1 else np.zeros(loadings.shape[0])
    gap = primary - secondary
    ratio = np.divide(secondary, primary, out=np.zeros_like(secondary), where=primary > 0)

    status = np.where(
        primary < config.min_primary_loading,
        "revisar: carga primaria baja",
        np.where(
            gap < config.min_loading_gap,
            "revisar: baja diferenciacion",
            np.where(
                (secondary > config.max_cross_loading) & (ratio > config.max_cross_loading_ratio),
                "revisar: posible carga cruzada",
                "conservar",
            ),
        ),
    )

    return pd.DataFrame(
        {
            "assigned_factor": primary_idx + 1,
            "primary_abs_loading": primary.round(4),
            "secondary_abs_loading": secondary.round(4),
            "secondary_primary_ratio": ratio.round(4),
            "loading_gap": gap.round(4),
            "decision": status,
        }
    )


def reconstruct_correlation(loadings: np.ndarray) -> np.ndarray:
    communalities = np.sum(loadings**2, axis=1)
    uniqueness = np.clip(1 - communalities, 0, 1)
    reproduced = loadings @ loadings.T
    np.fill_diagonal(reproduced, communalities + uniqueness)
    return reproduced


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: Iterable[str]) -> None:
    labels = list(labels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *np.round(row, 6)])


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path, center: float = 0.0) -> None:
    size = max(7, min(18, len(labels) * 0.45))
    plt.figure(figsize=(size, size))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="vlag",
        center=center,
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scree(eigenvalues: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(eigenvalues) + 1)
    plt.plot(x, eigenvalues, marker="o")
    plt.axhline(1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Componente/factor")
    plt.ylabel("Autovalor")
    plt.title("Scree plot de la matriz de similitud")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_residual_distribution(residual: np.ndarray, path: Path) -> dict[str, float]:
    off_diag = residual[~np.eye(residual.shape[0], dtype=bool)]
    stats = {
        "mean_abs_residual": float(np.mean(np.abs(off_diag))),
        "median_abs_residual": float(np.median(np.abs(off_diag))),
        "p95_abs_residual": float(np.quantile(np.abs(off_diag), 0.95)),
        "max_abs_residual": float(np.max(np.abs(off_diag))),
    }

    plt.figure(figsize=(8, 5))
    sns.histplot(off_diag, bins=30, kde=True)
    plt.xlabel("Residual fuera de la diagonal")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion de residuales")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return stats


def validate_constructs(report: pd.DataFrame) -> tuple[pd.DataFrame | None, dict | None]:
    if "construct" not in report.columns:
        return None, None

    construct_df = (
        report.groupby(["construct", "assigned_factor"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["construct", "n"], ascending=[True, False])
    )
    totals = report.groupby("construct", dropna=False).size().rename("total").reset_index()
    dominant = construct_df.drop_duplicates("construct").merge(totals, on="construct")
    dominant["dominant_factor_match_rate"] = (dominant["n"] / dominant["total"]).round(4)
    dominant = dominant.rename(columns={"assigned_factor": "dominant_factor"})

    ari = adjusted_rand_score(report["construct"].astype(str), report["assigned_factor"].astype(str))
    summary = {
        "construct_adjusted_rand_index": round(float(ari), 4),
        "mean_construct_dominant_factor_match": round(float(dominant["dominant_factor_match_rate"].mean()), 4),
        "constructs": dominant.to_dict(orient="records"),
    }
    return dominant, summary


def write_html_report(output_dir: Path, summary: dict, report: pd.DataFrame, construct_validation: pd.DataFrame | None) -> None:
    review_items = report[report["decision"].ne("conservar")].copy()
    review_html = (
        review_items[["item_id", "construct", "text", "assigned_factor", "primary_abs_loading", "secondary_abs_loading", "secondary_primary_ratio", "loading_gap", "decision"]]
        .to_html(index=False, escape=True)
        if "construct" in report.columns
        else review_items[["item_id", "text", "assigned_factor", "primary_abs_loading", "secondary_abs_loading", "secondary_primary_ratio", "loading_gap", "decision"]].to_html(index=False, escape=True)
    )
    construct_html = (
        construct_validation.to_html(index=False, escape=True)
        if construct_validation is not None
        else "<p>No se incluyo columna <code>construct</code>; se omitio esta validacion.</p>"
    )

    body = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte pre-factorial</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
    .metric {{ border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; background: #fbfcfd; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; margin: 8px 0 24px; }}
    code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Reporte pre-factorial semantico</h1>
  <p>Analisis inspirado en embedding-based factor analysis: embeddings, matriz de similitud, pseudo factor analysis, residuales y refinamiento de items.</p>
  <div class="grid">
    <div class="metric"><strong>Items</strong><br>{summary["items"]}</div>
    <div class="metric"><strong>Backend</strong><br>{html.escape(summary["embedding_backend"])}</div>
    <div class="metric"><strong>Factores</strong><br>{summary["selected_factors"]}</div>
    <div class="metric"><strong>KMO pseudo</strong><br>{summary["kmo_total_pseudo"]}</div>
  </div>
  <h2>Items a revisar</h2>
  {review_html}
  <h2>Validacion contra constructos esperados</h2>
  {construct_html}
  <h2>Graficos</h2>
  <h3>Scree plot</h3>
  <img src="scree_plot.png" alt="Scree plot">
  <h3>Similitud semantica</h3>
  <img src="heatmap_similitud.png" alt="Heatmap de similitud">
  <h3>Residuales</h3>
  <img src="heatmap_residuales.png" alt="Heatmap de residuales">
  <img src="distribucion_residuales.png" alt="Distribucion de residuales">
</body>
</html>
"""
    (output_dir / "reporte_pre_factorial.html").write_text(body, encoding="utf-8")


def analyze_once(items: pd.DataFrame, output_dir: Path, config: FactorConfig, write_outputs: bool = True) -> AnalysisResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, backend_label = make_embeddings(items["text"].tolist(), config)
    similarity = cosine_similarity(embeddings)
    similarity = np.clip(similarity, -1, 1)
    np.fill_diagonal(similarity, 1.0)

    loadings, eigenvalues, selected_factors = factor_from_similarity(similarity, config.n_factors, config.rotation)
    reproduced = reconstruct_correlation(loadings)
    residual = similarity - reproduced
    np.fill_diagonal(residual, 0)

    kmo_total, kmo_item = kmo_from_correlation(nearest_positive_semidefinite(similarity))
    decisions = factor_decisions(loadings, config)

    loadings_df = pd.DataFrame(
        loadings,
        columns=[f"factor_{i + 1}" for i in range(loadings.shape[1])],
    ).round(4)
    report = pd.concat(
        [
            items[["item_id", "text"] + (["construct"] if "construct" in items.columns else [])],
            loadings_df,
            decisions,
            pd.Series(kmo_item.round(4), name="kmo_item"),
        ],
        axis=1,
    )

    eigen_df = pd.DataFrame(
        {
            "factor": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "variance_ratio": eigenvalues / eigenvalues.sum(),
            "cumulative_variance": np.cumsum(eigenvalues / eigenvalues.sum()),
        }
    )
    residual_stats = {
        "mean_abs_residual": float(np.mean(np.abs(residual[~np.eye(residual.shape[0], dtype=bool)]))),
        "median_abs_residual": float(np.median(np.abs(residual[~np.eye(residual.shape[0], dtype=bool)]))),
        "p95_abs_residual": float(np.quantile(np.abs(residual[~np.eye(residual.shape[0], dtype=bool)]), 0.95)),
        "max_abs_residual": float(np.max(np.abs(residual[~np.eye(residual.shape[0], dtype=bool)]))),
    }
    construct_validation, construct_summary = validate_constructs(report)

    summary = {
        "items": int(len(items)),
        "embedding_backend": backend_label,
        "rotation": config.rotation,
        "selected_factors": int(selected_factors),
        "kmo_total_pseudo": round(kmo_total, 4),
        "eigenvalues_first_10": [round(float(x), 4) for x in eigenvalues[:10]],
        "retention_counts": report["decision"].value_counts().to_dict(),
        "residual_stats": {k: round(v, 4) for k, v in residual_stats.items()},
        "construct_validation": construct_summary,
        "interpretation_notes": [
            "Este analisis es pre-factorial: no reemplaza EFA/CFA con respuestas reales.",
            "Use cargas, cargas cruzadas y residuales para depurar el banco de items antes del pilotaje.",
            "KMO se calcula sobre la matriz de similitud semantica; tratelo como indicador descriptivo, no como prueba inferencial.",
        ],
    }

    if write_outputs:
        save_matrix_csv(output_dir / "matriz_similitud.csv", similarity, items["item_id"])
        save_matrix_csv(output_dir / "matriz_residuales.csv", residual, items["item_id"])
        report.to_csv(output_dir / "loadings_y_decisiones.csv", index=False, encoding="utf-8")
        eigen_df.to_csv(output_dir / "autovalores.csv", index=False, encoding="utf-8")
        if construct_validation is not None:
            construct_validation.to_csv(output_dir / "validacion_constructos.csv", index=False, encoding="utf-8")

        plot_heatmap(similarity, items["item_id"].tolist(), "Matriz de similitud semantica", output_dir / "heatmap_similitud.png", center=float(np.mean(similarity)))
        plot_heatmap(residual, items["item_id"].tolist(), "Residuales: similitud original - reproducida", output_dir / "heatmap_residuales.png")
        plot_scree(eigenvalues, output_dir / "scree_plot.png")
        plot_residual_distribution(residual, output_dir / "distribucion_residuales.png")
        write_html_report(output_dir, summary, report, construct_validation)
        (output_dir / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return AnalysisResult(items, report, summary, similarity, residual, eigenvalues, loadings)


def compare_models(items: pd.DataFrame, output_dir: Path, base_config: FactorConfig, model_names: list[str]) -> pd.DataFrame:
    rows = []
    assignments = {}
    for model_name in model_names:
        config = FactorConfig(
            n_factors=base_config.n_factors,
            min_primary_loading=base_config.min_primary_loading,
            max_cross_loading=base_config.max_cross_loading,
            max_cross_loading_ratio=base_config.max_cross_loading_ratio,
            min_loading_gap=base_config.min_loading_gap,
            rotation=base_config.rotation,
            backend="sentence-transformers",
            sentence_model=model_name,
        )
        model_dir = output_dir / "comparacion_modelos" / safe_name(model_name)
        result = analyze_once(items, model_dir, config, write_outputs=True)
        assignments[model_name] = result.report["assigned_factor"].astype(str).tolist()
        rows.append(
            {
                "model": model_name,
                "items": result.summary["items"],
                "selected_factors": result.summary["selected_factors"],
                "kmo_total_pseudo": result.summary["kmo_total_pseudo"],
                "conservar": result.summary["retention_counts"].get("conservar", 0),
                "revisar": int(result.summary["items"] - result.summary["retention_counts"].get("conservar", 0)),
                "mean_abs_residual": result.summary["residual_stats"]["mean_abs_residual"],
                "construct_ari": (result.summary.get("construct_validation") or {}).get("construct_adjusted_rand_index"),
            }
        )

    comparison = pd.DataFrame(rows)
    if len(assignments) > 1:
        reference_name = model_names[0]
        reference = assignments[reference_name]
        comparison["ari_vs_first_model"] = [
            1.0 if row["model"] == reference_name else round(float(adjusted_rand_score(reference, assignments[row["model"]])), 4)
            for _, row in comparison.iterrows()
        ]
    comparison.to_csv(output_dir / "comparacion_modelos.csv", index=False, encoding="utf-8")
    return comparison


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def iterative_refinement(items: pd.DataFrame, output_dir: Path, config: FactorConfig, max_iterations: int) -> pd.DataFrame:
    current = items.copy()
    rows = []
    for iteration in range(1, max_iterations + 1):
        iteration_dir = output_dir / "iteraciones" / f"iteracion_{iteration:02d}"
        result = analyze_once(current, iteration_dir, config, write_outputs=True)
        flagged = result.report[result.report["decision"].ne("conservar")]
        rows.append(
            {
                "iteration": iteration,
                "items_included": len(current),
                "items_flagged": len(flagged),
                "items_conserved": int(result.summary["retention_counts"].get("conservar", 0)),
                "kmo_total_pseudo": result.summary["kmo_total_pseudo"],
                "mean_abs_residual": result.summary["residual_stats"]["mean_abs_residual"],
                "removed_item_ids": ", ".join(flagged["item_id"].astype(str).tolist()),
            }
        )
        if flagged.empty or len(current) - len(flagged) < 4:
            break
        current = current[~current["item_id"].isin(flagged["item_id"])].reset_index(drop=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "resumen_iterativo.csv", index=False, encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FactorConfig(
        n_factors=args.n_factors,
        min_primary_loading=args.min_primary_loading,
        max_cross_loading=args.max_cross_loading,
        max_cross_loading_ratio=args.max_cross_loading_ratio,
        min_loading_gap=args.min_loading_gap,
        rotation=args.rotation,
        backend=args.backend,
        sentence_model=args.sentence_model,
    )

    items = read_items(Path(args.items))
    result = analyze_once(items, output_dir, config, write_outputs=True)

    if args.compare_models:
        models = [model.strip() for model in args.compare_models.split(",") if model.strip()]
        compare_models(items, output_dir, config, models)

    if args.iterate:
        iterative_refinement(items, output_dir, config, args.max_iterations)

    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    print(f"\nArchivos generados en: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisis pre-factorial semantico de items.")
    parser.add_argument("--items", required=True, help="CSV con columnas item_id,text y opcionalmente construct.")
    parser.add_argument("--output-dir", default="salidas_pre_factorial", help="Carpeta de salida.")
    parser.add_argument("--n-factors", type=int, default=None, help="Numero de factores. Si se omite usa autovalores > 1.")
    parser.add_argument(
        "--rotation",
        choices=["varimax", "oblimin", "none"],
        default="oblimin",
        help="Rotacion factorial. Oblimin permite factores correlacionados.",
    )
    parser.add_argument("--backend", choices=["auto", "tfidf", "sentence-transformers"], default="auto")
    parser.add_argument("--sentence-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument(
        "--compare-models",
        default="",
        help="Lista separada por comas de modelos sentence-transformers para comparar.",
    )
    parser.add_argument("--iterate", action="store_true", help="Ejecuta refinamiento iterativo eliminando items marcados para revision.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximo de iteraciones de refinamiento.")
    parser.add_argument("--min-primary-loading", type=float, default=0.40)
    parser.add_argument("--max-cross-loading", type=float, default=0.30)
    parser.add_argument(
        "--max-cross-loading-ratio",
        type=float,
        default=0.75,
        help="Relacion maxima secundaria/primaria antes de marcar carga cruzada.",
    )
    parser.add_argument("--min-loading-gap", type=float, default=0.15)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
