"""Экспорт графиков предсказаний AutoML в assets/chapter2/.

    uv run python src/scripts/export_prediction_charts.py
"""

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go  # type: ignore[import-untyped]
import requests  # type: ignore[import-untyped]

API_URL = "http://localhost:8000"
OUT_DIR = Path(__file__).resolve().parents[2] / "assets" / "chapter2"

TRAIN_COLOR = "rgba(99, 149, 230, 0.12)"
VAL_COLOR = "rgba(255, 180, 50, 0.15)"
TEST_COLOR = "rgba(229, 100, 100, 0.15)"

MODEL_COLORS = {
    "seasonal_naive": "#4CAF50",
    "catboost": "#FF6B6B",
    "catboost_per_panel": "#FF9999",
    "catboost_clustered": "#FF6ED8",
    "autoarima": "#FFB347",
    "autoets": "#87CEEB",
    "autotheta": "#F7C948",
    "mstl": "#9B59B6",
    "chronos": "#1ABC9C",
    "ts2vec": "#E67E22",
    "ts2vec_clustered": "#D35400",
    "patchtst": "#5DADE2",
}

MODEL_LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "catboost": "CatBoost",
    "catboost_per_panel": "CatBoost per-panel",
    "catboost_clustered": "CatBoost clustered",
    "autoarima": "AutoARIMA",
    "autoets": "AutoETS",
    "autotheta": "AutoTheta",
    "mstl": "MSTL",
    "chronos": "Chronos-2",
    "ts2vec": "TS2Vec + CatBoost",
    "ts2vec_clustered": "TS2Vec clustered",
    "patchtst": "PatchTST",
}


def get_projects() -> list[dict]:
    resp = requests.get(f"{API_URL}/projects")
    resp.raise_for_status()
    return resp.json()


def get_project_result(project_id: str) -> dict:
    resp = requests.get(f"{API_URL}/projects/{project_id}/automl_result")
    resp.raise_for_status()
    return resp.json()


def get_panel_data(project_id: str, panel_ids: list[str]) -> list[dict]:
    resp = requests.get(
        f"{API_URL}/projects/{project_id}/panels",
        params={"ids": ",".join(panel_ids)},
    )
    resp.raise_for_status()
    return resp.json()


def get_predictions(project_id: str, panel_ids: list[str], models: list[str]) -> dict:
    resp = requests.get(
        f"{API_URL}/projects/{project_id}/automl_predictions",
        params={"panel_ids": ",".join(panel_ids), "models": ",".join(models)},
    )
    resp.raise_for_status()
    return resp.json()


def build_chart(
    panel_id: str,
    dates: list[str],
    values: list[float],
    predictions: dict[str, list[dict]],
    val_periods: int,
    test_periods: int,
    show_models: list[str],
) -> go.Figure:
    n = len(dates)
    train_end = n - val_periods - test_periods
    val_end = n - test_periods

    fig = go.Figure()

    if train_end > 0:
        fig.add_vrect(x0=dates[0], x1=dates[train_end - 1], fillcolor=TRAIN_COLOR, line_width=0)
    if val_periods > 0 and val_end > train_end:
        fig.add_vrect(x0=dates[train_end], x1=dates[val_end - 1], fillcolor=VAL_COLOR, line_width=0)
    if test_periods > 0:
        fig.add_vrect(x0=dates[val_end], x1=dates[-1], fillcolor=TEST_COLOR, line_width=0)

    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines", name="Фактическое",
        line=dict(color="#7C6AF7", width=2),
    ))

    dashes = ["dash", "dot", "dashdot"]
    for i, model in enumerate(show_models):
        per_panel: dict = predictions.get(model) or {}  # type: ignore[assignment]
        model_preds = per_panel.get(panel_id, []) if isinstance(per_panel, dict) else []
        if not model_preds:
            continue
        pred_dates = [p["date"] for p in model_preds]
        pred_values = [p["y_pred"] for p in model_preds]
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_values, mode="lines",
            name=MODEL_LABELS.get(model, model),
            line=dict(
                color=MODEL_COLORS.get(model, "#AAAAAA"),
                width=1.5,
                dash=dashes[i % len(dashes)],
            ),
        ))

    fig.update_layout(
        title=f"ID: {panel_id}",
        width=1400,
        height=450,
        margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#FAFAFA", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Продажи"),
    )
    return fig


def process_project(project: dict, dataset_name: str, top_n: int = 3, show_models: list[str] | None = None) -> None:
    project_id = str(project["id"])
    full = get_project_result(project_id)
    result = full.get("result") or {}
    automl = result.get("automl")
    split = result.get("split") or {}
    if not automl:
        print("  Нет результатов AutoML, пропускаю")
        return

    best_model = automl["best_model"]
    metric = automl["selection_metric"]
    model_results = automl["model_results"]
    val_periods = split.get("val_periods", 0)
    test_periods = split.get("test_periods", 0)

    best_mr = next(mr for mr in model_results if mr["name"] == best_model)
    panel_metrics = best_mr.get("panel_metrics", [])
    sorted_panels = sorted(panel_metrics, key=lambda p: p.get("test") or float("inf"))
    top_panels = [str(p["panel_id"]) for p in sorted_panels[:top_n]]
    bottom_panels = [str(p["panel_id"]) for p in sorted_panels[-top_n:]]
    all_panels = list(dict.fromkeys(top_panels + bottom_panels))

    sorted_mrs = sorted(model_results, key=lambda mr: mr.get(f"val_{metric}", float("inf")))
    if show_models:
        chart_models = show_models
    else:
        chart_models = [mr["name"] for mr in sorted_mrs[:3]]

    print(f"  Лучшая модель: {best_model}")
    print(f"  Модели на графиках: {chart_models}")
    print(f"  Топ панели: {top_panels}")
    print(f"  Худшие панели: {bottom_panels}")

    panel_data = get_panel_data(project_id, all_panels)
    predictions = get_predictions(project_id, all_panels, chart_models)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rank, panel_id in enumerate(top_panels, 1):
        series = next((p for p in panel_data if str(p["panel_id"]) == panel_id), None)
        if not series:
            continue
        fig = build_chart(panel_id, series["dates"], series["values"], predictions, val_periods, test_periods, chart_models)
        path = OUT_DIR / f"{dataset_name}_top_{rank}.png"
        fig.write_image(str(path), scale=2)
        print(f"  Сохранено: {path.name}")

    for rank, panel_id in enumerate(bottom_panels, 1):
        series = next((p for p in panel_data if str(p["panel_id"]) == panel_id), None)
        if not series:
            continue
        fig = build_chart(panel_id, series["dates"], series["values"], predictions, val_periods, test_periods, chart_models)
        path = OUT_DIR / f"{dataset_name}_bottom_{rank}.png"
        fig.write_image(str(path), scale=2)
        print(f"  Сохранено: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт графиков предсказаний для ВКР")
    parser.add_argument("--dataset", help="Имя датасета (wb/kaggle), по умолчанию все")
    parser.add_argument("--models", nargs="*", help="Модели для отображения (по умолчанию топ-3)")
    args = parser.parse_args()

    projects = get_projects()
    print(f"Найдено проектов: {len(projects)}\n")

    dataset_map = {}
    for p in projects:
        name = (p.get("name") or "").lower()
        if "mirror" in name or "wb" in name or "зеркал" in name:
            dataset_map["wb"] = p
        elif "kaggle" in name or "store" in name or "demand" in name:
            dataset_map["kaggle"] = p

    if not dataset_map:
        print("Не найдены проекты WB/Kaggle. Доступные проекты:")
        for p in projects:
            status = p.get("status", "?")
            print(f"  {p.get('name', '?')} ({status}) — {p.get('project_id', '?')}")
        sys.exit(1)

    for ds_name, project in dataset_map.items():
        if args.dataset and ds_name != args.dataset:
            continue
        print(f"=== {ds_name.upper()}: {project.get('name')} ===")
        process_project(project, ds_name, show_models=args.models)
        print()

    print(f"Все графики в {OUT_DIR}/")


if __name__ == "__main__":
    main()
