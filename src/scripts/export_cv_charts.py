"""Экспорт результатов кросс-валидации в assets/chapter2/.

    uv run python src/scripts/export_cv_charts.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import requests  # type: ignore[import-untyped]

API_URL = "http://localhost:8000"
OUT_DIR = Path(__file__).resolve().parents[2] / "assets" / "chapter2"

DARK_LAYOUT = dict(
    paper_bgcolor="#1E1E2E",
    plot_bgcolor="#1E1E2E",
    font=dict(color="#FAFAFA", size=14),
)


def get_projects() -> list[dict]:
    resp = requests.get(f"{API_URL}/projects")
    resp.raise_for_status()
    return resp.json()


def get_cv_result(project_id: str) -> dict:
    resp = requests.get(f"{API_URL}/projects/{project_id}/cv_result")
    resp.raise_for_status()
    return resp.json()


def get_project_result(project_id: str) -> dict:
    resp = requests.get(f"{API_URL}/projects/{project_id}/automl_result")
    resp.raise_for_status()
    return resp.json()


def export_cv(project: dict, dataset_name: str) -> None:
    project_id = str(project["id"])

    full = get_project_result(project_id)
    cv_meta = (full.get("result") or {}).get("cross_validation", {})
    model_type = cv_meta.get("model_type", "?")
    n_folds = cv_meta.get("n_folds", "?")

    try:
        cv = get_cv_result(project_id)
    except requests.HTTPError:
        print("  Нет результатов CV, пропускаю")
        return

    summary = cv.get("summary", {})
    folds = cv.get("folds", [])
    panel_metrics = cv.get("panel_metrics", [])

    print(f"  Модель: {model_type}, фолдов: {n_folds}")
    print(f"  Mean MAPE: {summary.get('mean_mape', 0) * 100:.2f}% ± {summary.get('std_mape', 0) * 100:.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if folds:
        fold_df = pd.DataFrame(folds)
        if "mape" in fold_df.columns:
            fold_df["mape_pct"] = fold_df["mape"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {f}" for f in fold_df["fold"]],
            y=fold_df["mape_pct"],
            marker_color="#7C6AF7",
            text=[f"{v:.2f}%" for v in fold_df["mape_pct"]],
            textposition="outside",
        ))
        mean_mape_pct = summary.get("mean_mape", 0) * 100
        fig.add_hline(y=mean_mape_pct, line_dash="dash", line_color="#FF6B6B",
                      annotation_text=f"Среднее: {mean_mape_pct:.2f}%")
        fig.update_layout(
            title=f"MAPE по фолдам ({model_type})",
            yaxis_title="MAPE, %",
            width=800, height=450,
            margin=dict(l=60, r=30, t=50, b=50),
            **DARK_LAYOUT,
        )
        path = OUT_DIR / f"{dataset_name}_cv_folds.png"
        fig.write_image(str(path), scale=2)
        print(f"  Сохранено: {path.name}")

    if panel_metrics:
        pm_df = pd.DataFrame(panel_metrics)
        if "mape" in pm_df.columns:
            pm_df["mape"] = pm_df["mape"] * 100
            fig = px.box(
                pm_df, x="fold", y="mape",
                labels={"fold": "Fold", "mape": "MAPE, %"},
                title=f"Разброс MAPE по панелям ({model_type})",
            )
            fig.update_layout(
                width=800, height=450,
                margin=dict(l=60, r=30, t=50, b=50),
                **DARK_LAYOUT,
            )
            path = OUT_DIR / f"{dataset_name}_cv_boxplot.png"
            fig.write_image(str(path), scale=2)
            print(f"  Сохранено: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт CV результатов для ВКР")
    parser.add_argument("--dataset", help="wb/kaggle, по умолчанию все")
    args = parser.parse_args()

    projects = get_projects()
    print(f"Найдено проектов: {len(projects)}\n")

    dataset_map: dict[str, dict] = {}
    for p in projects:
        name = (p.get("name") or "").lower()
        if "mirror" in name or "wb" in name:
            dataset_map["wb"] = p
        elif "kaggle" in name or "store" in name or "demand" in name:
            dataset_map["kaggle"] = p

    if not dataset_map:
        print("Не найдены проекты WB/Kaggle")
        sys.exit(1)

    for ds_name, project in dataset_map.items():
        if args.dataset and ds_name != args.dataset:
            continue
        print(f"=== {ds_name.upper()}: {project.get('name')} ===")
        export_cv(project, ds_name)
        print()

    print(f"Все графики в {OUT_DIR}/")


if __name__ == "__main__":
    main()
