import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.api_client import get_automl_progress, get_job, get_panels_data, run_automl
from app.state import get_current_project, set_page

_ALL_MODELS = ["seasonal_naive", "catboost", "autoarima", "autoets", "autotheta"]
_MODEL_LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "catboost": "CatBoost",
    "autoarima": "AutoARIMA",
    "autoets": "AutoETS",
    "autotheta": "AutoTheta",
}
_METRICS = ["mape", "rmse", "mae"]

_TRAIN_COLOR = "rgba(99, 149, 230, 0.15)"
_VAL_COLOR = "rgba(255, 180, 50, 0.2)"
_TEST_COLOR = "rgba(229, 100, 100, 0.2)"


def _render_config() -> tuple[list[str], str, bool]:
    """Конфигурация AutoML. Возвращает (models, metric, use_hyperopt)."""
    st.markdown("**Модели**")
    cols = st.columns(len(_ALL_MODELS))
    selected = []
    defaults = {"seasonal_naive", "catboost"}
    for i, model in enumerate(_ALL_MODELS):
        with cols[i]:
            if st.checkbox(_MODEL_LABELS[model], value=model in defaults, key=f"model_{model}"):
                selected.append(model)

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Метрика отбора", _METRICS, key="automl_metric")
    with col2:
        use_hyperopt = st.toggle("Hyperopt (Optuna)", value=False, key="automl_hyperopt")
        if use_hyperopt:
            st.caption("Значительно увеличивает время обучения CatBoost")

    return selected, metric, use_hyperopt


def _render_progress(project_id: str, job_id: str, models: list[str]) -> bool:
    """Отображает прогресс AutoML. Возвращает True если завершено."""
    try:
        job = get_job(job_id)
        events = get_automl_progress(project_id, job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        return False

    done_models = {e["model"] for e in events if e.get("type") == "model_done"}
    current_model = next((e["model"] for e in reversed(events) if e.get("type") == "model_start"), None)
    n_done = len(done_models)
    n_total = len(models)
    pct = int(n_done / n_total * 100) if n_total else 0

    st.progress(pct, text=f"{n_done} / {n_total} моделей")
    for model in models:
        label = _MODEL_LABELS.get(model, model)
        if model in done_models:
            metric_event = next((e for e in events if e.get("type") == "model_done" and e.get("model") == model), {})
            metric_val = next((v for k, v in metric_event.items() if k.startswith("val_")), "")
            st.markdown(f"✅ {label}" + (f" — val: {metric_val}" if metric_val else ""))
        elif model == current_model:
            st.markdown(f"⏳ {label}")
        else:
            st.markdown(f"⬜ {label}")

    if job["status"] == "done":
        return True
    if job["status"] == "failed":
        st.error("AutoML завершился с ошибкой")
        del st.session_state.automl_job_id
        return False

    time.sleep(2)
    st.rerun()
    return False


def _render_results(project: dict, automl_result: dict, split_result: dict) -> None:
    """Отображает результаты AutoML."""
    best_model = automl_result["best_model"]
    metric = automl_result["selection_metric"]
    model_results = automl_result["model_results"]
    val_periods = split_result.get("val_periods", 0)
    test_periods = split_result.get("test_periods", 0)

    st.success(f"Лучшая модель: **{_MODEL_LABELS.get(best_model, best_model)}**")

    # Сводная таблица моделей
    st.markdown("**Сравнение моделей**")
    summary_rows = []
    for mr in model_results:
        summary_rows.append({
            "Модель": _MODEL_LABELS.get(mr["name"], mr["name"]),
            f"Val {metric.upper()}": round(mr.get(f"val_{metric}", float("inf")), 4),
            f"Test {metric.upper()}": round(mr.get(f"test_{metric}", float("inf")), 4),
            "Лучшая": "⭐" if mr["name"] == best_model else "",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Таблица по панелям для лучшей модели
    st.markdown("**Результаты по панелям (лучшая модель)**")
    best_mr = next(mr for mr in model_results if mr["name"] == best_model)
    panel_rows = [
        {
            "Panel ID": p["panel_id"],
            f"Val {metric.upper()}": round(p["val"], 4) if p["val"] is not None else None,
            f"Test {metric.upper()}": round(p["test"], 4) if p["test"] is not None else None,
        }
        for p in best_mr.get("panel_metrics", [])
    ]
    panel_df = pd.DataFrame(panel_rows).sort_values(f"Test {metric.upper()}")

    selection = st.dataframe(
        panel_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
    )

    selected_rows = selection.selection.get("rows", [])
    if selected_rows:
        panel_id = str(panel_df.iloc[selected_rows[0]]["Panel ID"])
        _render_panel_chart(str(project.get("project_id", "")), panel_id, val_periods, test_periods)

    st.divider()

    # Лучшие / худшие панели
    col_best, col_worst = st.columns(2)
    top3 = panel_df.head(3)["Panel ID"].tolist()
    bot3 = panel_df.tail(3)["Panel ID"].tolist()

    with col_best:
        st.markdown("**Топ-3 (лучший test)**")
        _render_mini_charts(str(project.get("project_id", "")), top3, val_periods, test_periods, key_prefix="top")
    with col_worst:
        st.markdown("**Антитоп-3 (худший test)**")
        _render_mini_charts(str(project.get("project_id", "")), bot3, val_periods, test_periods, key_prefix="bot")

    st.divider()
    if st.button("→ Прогноз", type="primary"):
        set_page("forecast")
        st.rerun()


def _render_panel_chart(project_id: str, panel_id: str, val_periods: int, test_periods: int) -> None:
    try:
        data = get_panels_data(project_id, [panel_id])
    except Exception:
        return
    if not data:
        return
    series = data[0]
    dates, values = series["dates"], series["values"]
    n = len(dates)
    train_end = n - val_periods - test_periods
    val_end = n - test_periods

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", line=dict(color="#7C6AF7", width=1.5)))
    if train_end > 0:
        fig.add_vrect(x0=dates[0], x1=dates[train_end] if train_end < n else dates[-1],
                      fillcolor=_TRAIN_COLOR, line_width=0, annotation_text="train", annotation_position="top left")
    if 0 < train_end < val_end:
        fig.add_vrect(x0=dates[train_end], x1=dates[val_end] if val_end < n else dates[-1],
                      fillcolor=_VAL_COLOR, line_width=0, annotation_text="val", annotation_position="top left")
    if val_end < n:
        fig.add_vrect(x0=dates[val_end], x1=dates[-1],
                      fillcolor=_TEST_COLOR, line_width=0, annotation_text="test", annotation_position="top left")
    fig.update_layout(
        title=f"Панель {panel_id}", height=300, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#333"), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"panel_chart_{panel_id}")


def _render_mini_charts(project_id: str, panel_ids: list[str], val_periods: int, test_periods: int, key_prefix: str = "") -> None:
    if not panel_ids:
        return
    try:
        data = get_panels_data(project_id, panel_ids)
    except Exception:
        return
    for series in data:
        dates, values = series["dates"], series["values"]
        n = len(dates)
        train_end = n - val_periods - test_periods
        val_end = n - test_periods
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", line=dict(color="#7C6AF7", width=1)))
        if train_end > 0:
            fig.add_vrect(x0=dates[0], x1=dates[train_end] if train_end < n else dates[-1],
                          fillcolor=_TRAIN_COLOR, line_width=0)
        if 0 < train_end < val_end:
            fig.add_vrect(x0=dates[train_end], x1=dates[val_end] if val_end < n else dates[-1],
                          fillcolor=_VAL_COLOR, line_width=0)
        if val_end < n:
            fig.add_vrect(x0=dates[val_end], x1=dates[-1], fillcolor=_TEST_COLOR, line_width=0)
        fig.update_layout(
            title=f"ID: {series['panel_id']}", height=180, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"mini_{key_prefix}_{series['panel_id']}")


def render() -> None:
    """Отображает экран AutoML."""
    project = get_current_project()
    if project is None:
        st.warning("Проект не выбран")
        return

    result = project.get("result") or {}
    split_result = result.get("split", {})
    automl_result = result.get("automl")
    project_id = str(project.get("project_id", ""))

    st.title("Моделирование")

    # Если уже есть результат automl — показываем результаты
    if automl_result and not st.session_state.get("automl_job_id"):
        _render_results(project, automl_result, split_result)
        return

    # Если идёт polling — показываем прогресс
    if "automl_job_id" in st.session_state:
        job_id = st.session_state.automl_job_id
        models = st.session_state.get("automl_models", ["seasonal_naive", "catboost"])
        st.markdown("**Обучение моделей...**")
        done = _render_progress(project_id, job_id, models)
        if done:
            try:
                job = get_job(job_id)
                new_result = {**result, **job["result"]}
                updated_project = {**project, "result": new_result}
                st.session_state.current_project = {**updated_project, "project_id": project_id}
            except Exception:
                pass
            del st.session_state.automl_job_id
            st.rerun()
        return

    # Конфигурация и кнопка запуска
    selected_models, metric, use_hyperopt = _render_config()

    st.divider()
    if not selected_models:
        st.warning("Выберите хотя бы одну модель")
        return

    if st.button("▶ Запустить AutoML", type="primary", use_container_width=True):
        with st.spinner("Запускаю..."):
            try:
                job = run_automl(project_id, selected_models, metric, use_hyperopt)
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.automl_job_id = str(job["id"])
        st.session_state.automl_models = selected_models
        st.rerun()
