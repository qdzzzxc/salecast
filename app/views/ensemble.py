"""Экран ансамблирования — комбинирование предсказаний нескольких моделей."""

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from app.api_client import get_ensemble_progress, get_job, run_ensemble
from app.state import get_current_project

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
)

_MODEL_LABELS = {
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
    "ensemble": "Ансамбль",
}

_METHOD_LABELS = {
    "weighted_avg": "Взвешенное среднее",
    "best_per_panel": "Лучшая модель на панель",
}


def _render_progress(project_id: str, job_id: str) -> bool:
    """Отображает прогресс ансамбля. Возвращает True когда завершено."""
    try:
        job = get_job(job_id)
        events = get_ensemble_progress(project_id, job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        return False

    status = job.get("status", "")

    if status == "failed":
        st.error("Расчёт ансамбля завершился с ошибкой")
        del st.session_state.ensemble_job_id
        return False

    is_completed = any(e.get("type") == "completed" for e in events)
    is_computing = any(e.get("type") == "computing" for e in events)

    if is_completed:
        st.progress(100, text="Готово")
    elif is_computing:
        st.progress(66, text="Расчёт ансамбля...")
    else:
        st.progress(33, text="Загрузка данных...")

    if status == "done":
        return True

    time.sleep(2)
    st.rerun()
    return False


def _render_results(project_id: str, ensemble_info: dict) -> None:
    """Отображает результаты ансамбля."""
    method = ensemble_info.get("method", "?")
    models = ensemble_info.get("models", [])
    metrics = ensemble_info.get("metrics", {})
    comparison = ensemble_info.get("comparison", [])
    method_info = ensemble_info.get("method_info", {})

    method_label = _METHOD_LABELS.get(method, method)
    model_labels = [_MODEL_LABELS.get(m, m) for m in models]
    st.success(f"Метод: **{method_label}** · Модели: **{', '.join(model_labels)}**")

    # Метрики ансамбля
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})

    col1, col2, col3, col4 = st.columns(4)
    if val_metrics:
        col1.metric("Val MAPE", f"{val_metrics.get('mape', 0):.2f}%")
        col2.metric("Val RMSE", f"{val_metrics.get('rmse', 0):.2f}")
    if test_metrics:
        col3.metric("Test MAPE", f"{test_metrics.get('mape', 0):.2f}%")
        col4.metric("Test RMSE", f"{test_metrics.get('rmse', 0):.2f}")

    # Таблица сравнения
    if comparison:
        st.markdown("**Сравнение моделей**")
        comp_df = pd.DataFrame(comparison)
        # Ищем колонки с метриками
        metric_cols = [c for c in comp_df.columns if c.startswith("val_") or c.startswith("test_")]
        display_cols = ["name"] + metric_cols
        rename = {"name": "Модель"}
        for c in metric_cols:
            rename[c] = c.replace("_", " ").upper()
        comp_df["name"] = comp_df["name"].map(lambda x: _MODEL_LABELS.get(x, x))

        st.dataframe(
            comp_df[display_cols].rename(columns=rename),
            width="stretch",
            hide_index=True,
        )

    # Визуализация по методу
    if method == "weighted_avg":
        weights = method_info.get("weights", {})
        if weights:
            st.markdown("**Веса моделей**")
            w_df = pd.DataFrame(
                [{"Модель": _MODEL_LABELS.get(k, k), "Вес": v} for k, v in weights.items()]
            )
            fig = px.bar(
                w_df,
                x="Вес",
                y="Модель",
                orientation="h",
                color="Вес",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=max(200, len(weights) * 50), **_DARK_LAYOUT)
            st.plotly_chart(fig, width="stretch", key="ensemble_weights_chart")

    elif method == "best_per_panel":
        model_wins = method_info.get("model_wins", {})
        if model_wins:
            st.markdown("**Распределение панелей по моделям**")
            wins_df = pd.DataFrame(
                [{"Модель": _MODEL_LABELS.get(k, k), "Панелей": v} for k, v in model_wins.items()]
            )
            fig = px.pie(wins_df, values="Панелей", names="Модель")
            fig.update_layout(height=350, **_DARK_LAYOUT)
            st.plotly_chart(fig, width="stretch", key="ensemble_wins_chart")


def render() -> None:
    """Отображает экран ансамблирования."""
    project = get_current_project()
    if project is None:
        st.warning("Проект не выбран")
        return

    result = project.get("result") or {}
    automl_result = result.get("automl")
    project_id = str(project.get("project_id", ""))

    st.title("Ансамбль")
    st.caption("Комбинирование предсказаний нескольких моделей для улучшения качества.")

    if not automl_result:
        st.info("Сначала завершите шаг Моделирования")
        return

    # Если есть результат — показываем
    ensemble_result = result.get("ensemble")
    if ensemble_result and not st.session_state.get("ensemble_job_id"):
        _render_results(project_id, ensemble_result)

        if st.button("🔄 Пересчитать ансамбль", key="ensemble_rerun"):
            st.session_state["ensemble_show_form"] = True
            st.rerun()
        if not st.session_state.get("ensemble_show_form"):
            return

    # Polling
    if "ensemble_job_id" in st.session_state:
        job_id = st.session_state.ensemble_job_id
        st.markdown("**Расчёт ансамбля...**")
        done = _render_progress(project_id, job_id)
        if "ensemble_job_id" not in st.session_state:
            st.rerun()
            return
        if done:
            try:
                job = get_job(job_id)
                new_result = {**result, **job["result"]}
                updated_project = {**project, "result": new_result}
                st.session_state.current_project = {**updated_project, "project_id": project_id}
            except Exception:
                pass
            del st.session_state.ensemble_job_id
            st.session_state.pop("ensemble_show_form", None)
            st.rerun()
        return

    # Форма запуска
    model_results = automl_result.get("model_results", [])
    all_models = [mr["name"] for mr in model_results]

    if len(all_models) < 2:
        st.warning("Для ансамбля нужно минимум 2 обученные модели. Вернитесь к шагу Моделирования.")
        return

    selected_models = st.multiselect(
        "Модели для ансамбля",
        options=all_models,
        default=all_models,
        format_func=lambda x: _MODEL_LABELS.get(x, x),
        key="ensemble_model_select",
    )

    method = st.radio(
        "Метод ансамблирования",
        options=["weighted_avg", "best_per_panel"],
        format_func=lambda x: _METHOD_LABELS.get(x, x),
        key="ensemble_method",
        horizontal=True,
    )

    if method == "weighted_avg":
        st.caption(
            "Предсказания комбинируются через взвешенное среднее. "
            "Веса обратно пропорциональны ошибке на валидации — "
            "модель с меньшей ошибкой получает больший вес."
        )
    else:
        st.caption(
            "Для каждой панели используется модель с наименьшей ошибкой на валидации. "
            "Разные панели могут прогнозироваться разными моделями."
        )

    st.divider()

    can_run = len(selected_models) >= 2
    if not can_run:
        st.warning("Выберите минимум 2 модели")

    if st.button(
        "▶ Рассчитать ансамбль",
        type="primary",
        width="stretch",
        disabled=not can_run,
    ):
        with st.spinner("Запускаю..."):
            try:
                job = run_ensemble(project_id, selected_models, method)
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.ensemble_job_id = str(job["id"])
        st.session_state.pop("ensemble_show_form", None)
        st.rerun()
