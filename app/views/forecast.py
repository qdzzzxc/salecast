import time

import plotly.graph_objects as go
import streamlit as st

from app.api_client import (
    get_forecast_csv_bytes,
    get_forecast_data,
    get_job,
    get_panels_data,
    run_forecast,
)
from app.state import get_current_project

_HISTORY_COLOR = "#7C6AF7"
_FORECAST_COLOR = "#FF6B6B"


def _render_config(automl_result: dict) -> tuple[str, int, list[str] | None]:
    """Конфигурация прогноза. Возвращает (model_name, horizon, panel_ids | None)."""
    best_model = automl_result.get("best_model", "seasonal_naive")
    model_results = automl_result.get("model_results", [])
    all_models = [mr["name"] for mr in model_results]

    _model_labels = {
        "seasonal_naive": "Seasonal Naive",
        "catboost": "CatBoost",
        "autoarima": "AutoARIMA",
        "autoets": "AutoETS",
        "autotheta": "AutoTheta",
    }

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "Модель",
            options=all_models or [best_model],
            index=all_models.index(best_model) if best_model in all_models else 0,
            format_func=lambda x: _model_labels.get(x, x),
            key="forecast_model",
        )
    with col2:
        horizon = st.number_input("Горизонт (точек)", min_value=1, value=6, step=1, key="forecast_horizon")

    return model_name, int(horizon), None


def _render_progress(project_id: str, job_id: str) -> bool:
    """Опрашивает статус задачи прогноза. Возвращает True когда завершено."""
    try:
        job = get_job(job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        return False

    status = job.get("status", "")
    steps = job.get("steps") or []
    last_step = steps[-1]["message"] if steps else "Инициализация..."

    if status in ("pending", "running"):
        st.info(f"⏳ {last_step}")
        time.sleep(2)
        st.rerun()
        return False

    if status == "failed":
        st.error("Прогноз завершился с ошибкой")
        del st.session_state.forecast_job_id
        return False

    return True


def _render_results(project_id: str, forecast_result: dict, split_result: dict) -> None:
    """Отображает результаты прогноза."""
    model = forecast_result.get("model", "?")
    horizon = forecast_result.get("horizon", "?")
    panel_count = forecast_result.get("panel_count", "?")

    _model_labels = {
        "seasonal_naive": "Seasonal Naive",
        "catboost": "CatBoost",
        "autoarima": "AutoARIMA",
        "autoets": "AutoETS",
        "autotheta": "AutoTheta",
    }

    col_info, col_dl = st.columns([4, 1])
    with col_info:
        st.success(
            f"Модель: **{_model_labels.get(model, model)}** · "
            f"Горизонт: **{horizon}** · "
            f"Панелей: **{panel_count}**"
        )
    with col_dl:
        try:
            csv_bytes = get_forecast_csv_bytes(project_id)
            st.download_button(
                "⬇ CSV",
                data=csv_bytes,
                file_name=f"forecast_{project_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            pass

    # Загружаем данные прогноза — кешируем
    cache_key = f"forecast_data_{project_id}"
    if cache_key not in st.session_state:
        with st.spinner("Загрузка прогноза..."):
            try:
                st.session_state[cache_key] = get_forecast_data(project_id)
            except Exception as e:
                st.error(f"Ошибка загрузки прогноза: {e}")
                return

    forecast_data: dict = st.session_state.get(cache_key, {})
    all_panel_ids = sorted(forecast_data.keys())

    if not all_panel_ids:
        st.warning("Нет данных прогноза")
        return

    # Таблица панелей — кликом выбираем панель для просмотра
    st.markdown("**Панели** — выберите строку для просмотра графика")

    import pandas as pd
    panel_df = pd.DataFrame({"Panel ID": all_panel_ids})
    selection = st.dataframe(
        panel_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="forecast_panel_table",
    )
    selected_rows = selection.selection.get("rows", [])
    if selected_rows:
        panel_id = all_panel_ids[selected_rows[0]]
        _render_panel_chart(project_id, panel_id, forecast_data.get(panel_id, []))


def _render_panel_chart(project_id: str, panel_id: str, forecast_points: list[dict]) -> None:
    """Рисует историю + прогноз для одной панели."""
    try:
        data = get_panels_data(project_id, [panel_id])
    except Exception:
        st.error("Не удалось загрузить историю")
        return
    if not data:
        return

    series = data[0]
    hist_dates = series["dates"]
    hist_values = series["values"]

    fc_dates = [p["date"] for p in forecast_points]
    fc_values = [p["forecast"] for p in forecast_points]

    fig = go.Figure()

    # Исторические данные
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_values,
        mode="lines", name="История",
        line=dict(color=_HISTORY_COLOR, width=1.5),
    ))

    # Прогноз
    if fc_dates:
        # Связать линию истории с прогнозом
        fig.add_trace(go.Scatter(
            x=[hist_dates[-1]] + fc_dates,
            y=[hist_values[-1]] + fc_values,
            mode="lines+markers", name="Прогноз",
            line=dict(color=_FORECAST_COLOR, width=2, dash="dot"),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=f"Панель {panel_id}",
        height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#333"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{panel_id}")


def render() -> None:
    """Отображает экран Прогноза."""
    project = get_current_project()
    if project is None:
        st.warning("Проект не выбран")
        return

    result = project.get("result") or {}
    automl_result = result.get("automl")
    project_id = str(project.get("project_id", ""))

    st.title("Прогноз")

    if not automl_result:
        st.info("Сначала завершите шаг Моделирования")
        return

    # Если уже есть готовый прогноз
    forecast_result = result.get("forecast")
    if forecast_result and not st.session_state.get("forecast_job_id"):
        _render_results(project_id, forecast_result, result.get("split", {}))
        st.divider()
        if st.button("↺ Построить новый прогноз", use_container_width=True):
            # Сбросить кеш и показать форму
            st.session_state.pop(f"forecast_data_{project_id}", None)
            st.session_state["forecast_show_form"] = True
            st.rerun()
        if not st.session_state.get("forecast_show_form"):
            return

    # Если идёт polling
    if "forecast_job_id" in st.session_state:
        job_id = st.session_state.forecast_job_id
        st.markdown("**Построение прогноза...**")
        done = _render_progress(project_id, job_id)
        if done:
            try:
                job = get_job(job_id)
                new_result = {**result, **job["result"]}
                updated_project = {**project, "result": new_result}
                st.session_state.current_project = {**updated_project, "project_id": project_id}
                # Сбросить кеш данных чтобы загрузить свежий прогноз
                st.session_state.pop(f"forecast_data_{project_id}", None)
            except Exception:
                pass
            del st.session_state.forecast_job_id
            st.session_state.pop("forecast_show_form", None)
            st.rerun()
        return

    # Форма запуска
    model_name, horizon, panel_ids = _render_config(automl_result)

    st.divider()
    if st.button("▶ Построить прогноз", type="primary", use_container_width=True):
        with st.spinner("Запускаю..."):
            try:
                job = run_forecast(project_id, model_name, horizon, panel_ids or [])
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.forecast_job_id = str(job["id"])
        st.session_state.pop("forecast_show_form", None)
        st.rerun()
