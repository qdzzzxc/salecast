import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.api_client import (
    get_cv_progress,
    get_cv_result,
    get_forecast_csv_bytes,
    get_forecast_data,
    get_forecast_progress,
    get_job,
    get_panels_data,
    run_cv,
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
        horizon = st.number_input(
            "Горизонт (точек)", min_value=1, value=6, step=1, key="forecast_horizon"
        )

    return model_name, int(horizon), None


_FORECAST_STEPS = [
    ("loading", "Загрузка данных"),
    ("training", "Обучение модели"),
    ("forecasting", "Построение прогноза"),
    ("saving", "Сохранение"),
]


def _render_progress(project_id: str, job_id: str) -> bool:
    """Отображает прогресс прогноза. Возвращает True когда завершено."""
    try:
        job = get_job(job_id)
        events = get_forecast_progress(project_id, job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        return False

    status = job.get("status", "")

    if status == "failed":
        st.error("Прогноз завершился с ошибкой")
        del st.session_state.forecast_job_id
        return False

    # Определяем какие шаги начались
    started_steps = {e["step"] for e in events if e.get("type") == "step_start"}
    is_completed = any(e.get("type") == "completed" for e in events)

    n_done = sum(
        1
        for i, (step_key, _) in enumerate(_FORECAST_STEPS)
        if step_key in started_steps
        and (
            is_completed
            or any(
                _FORECAST_STEPS[j][0] in started_steps for j in range(i + 1, len(_FORECAST_STEPS))
            )
        )
    )
    n_total = len(_FORECAST_STEPS)
    pct = int(n_done / n_total * 100) if n_total else 0

    # Прогресс для шага forecasting (CatBoost)
    forecast_steps = [e for e in events if e.get("type") == "forecast_step"]
    last_fc = forecast_steps[-1] if forecast_steps else None

    st.progress(pct if not is_completed else 100, text=f"{n_done} / {n_total} шагов")

    for step_key, step_label in _FORECAST_STEPS:
        if is_completed or (
            step_key in started_steps
            and any(
                _FORECAST_STEPS[j][0] in started_steps
                for j in range(
                    _FORECAST_STEPS.index((step_key, step_label)) + 1, len(_FORECAST_STEPS)
                )
            )
        ):
            st.markdown(f"✅ {step_label}")
        elif step_key in started_steps:
            if step_key == "forecasting" and last_fc:
                step_i = last_fc.get("step_i", "?")
                total = last_fc.get("total", "?")
                fc_pct = (
                    int(int(step_i) / int(total) * 100)
                    if str(step_i).isdigit() and str(total).isdigit()
                    else 0
                )
                st.progress(fc_pct, text=f"⏳ {step_label} (шаг {step_i}/{total})")
            else:
                st.markdown(f"⏳ {step_label}")
        else:
            st.markdown(f"⬜ {step_label}")

    if status == "done":
        return True

    time.sleep(2)
    st.rerun()
    return False


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
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode="lines",
            name="История",
            line=dict(color=_HISTORY_COLOR, width=1.5),
        )
    )

    # Прогноз
    if fc_dates:
        # Связать линию истории с прогнозом
        fig.add_trace(
            go.Scatter(
                x=[hist_dates[-1]] + fc_dates,
                y=[hist_values[-1]] + fc_values,
                mode="lines+markers",
                name="Прогноз",
                line=dict(color=_FORECAST_COLOR, width=2, dash="dot"),
                marker=dict(size=6),
            )
        )

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


_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
)


def _render_cv_progress(project_id: str, job_id: str, n_folds: int) -> bool:
    """Отображает прогресс кросс-валидации. Возвращает True когда завершено."""
    try:
        job = get_job(job_id)
        events = get_cv_progress(project_id, job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        return False

    status = job.get("status", "")

    if status == "failed":
        st.error("Кросс-валидация завершилась с ошибкой")
        del st.session_state.cv_job_id
        return False

    done_folds = [e for e in events if e.get("type") == "fold_done"]
    is_completed = any(e.get("type") == "completed" for e in events)

    pct = len(done_folds) / n_folds if n_folds else 0
    st.progress(
        1.0 if is_completed else pct,
        text=f"Fold {len(done_folds)} / {n_folds}",
    )

    for e in done_folds:
        fold = e.get("fold", "?")
        mape = e.get("mape", "?")
        rmse = e.get("rmse", "?")
        st.markdown(f"✅ Fold {fold}: MAPE = {mape}%, RMSE = {rmse}")

    # Текущий fold
    started = [e for e in events if e.get("type") == "fold_start"]
    if started and not is_completed and len(started) > len(done_folds):
        current = started[-1]
        st.markdown(f"⏳ Fold {current.get('fold', '?')}/{n_folds}...")

    if status == "done":
        return True

    time.sleep(2)
    st.rerun()
    return False


def _render_cv_results(project_id: str) -> None:
    """Отображает результаты кросс-валидации."""
    cache_key = f"cv_result_{project_id}"
    if cache_key not in st.session_state:
        with st.spinner("Загрузка результатов CV..."):
            try:
                st.session_state[cache_key] = get_cv_result(project_id)
            except Exception as e:
                st.warning(f"Не удалось загрузить результаты CV: {e}")
                return

    cv_data = st.session_state[cache_key]
    summary = cv_data.get("summary", {})
    folds = cv_data.get("folds", [])
    panel_metrics = cv_data.get("panel_metrics", [])

    # Метрики summary
    col1, col2, col3 = st.columns(3)
    mean_mape = summary.get("mean_mape")
    std_mape = summary.get("std_mape")
    mean_rmse = summary.get("mean_rmse")
    std_rmse = summary.get("std_rmse")
    mean_mae = summary.get("mean_mae")
    std_mae = summary.get("std_mae")

    if mean_mape is not None:
        col1.metric("MAPE", f"{mean_mape:.2f}%", delta=f"± {std_mape:.2f}" if std_mape else None)
    if mean_rmse is not None:
        col2.metric("RMSE", f"{mean_rmse:.2f}", delta=f"± {std_rmse:.2f}" if std_rmse else None)
    if mean_mae is not None:
        col3.metric("MAE", f"{mean_mae:.2f}", delta=f"± {std_mae:.2f}" if std_mae else None)

    # Таблица фолдов
    if folds:
        fold_df = pd.DataFrame(folds)
        display_cols = [c for c in ["fold", "mape", "rmse", "mae", "train_rows", "test_rows"]
                        if c in fold_df.columns]
        st.dataframe(
            fold_df[display_cols].rename(columns={
                "fold": "Fold", "mape": "MAPE", "rmse": "RMSE",
                "mae": "MAE", "train_rows": "Train", "test_rows": "Test",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Box plot MAPE по панелям
    if panel_metrics:
        pm_df = pd.DataFrame(panel_metrics)
        if "mape" in pm_df.columns and len(pm_df) > 0:
            import plotly.express as px

            fig = px.box(
                pm_df,
                x="fold",
                y="mape",
                title="Разброс MAPE по панелям",
                labels={"fold": "Fold", "mape": "MAPE"},
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                **_DARK_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="cv_box_plot")


def _render_cv_section(
    project: dict,
    project_id: str,
    automl_result: dict,
    result: dict,
) -> None:
    """Секция кросс-валидации на странице Forecast."""
    st.divider()
    st.subheader("Кросс-валидация")
    st.caption("Temporal CV с expanding window — проверка стабильности модели на нескольких фолдах.")

    # Если есть результат CV — показываем
    cv_result = result.get("cross_validation")
    if cv_result and not st.session_state.get("cv_job_id"):
        model_type = cv_result.get("model_type", "?")
        n_folds = cv_result.get("n_folds", "?")
        st.success(f"Модель: **{model_type}** · Фолдов: **{n_folds}**")
        _render_cv_results(project_id)

        if st.button("🔄 Перезапустить CV", key="cv_rerun"):
            st.session_state.pop(f"cv_result_{project_id}", None)
            st.session_state["cv_show_form"] = True
            st.rerun()
        if not st.session_state.get("cv_show_form"):
            return

    # Polling
    if "cv_job_id" in st.session_state:
        job_id = st.session_state.cv_job_id
        n_folds = st.session_state.get("cv_n_folds", 5)
        st.markdown("**Кросс-валидация...**")
        done = _render_cv_progress(project_id, job_id, n_folds)
        if "cv_job_id" not in st.session_state:
            st.rerun()
            return
        if done:
            try:
                job = get_job(job_id)
                new_result = {**result, **job["result"]}
                updated_project = {**project, "result": new_result}
                st.session_state.current_project = {**updated_project, "project_id": project_id}
                st.session_state.pop(f"cv_result_{project_id}", None)
            except Exception:
                pass
            del st.session_state.cv_job_id
            st.session_state.pop("cv_show_form", None)
            st.rerun()
        return

    # Форма запуска
    model_results = automl_result.get("model_results", [])
    best_model = automl_result.get("best_model", "")
    all_models = [mr["name"] for mr in model_results]

    col1, col2 = st.columns(2)
    with col1:
        cv_model = st.selectbox(
            "Модель для CV",
            options=all_models or [best_model],
            index=all_models.index(best_model) if best_model in all_models else 0,
            key="cv_model_select",
        )
    with col2:
        n_folds = st.number_input(
            "Количество фолдов",
            min_value=2, max_value=10, value=5, step=1,
            key="cv_n_folds_input",
        )

    if st.button("▶ Запустить кросс-валидацию", type="primary", use_container_width=True):
        with st.spinner("Запускаю..."):
            try:
                job = run_cv(project_id, cv_model, int(n_folds))
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.cv_job_id = str(job["id"])
        st.session_state.cv_n_folds = int(n_folds)
        st.session_state.pop("cv_show_form", None)
        st.rerun()


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
            # Показываем CV секцию после прогноза
            _render_cv_section(project, project_id, automl_result, result)
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
