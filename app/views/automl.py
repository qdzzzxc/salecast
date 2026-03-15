import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from app.api_client import (
    get_automl_predictions,
    get_automl_progress,
    get_job,
    get_panels_data,
    run_automl,
    skip_model,
)
from app.state import get_current_project

_ALL_MODELS = ["seasonal_naive", "catboost", "autoarima", "autoets", "autotheta"]

_FREQ_LABELS: dict[str, str] = {
    "D": "Дневная", "B": "Рабочие дни",
    "W": "Недельная",
    "MS": "Месячная", "ME": "Месячная", "M": "Месячная",
    "QS": "Квартальная", "Q": "Квартальная",
    "A": "Годовая", "AS": "Годовая",
}
_FREQ_SEASON: dict[str, int] = {"D": 7, "W": 52, "MS": 12, "QS": 4}
_MODEL_LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "catboost": "CatBoost",
    "autoarima": "AutoARIMA",
    "autoets": "AutoETS",
    "autotheta": "AutoTheta",
}
_MODEL_COLORS = {
    "seasonal_naive": "#4CAF50",
    "catboost": "#FF6B6B",
    "autoarima": "#FFB347",
    "autoets": "#87CEEB",
    "autotheta": "#F7C948",
}
_METRICS = ["mape", "rmse", "mae"]

_TRAIN_COLOR = "rgba(99, 149, 230, 0.15)"
_VAL_COLOR = "rgba(255, 180, 50, 0.2)"
_TEST_COLOR = "rgba(229, 100, 100, 0.2)"


def _load_yaml_config(uploaded) -> None:
    """Записывает параметры из загруженного YAML в session_state."""
    cfg = yaml.safe_load(uploaded.read())
    st.session_state["automl_metric"] = cfg.get("selection_metric", "mape")
    st.session_state["automl_hyperopt"] = cfg.get("use_hyperopt", False)
    st.session_state["automl_n_trials"] = cfg.get("n_trials", 30)
    for model in _ALL_MODELS:
        st.session_state[f"model_{model}"] = model in cfg.get("models", [])
    cb = cfg.get("catboost_params") or {}
    st.session_state["cb_iterations"] = cb.get("iterations", 1000)
    st.session_state["cb_lr"] = cb.get("learning_rate", 0.03)
    st.session_state["cb_depth"] = cb.get("depth", 6)
    st.session_state["autoarima_approx"] = cfg.get("autoarima_approximation", True)


def _render_config() -> dict:
    """Конфигурация AutoML. Возвращает словарь параметров для API."""
    with st.popover("Загрузить конфиг (YAML)"):
        uploaded = st.file_uploader("YAML файл", type=["yaml", "yml"], key="config_upload", label_visibility="collapsed")
        if uploaded:
            _load_yaml_config(uploaded)
            st.rerun()

    st.markdown("**Модели**")
    cols = st.columns(len(_ALL_MODELS))
    selected = []
    defaults = {"seasonal_naive", "catboost"}
    for i, model in enumerate(_ALL_MODELS):
        with cols[i]:
            if st.checkbox(_MODEL_LABELS[model], value=model in defaults, key=f"model_{model}"):
                selected.append(model)

    cb_iterations = 1000
    cb_lr = 0.03
    cb_depth = 6
    if "catboost" in selected:
        with st.expander("Настройки CatBoost"):
            cb_iterations = st.number_input(
                "iterations", min_value=50, max_value=5000, step=50,
                value=st.session_state.get("cb_iterations", 1000), key="cb_iterations",
            )
            cb_lr = st.number_input(
                "learning_rate", min_value=0.001, max_value=1.0, step=0.005, format="%.3f",
                value=st.session_state.get("cb_lr", 0.03), key="cb_lr",
            )
            cb_depth = st.number_input(
                "depth", min_value=2, max_value=12, step=1,
                value=st.session_state.get("cb_depth", 6), key="cb_depth",
            )
            st.caption("При hyperopt=True Optuna перезапишет эти параметры")

    autoarima_approx = True
    if "autoarima" in selected:
        with st.expander("Настройки AutoARIMA"):
            autoarima_approx = st.checkbox(
                "Быстрый режим (approximation)",
                value=st.session_state.get("autoarima_approx", True),
                key="autoarima_approx",
                help="approximation=True ускоряет подбор порядков ARIMA, рекомендуется для больших датасетов",
            )

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Метрика отбора", _METRICS, key="automl_metric")
    with col2:
        use_hyperopt = st.toggle("Hyperopt (Optuna)", value=False, key="automl_hyperopt")

    n_trials = 30
    if use_hyperopt:
        n_trials = st.number_input(
            "n_trials", min_value=5, max_value=500, step=5,
            value=st.session_state.get("automl_n_trials", 30), key="automl_n_trials",
        )
        st.caption("Значительно увеличивает время обучения CatBoost")

    return {
        "models": selected,
        "selection_metric": metric,
        "use_hyperopt": use_hyperopt,
        "n_trials": int(n_trials),
        "catboost_params": {"iterations": int(cb_iterations), "learning_rate": float(cb_lr), "depth": int(cb_depth)},
        "autoarima_approximation": bool(autoarima_approx),
    }


def _render_progress(project_id: str, job_id: str, models: list[str]) -> bool:
    """Отображает прогресс AutoML. Возвращает True если завершено."""
    try:
        job = get_job(job_id)
        events = get_automl_progress(project_id, job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        return False

    done_models = {e["model"] for e in events if e.get("type") == "model_done"}
    skipped_models = {e["model"] for e in events if e.get("type") in ("model_skipped", "model_timeout")}
    finished_models = done_models | skipped_models
    current_model = next(
        (e["model"] for e in reversed(events) if e.get("type") == "model_start" and e["model"] not in finished_models),
        None,
    )

    # Последнее progress-сообщение для текущей модели
    last_progress = next(
        (e for e in reversed(events) if e.get("type") == "model_progress" and e.get("model") == current_model),
        None,
    )

    n_done = len(finished_models)
    n_total = len(models)
    pct = int(n_done / n_total * 100) if n_total else 0

    st.progress(pct, text=f"{n_done} / {n_total} моделей")
    for model in models:
        label = _MODEL_LABELS.get(model, model)
        if model in done_models:
            metric_event = next((e for e in events if e.get("type") == "model_done" and e.get("model") == model), {})
            metric_val = next((v for k, v in metric_event.items() if k.startswith("val_")), "")
            st.markdown(f"✅ {label}" + (f" — val: {metric_val}" if metric_val else ""))
        elif model in skipped_models:
            timeout = any(e.get("type") == "model_timeout" and e.get("model") == model for e in events)
            st.markdown(f"{'⏱' if timeout else '⏭'} {label} — {'таймаут' if timeout else 'пропущено'}")
        elif model == current_model:
            progress_msg = last_progress.get("message", "") if last_progress else ""
            progress_pct_str = last_progress.get("pct") if last_progress else None
            progress_pct = float(progress_pct_str) if progress_pct_str else None

            col_status, col_skip = st.columns([5, 1])
            with col_status:
                suffix = f" — {progress_msg}" if progress_msg else ""
                if progress_pct is not None:
                    st.progress(int(progress_pct) / 100, text=f"⏳ {label}{suffix}")
                else:
                    st.markdown(f"⏳ {label}{suffix}")
            with col_skip:
                if st.button("Пропустить", key=f"skip_{model}", use_container_width=True):
                    try:
                        skip_model(project_id, job_id, model)
                    except Exception:
                        pass
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
    project_id = str(project.get("project_id", ""))

    # Сортируем модели по val-метрике для определения топ-3
    sorted_mrs = sorted(model_results, key=lambda mr: mr.get(f"val_{metric}", float("inf")))
    all_model_names = [mr["name"] for mr in sorted_mrs]
    default_model_names = [mr["name"] for mr in sorted_mrs[:3]]

    ts_info = automl_result.get("ts") or {}
    ts_freq = ts_info.get("freq", "MS")
    ts_season = ts_info.get("season_length", 12)
    ts_source = ts_info.get("freq_source", "auto")
    freq_label = _FREQ_LABELS.get(ts_freq, ts_freq)
    source_label = "авто" if ts_source == "auto" else "задана вручную"
    col_best, col_ts1, col_ts2 = st.columns(3)
    col_best.success(f"Лучшая модель: **{_MODEL_LABELS.get(best_model, best_model)}**")
    col_ts1.metric("Частота данных", f"{freq_label} ({ts_freq})", help=f"Источник: {source_label}")
    col_ts2.metric("Сезонный период", ts_season, help="Используется в SeasonalNaive и StatsModels")

    # Глобальный селектор моделей — применяется ко всем графикам на странице
    selected_models = st.multiselect(
        "Модели на графиках",
        options=all_model_names,
        default=[m for m in default_model_names if m in all_model_names],
        format_func=lambda x: _MODEL_LABELS.get(x, x),
        key="automl_model_selector",
    )

    # Сводная таблица моделей
    st.markdown("**Сравнение моделей**")
    summary_rows = [
        {
            "Модель": _MODEL_LABELS.get(mr["name"], mr["name"]),
            f"Val {metric.upper()}": round(mr.get(f"val_{metric}", float("inf")), 4),
            f"Test {metric.upper()}": round(mr.get(f"test_{metric}", float("inf")), 4),
            "Лучшая": "⭐" if mr["name"] == best_model else "",
        }
        for mr in sorted_mrs
    ]
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
        _render_panel_chart(project_id, panel_id, val_periods, test_periods,
                            all_model_names, selected_models)

    st.divider()

    # Лучшие / худшие панели
    col_best, col_worst = st.columns(2)
    top3 = [str(p) for p in panel_df.head(3)["Panel ID"].tolist()]
    bot3 = [str(p) for p in panel_df.tail(3)["Panel ID"].tolist()]
    mini_panel_ids = list(dict.fromkeys(top3 + bot3))

    # Предсказания для всех обученных моделей — кешируем, переключение моделей без запросов
    mini_cache_key = f"automl_mini_preds_{project_id}"
    if mini_cache_key not in st.session_state and mini_panel_ids:
        with st.spinner("Загрузка предсказаний..."):
            try:
                st.session_state[mini_cache_key] = get_automl_predictions(
                    project_id, mini_panel_ids, all_model_names
                )
            except Exception:
                st.session_state[mini_cache_key] = {}
    mini_predictions = st.session_state.get(mini_cache_key, {})

    with col_best:
        st.markdown("**Топ-3 (лучший test)**")
        _render_mini_charts(project_id, top3, val_periods, test_periods, key_prefix="top",
                            predictions=mini_predictions, selected_models=selected_models)
    with col_worst:
        st.markdown("**Антитоп-3 (худший test)**")
        _render_mini_charts(project_id, bot3, val_periods, test_periods, key_prefix="bot",
                            predictions=mini_predictions, selected_models=selected_models)



def _render_panel_chart(
    project_id: str,
    panel_id: str,
    val_periods: int,
    test_periods: int,
    all_models: list[str],
    selected_models: list[str],
) -> None:
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

    # Загружаем предсказания один раз и кешируем в session_state
    cache_key = f"automl_preds_{project_id}_{panel_id}"
    if cache_key not in st.session_state and selected_models:
        with st.spinner("Загрузка предсказаний..."):
            try:
                st.session_state[cache_key] = get_automl_predictions(project_id, [panel_id], all_models)
            except Exception:
                st.session_state[cache_key] = {}
    predictions = st.session_state.get(cache_key, {})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name="Фактическое",
                             line=dict(color="#7C6AF7", width=1.5)))

    for model in selected_models:
        model_preds = predictions.get(model, {}).get(panel_id, [])
        if model_preds:
            pred_dates = [p["date"] for p in model_preds if p["split"] in ("val", "test")]
            pred_vals = [p["y_pred"] for p in model_preds if p["split"] in ("val", "test")]
            if pred_dates:
                fig.add_trace(go.Scatter(
                    x=pred_dates, y=pred_vals, mode="lines",
                    name=_MODEL_LABELS.get(model, model),
                    line=dict(color=_MODEL_COLORS.get(model, "#aaa"), width=1.5, dash="dot"),
                ))

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
        title=f"Панель {panel_id}", height=320, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#333"), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"panel_chart_{panel_id}")


def _render_mini_charts(
    project_id: str,
    panel_ids: list[str],
    val_periods: int,
    test_periods: int,
    key_prefix: str = "",
    predictions: dict | None = None,
    selected_models: list[str] | None = None,
) -> None:
    if not panel_ids:
        return
    try:
        data = get_panels_data(project_id, panel_ids)
    except Exception:
        return
    for series in data:
        panel_id = str(series["panel_id"])
        dates, values = series["dates"], series["values"]
        n = len(dates)
        train_end = n - val_periods - test_periods
        val_end = n - test_periods
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name="Фактическое",
                                 line=dict(color="#7C6AF7", width=1)))

        if predictions and selected_models:
            for model in selected_models:
                model_preds = predictions.get(model, {}).get(panel_id, [])
                pred_dates = [p["date"] for p in model_preds if p["split"] in ("val", "test")]
                pred_vals = [p["y_pred"] for p in model_preds if p["split"] in ("val", "test")]
                if pred_dates:
                    fig.add_trace(go.Scatter(
                        x=pred_dates, y=pred_vals, mode="lines",
                        name=_MODEL_LABELS.get(model, model),
                        line=dict(color=_MODEL_COLORS.get(model, "#4CAF50"), width=1, dash="dot"),
                    ))

        if train_end > 0:
            fig.add_vrect(x0=dates[0], x1=dates[train_end] if train_end < n else dates[-1],
                          fillcolor=_TRAIN_COLOR, line_width=0)
        if 0 < train_end < val_end:
            fig.add_vrect(x0=dates[train_end], x1=dates[val_end] if val_end < n else dates[-1],
                          fillcolor=_VAL_COLOR, line_width=0)
        if val_end < n:
            fig.add_vrect(x0=dates[val_end], x1=dates[-1], fillcolor=_TEST_COLOR, line_width=0)
        fig.update_layout(
            title=f"ID: {panel_id}", height=200, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA",
            xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False),
            showlegend=bool(selected_models),
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="left", x=0, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"mini_{key_prefix}_{panel_id}")


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
        models = st.session_state.get("automl_models")
        if not models:
            # Восстанавливаем список моделей из событий прогресса (при возврате в проект)
            try:
                events = get_automl_progress(project_id, job_id)
                started = [e["model"] for e in events if e.get("type") == "model_start"]
                # total из первого события model_start содержит общее кол-во моделей,
                # но сами модели проще взять из шагов job'а
                job_data = get_job(job_id)
                step_models = [
                    s["name"].removeprefix("train_")
                    for s in (job_data.get("steps") or [])
                    if s.get("name", "").startswith("train_")
                ]
                models = list(dict.fromkeys(started + step_models)) or ["seasonal_naive", "catboost"]
                st.session_state["automl_models"] = models
            except Exception:
                models = ["seasonal_naive", "catboost"]
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
    cfg = _render_config()

    # Частота — из override на экране качества или из preprocessing result
    ts_from_prep = result.get("ts") or {}
    freq_override = st.session_state.get(f"freq_override_{project_id}")
    prep_freq = ts_from_prep.get("freq")
    effective_freq: str | None = freq_override if freq_override else prep_freq
    if prep_freq:
        prep_season = ts_from_prep.get("season_length", 12)
        effective_season = _FREQ_SEASON.get(effective_freq or prep_freq, prep_season)
        freq_lbl = _FREQ_LABELS.get(effective_freq or prep_freq, effective_freq or prep_freq)
        col_f1, col_f2 = st.columns([2, 1])
        col_f1.metric("Частота данных", f"{freq_lbl} ({effective_freq or prep_freq})")
        col_f2.metric("Сезонный период", effective_season)
        if freq_override and freq_override != prep_freq:
            st.caption(f"Автоопределено: {_FREQ_LABELS.get(prep_freq, prep_freq)} ({prep_freq}) — изменить на экране «Качество данных»")
        else:
            st.caption("Изменить на экране «Качество данных»")

    st.divider()
    if not cfg["models"]:
        st.warning("Выберите хотя бы одну модель")
        return

    col_run, col_dl = st.columns([4, 1])
    yaml_bytes = yaml.dump(cfg, allow_unicode=True, default_flow_style=False).encode()
    col_dl.download_button("Сохранить конфиг", yaml_bytes, "automl_config.yaml", "text/yaml", use_container_width=True)

    if col_run.button("▶ Запустить AutoML", type="primary", use_container_width=True):
        with st.spinner("Запускаю..."):
            try:
                job = run_automl(
                    project_id,
                    models=cfg["models"],
                    selection_metric=cfg["selection_metric"],
                    use_hyperopt=cfg["use_hyperopt"],
                    freq=effective_freq,
                    n_trials=cfg["n_trials"],
                    catboost_params=cfg["catboost_params"],
                    autoarima_approximation=cfg["autoarima_approximation"],
                )
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.automl_job_id = str(job["id"])
        st.session_state.automl_models = cfg["models"]
        st.rerun()
