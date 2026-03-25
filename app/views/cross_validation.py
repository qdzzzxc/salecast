"""Экран кросс-валидации — проверка стабильности модели на нескольких фолдах."""

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from app.api_client import get_cv_progress, get_cv_result, get_job, run_cv
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
    "weighted_avg": "Взвешенное среднее",
    "best_per_panel": "Лучшая модель на панель",
}


def _render_progress(project_id: str, job_id: str, n_folds: int) -> bool:
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


def _render_results(project_id: str) -> None:
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
        display_cols = [
            c
            for c in ["fold", "mape", "rmse", "mae", "train_rows", "test_rows"]
            if c in fold_df.columns
        ]
        st.dataframe(
            fold_df[display_cols].rename(
                columns={
                    "fold": "Fold",
                    "mape": "MAPE",
                    "rmse": "RMSE",
                    "mae": "MAE",
                    "train_rows": "Train",
                    "test_rows": "Test",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    # Box plot MAPE по панелям
    if panel_metrics:
        pm_df = pd.DataFrame(panel_metrics)
        if "mape" in pm_df.columns and len(pm_df) > 0:
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
            st.plotly_chart(fig, width="stretch", key="cv_box_plot")


def render() -> None:
    """Отображает экран кросс-валидации."""
    project = get_current_project()
    if project is None:
        st.warning("Проект не выбран")
        return

    result = project.get("result") or {}
    automl_result = result.get("automl")
    project_id = str(project.get("project_id", ""))

    st.title("Кросс-валидация")
    st.caption(
        "Temporal CV с expanding window — проверка стабильности модели на нескольких фолдах."
    )

    if not automl_result:
        st.info("Сначала завершите шаг Моделирования")
        return

    # Если есть результат CV — показываем
    cv_result = result.get("cross_validation")
    if cv_result and not st.session_state.get("cv_job_id"):
        model_type = cv_result.get("model_type", "?")
        model_label = _MODEL_LABELS.get(model_type, model_type)
        n_folds = cv_result.get("n_folds", "?")
        st.success(f"Модель: **{model_label}** · Фолдов: **{n_folds}**")
        _render_results(project_id)

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
        done = _render_progress(project_id, job_id, n_folds)
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

    # Добавляем "ensemble" если есть результат ансамбля
    has_ensemble = bool(result.get("ensemble"))
    cv_options = list(all_models)
    if has_ensemble:
        cv_options = ["ensemble"] + cv_options

    default_model = "ensemble" if has_ensemble else best_model

    col1, col2 = st.columns(2)
    with col1:
        cv_model = st.selectbox(
            "Модель для CV",
            options=cv_options or [best_model],
            index=cv_options.index(default_model) if default_model in cv_options else 0,
            format_func=lambda x: _MODEL_LABELS.get(x, x),
            key="cv_model_select",
        )
    with col2:
        n_folds = st.number_input(
            "Количество фолдов",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            key="cv_n_folds_input",
        )

    # Параметры ансамбля для CV
    ensemble_models_list: list[str] | None = None
    ensemble_method = "weighted_avg"
    if cv_model == "ensemble":
        ens_info = result.get("ensemble", {})
        ensemble_models_list = ens_info.get("models", all_models)
        ensemble_method = ens_info.get("method", "weighted_avg")
        st.caption(
            f"CV ансамбля: на каждом фолде обучаются все модели "
            f"({', '.join(_MODEL_LABELS.get(m, m) for m in ensemble_models_list)}) "
            f"и комбинируются методом «{_MODEL_LABELS.get(ensemble_method, ensemble_method)}»."
        )

    if st.button("▶ Запустить кросс-валидацию", type="primary", width="stretch"):
        with st.spinner("Запускаю..."):
            try:
                job = run_cv(
                    project_id,
                    cv_model,
                    int(n_folds),
                    ensemble_models=ensemble_models_list,
                    ensemble_method=ensemble_method,
                )
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.cv_job_id = str(job["id"])
        st.session_state.cv_n_folds = int(n_folds)
        st.session_state.pop("cv_show_form", None)
        st.rerun()
