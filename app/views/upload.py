import time

import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from app.api_client import create_project, get_job, get_panels_data, get_project_preview, run_project
from app.state import get_current_project, set_page, set_project

_STEP_LABELS = {
    "loading": "Загрузка данных",
    "filtration": "Фильтрация рядов",
    "diagnostics": "Диагностика качества",
    "saving": "Сохранение результатов",
}
_ALL_STEPS = list(_STEP_LABELS.keys())


def _render_progress(job: dict) -> None:
    """Отображает текущий прогресс задачи."""
    steps_done = {s["name"] for s in job["steps"]}
    n_done = len(steps_done)
    pct = int(n_done / len(_ALL_STEPS) * 100)

    st.progress(pct, text=f"{pct}%")
    for i, step_name in enumerate(_ALL_STEPS):
        label = _STEP_LABELS[step_name]
        if step_name in steps_done:
            st.markdown(f"✅ {label}")
        elif job["status"] == "running" and n_done == i:
            st.markdown(f"⏳ {label}")
        else:
            st.markdown(f"⬜ {label}")


def _render_polling() -> None:
    """Отображает прогресс активной задачи обработки."""
    job_id = st.session_state.polling_job_id
    st.title("Обработка данных")
    try:
        job = get_job(job_id)
    except Exception as e:
        st.error(f"Ошибка получения статуса: {e}")
        del st.session_state.polling_job_id
        return

    _render_progress(job)

    if job["status"] == "done":
        del st.session_state.polling_job_id
        project = get_current_project() or {}
        set_project({**job, "project_id": project.get("id", project.get("project_id", ""))})
        st.rerun()
    elif job["status"] == "failed":
        del st.session_state.polling_job_id
        st.error("Обработка завершилась с ошибкой")
    else:
        time.sleep(1.5)
        st.rerun()


def _render_ready_to_run(project: dict) -> None:
    """Отображает превью сырых данных проекта с кнопкой запуска обработки."""
    project_id = str(project.get("id", project.get("project_id", "")))

    st.title(project.get("name", "Проект"))

    try:
        preview = get_project_preview(project_id)
    except Exception as e:
        st.error(f"Ошибка загрузки превью: {e}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Панелей", preview["panel_count"])
    col2.metric("Строк", f"{preview['row_count']:,}")
    col3.metric("Начало", preview["date_min"])
    col4.metric("Конец", preview["date_max"])

    if st.button("▶ Запустить обработку", type="primary", use_container_width=True):
        with st.spinner("Запускаю..."):
            try:
                job = run_project(project_id)
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.polling_job_id = str(job["id"])
        st.rerun()

    st.divider()
    st.markdown("**Панели**")

    panels = preview["panels"]
    display_df = pd.DataFrame(panels).rename(columns={
        "panel_id": "Panel ID",
        "rows": "Строк",
        "date_min": "С",
        "date_max": "По",
    })

    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
    )

    selected_rows = selection.selection.get("rows", [])
    if selected_rows:
        panel_id = str(display_df.iloc[selected_rows[0]]["Panel ID"])
        try:
            data = get_panels_data(project_id, [panel_id])
        except Exception as e:
            st.error(f"Ошибка загрузки ряда: {e}")
            return
        if data:
            series = data[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["dates"], y=series["values"], mode="lines"))
            fig.update_layout(
                title=f"Панель {panel_id}",
                xaxis_title="Дата",
                yaxis_title="Значение",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_upload_form() -> None:
    """Отображает форму загрузки нового CSV."""
    st.title("Новый проект")
    st.caption("Загрузите CSV с временными рядами продаж")

    uploaded = st.file_uploader("CSV файл", type=["csv"])
    if uploaded is None:
        return

    df_preview = pd.read_csv(uploaded, nrows=5)
    uploaded.seek(0)
    columns = list(df_preview.columns)

    st.markdown("**Предпросмотр данных**")
    st.dataframe(df_preview, use_container_width=True)

    st.divider()
    st.markdown("**Маппинг колонок**")

    col1, col2, col3 = st.columns(3)
    with col1:
        panel_col = st.selectbox("Колонка ID (панель)", columns, index=0)
    with col2:
        date_col = st.selectbox("Колонка даты", columns, index=min(1, len(columns) - 1))
    with col3:
        value_col = st.selectbox("Колонка значений", columns, index=min(2, len(columns) - 1))

    name = st.text_input("Название проекта", value=uploaded.name.replace(".csv", ""))

    if st.button("Сохранить проект", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Введите название проекта")
            return
        with st.spinner("Сохраняю..."):
            try:
                project = create_project(
                    name=name,
                    file_bytes=uploaded.read(),
                    filename=uploaded.name,
                    panel_col=panel_col,
                    date_col=date_col,
                    value_col=value_col,
                )
            except Exception as e:
                st.error(f"Ошибка создания проекта: {e}")
                return

        st.session_state.current_project = project
        set_page("upload")
        st.rerun()


def render() -> None:
    """Отображает экран загрузки данных."""
    if "polling_job_id" in st.session_state:
        _render_polling()
        return

    project = get_current_project()
    if project and project.get("latest_job") is None:
        _render_ready_to_run(project)
        return

    _render_upload_form()
