import time

import pandas as pd
import streamlit as st

from app.api_client import create_project, get_job
from app.state import set_project

_STEP_LABELS = {
    "loading": "Загрузка данных",
    "filtration": "Фильтрация рядов",
    "diagnostics": "Диагностика качества",
    "saving": "Сохранение результатов",
}
_ALL_STEPS = list(_STEP_LABELS.keys())


def _poll_job(job_id: str) -> None:
    """Поллит статус задачи и показывает прогресс."""
    progress = st.progress(0, text="Запуск обработки...")
    status_placeholder = st.empty()

    completed_steps: set[str] = set()
    while True:
        job = get_job(job_id)
        status = job["status"]
        steps_done = {s["name"] for s in job["steps"]}

        n_done = len(steps_done)
        pct = int(n_done / len(_ALL_STEPS) * 100)
        progress.progress(pct, text=f"{pct}%")

        with status_placeholder.container():
            for step_name in _ALL_STEPS:
                label = _STEP_LABELS[step_name]
                if step_name in steps_done:
                    if step_name not in completed_steps:
                        completed_steps.add(step_name)
                    st.markdown(f"✅ {label}")
                elif status == "running" and n_done == _ALL_STEPS.index(step_name):
                    st.markdown(f"⏳ {label}")
                else:
                    st.markdown(f"⬜ {label}")

        if status in ("done", "failed"):
            break
        time.sleep(1.5)
        st.rerun()

    if status == "done":
        progress.progress(100, text="Готово")
        set_project(job)
        st.rerun()
    else:
        st.error("Обработка завершилась с ошибкой")


def render() -> None:
    """Отображает экран загрузки данных."""
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
        panel_col = st.selectbox("Колонка ID (панель)", columns)
    with col2:
        date_col = st.selectbox("Колонка даты", columns)
    with col3:
        value_col = st.selectbox("Колонка значений", columns)

    name = st.text_input("Название проекта", value=uploaded.name.replace(".csv", ""))

    if st.button("Запустить обработку", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Введите название проекта")
            return
        with st.spinner("Создаю проект..."):
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

        job_id = project["latest_job"]["id"]
        st.divider()
        st.markdown("**Обработка данных**")
        _poll_job(job_id)
