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


def render() -> None:
    """Отображает экран загрузки данных."""

    # Если идёт polling активного job — показываем только прогресс
    if "polling_job_id" in st.session_state:
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
            set_project(job)
            st.rerun()
        elif job["status"] == "failed":
            del st.session_state.polling_job_id
            st.error("Обработка завершилась с ошибкой")
        else:
            time.sleep(1.5)
            st.rerun()
        return

    # Форма загрузки
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

        st.session_state.polling_job_id = project["latest_job"]["id"]
        st.rerun()
