from pathlib import Path

import streamlit as st

from app.api_client import create_project, delete_project, list_projects
from app.pages import quality, upload
from app.state import get_current_project, init_state, set_page, set_project

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_DEMO_PROJECTS = [
    ("Demo: Базовый", "gui_data_example.csv"),
    ("Demo: С фильтрацией", "gui_data_example_with_filtration.csv"),
]


def _ensure_demo_projects() -> None:
    """Создаёт демо-проекты при первом запуске, если они ещё не существуют."""
    if st.session_state.get("demo_initialized"):
        return
    try:
        existing_names = {p["name"] for p in list_projects()}
        for name, filename in _DEMO_PROJECTS:
            if name not in existing_names:
                csv_path = _EXAMPLES_DIR / filename
                if csv_path.exists():
                    create_project(
                        name=name,
                        file_bytes=csv_path.read_bytes(),
                        filename=filename,
                        panel_col="article",
                        date_col="date",
                        value_col="sales",
                    )
        st.session_state["demo_initialized"] = True
    except Exception:
        pass  # API ещё не готов — попробуем при следующем рендере

st.set_page_config(
    page_title="Salecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
_ensure_demo_projects()


def _render_sidebar() -> None:
    """Отображает боковую панель с проектами."""
    with st.sidebar:
        st.markdown("## 📈 Salecast")
        st.divider()

        if st.button("＋ Новый проект", use_container_width=True, type="primary"):
            set_page("upload")
            st.rerun()

        st.markdown("**Проекты**")
        try:
            projects = list_projects()
        except Exception:
            projects = []
            st.caption("API недоступен")

        if not projects:
            st.caption("Нет проектов")
        else:
            current = get_current_project()
            current_id = current.get("project_id") if current else None
            for project in projects:
                job = project.get("latest_job") or {}
                status = job.get("status", "")
                icon = {"done": "✅", "running": "⏳", "failed": "❌", "pending": "🕐"}.get(status, "")
                label = f"{icon} {project['name']}"
                is_active = str(project["id"]) == str(current_id)

                col_name, col_del = st.columns([5, 1])
                with col_name:
                    if st.button(label, use_container_width=True, key=f"open_{project['id']}", disabled=is_active):
                        if job.get("result"):
                            full_job = {**job, "project_id": project["id"]}
                            set_project(full_job)
                            set_page("quality")
                        else:
                            st.session_state.current_project = {**project}
                            set_page("upload")
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{project['id']}", help="Удалить проект"):
                        delete_project(str(project["id"]))
                        if is_active:
                            st.session_state.current_project = None
                            set_page("upload")
                        st.rerun()


def _render_page() -> None:
    """Рендерит текущую страницу."""
    page = st.session_state.get("page", "upload")
    if page == "upload":
        upload.render()
    elif page == "quality":
        quality.render()
    elif page == "automl":
        st.title("AutoML")
        st.info("В разработке")
    elif page == "forecast":
        st.title("Прогноз")
        st.info("В разработке")


_render_sidebar()
_render_page()
