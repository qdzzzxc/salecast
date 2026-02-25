import streamlit as st

from app.api_client import list_projects
from app.pages import quality, upload
from app.state import get_current_project, init_state, set_page, set_project

st.set_page_config(
    page_title="Salecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()


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
                if st.button(label, use_container_width=True, key=project["id"], disabled=is_active):
                    if job.get("result"):
                        full_job = {**job, "project_id": project["id"]}
                        set_project(full_job)
                        set_page("quality")
                    else:
                        st.session_state.current_project = {**project}
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
