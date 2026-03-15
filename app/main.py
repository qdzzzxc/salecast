from pathlib import Path

import streamlit as st

from app.api_client import create_project, delete_project, list_projects
from app.state import get_current_project, init_state, set_page, set_project
from app.views import automl, forecast, quality, upload

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_DEMO_PROJECTS = [
    ("Demo: Базовый", "gui_data_example.csv"),
    ("Demo: С фильтрацией", "gui_data_example_with_filtration.csv"),
    ("Demo: Store Item Demand (Kaggle)", "store_item_demand.csv"),
]


def _create_demo_projects() -> None:
    """Создаёт демо-проекты (только те, которых ещё нет по имени)."""
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

st.set_page_config(
    page_title="Salecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()


def _project_icon(project: dict) -> str:
    """Возвращает иконку проекта в зависимости от стадии пайплайна."""
    job = project.get("latest_job") or {}
    status = job.get("status", "")
    if not status:
        return "○"
    if status in ("pending", "running"):
        return "⏳"
    if status == "failed":
        return "❌"
    # done — определяем стадию по содержимому result
    result = job.get("result") or {}
    if "forecast" in result:
        return "✅ 📈"
    if "automl" in result:
        return "✅ ⚙"
    if "split" in result:
        return "✅ 🗂"
    return "✅"


def _render_sidebar() -> None:
    """Отображает боковую панель с проектами."""
    with st.sidebar:
        st.markdown("## 📈 Salecast")
        st.divider()

        if st.button("＋ Новый проект", use_container_width=True, type="primary"):
            st.session_state.current_project = None
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
                label = f"{_project_icon(project)} {project['name']}"
                is_active = str(project["id"]) == str(current_id)

                col_name, col_del = st.columns([5, 1])
                with col_name:
                    if st.button(label, use_container_width=True, key=f"open_{project['id']}", disabled=is_active):
                        if job.get("result"):
                            full_job = {**job, "project_id": project["id"]}
                            set_project(full_job)
                            result = job.get("result") or {}
                            if "forecast" in result:
                                set_page("forecast")
                            elif "automl" in result:
                                set_page("automl")
                            else:
                                set_page("quality")
                        elif job.get("status") in ("running", "pending"):
                            # Job ещё выполняется — восстанавливаем состояние
                            job_id = str(job["id"])
                            step_names = {s.get("name", "") for s in (job.get("steps") or [])}
                            # Preprocessing имеет уникальные шаги; всё остальное — AutoML
                            is_preprocessing = bool(step_names & {"filtration", "diagnostics"})
                            if is_preprocessing:
                                st.session_state["polling_job_id"] = job_id
                                st.session_state.current_project = {**project, "project_id": project["id"], "result": {}}
                                set_page("upload")
                            else:
                                st.session_state["automl_job_id"] = job_id
                                st.session_state.current_project = {**project, "project_id": project["id"], "result": {}}
                                set_page("automl")
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

        st.sidebar.divider()
        with st.sidebar.expander("⚙ Настройки"):
            st.markdown("**Сервисы**")
            st.link_button("Flower (Celery)", "http://localhost:5555", use_container_width=True)
            st.link_button("RedisInsight", "http://localhost:5540", use_container_width=True)
            st.link_button("Adminer (PostgreSQL)", "http://localhost:8080", use_container_width=True)
            st.link_button("MinIO Console", "http://localhost:9001", use_container_width=True)
            st.divider()
            if st.button("↺ Сбросить демо-проекты", use_container_width=True):
                try:
                    demo_ids = {str(p["id"]) for p in (projects or []) if p["name"].startswith("Demo:")}
                    for pid in demo_ids:
                        delete_project(pid)
                    current = get_current_project()
                    if current and str(current.get("project_id", current.get("id", ""))) in demo_ids:
                        st.session_state.current_project = None
                        set_page("upload")
                    _create_demo_projects()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                st.rerun()


_STEP_LABELS = {
    "quality": "Качество данных",
    "automl": "Моделирование",
    "forecast": "Прогноз",
}


def _render_steps(page: str, result: dict) -> None:
    """Отображает переключатель шагов пайплайна."""
    options = ["quality", "automl"]
    if result.get("automl"):
        options.append("forecast")
    selected = st.segmented_control(
        label="Шаги",
        options=options,
        format_func=lambda x: _STEP_LABELS[x],
        default=page,
        key=f"pipeline_step_{page}",
        label_visibility="collapsed",
    )
    if selected and selected != page:
        set_page(selected)
        st.rerun()


def _render_page() -> None:
    """Рендерит текущую страницу."""
    page = st.session_state.get("page", "upload")
    if page == "upload":
        upload.render()
        return

    project = get_current_project()
    if project is None:
        upload.render()
        return

    if page in _STEP_LABELS:
        result = (project.get("result") or {})
        _render_steps(page, result)

    if page == "quality":
        quality.render()
    elif page == "automl":
        automl.render()
    elif page == "forecast":
        forecast.render()


_render_sidebar()
_render_page()
