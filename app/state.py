from typing import Any

import streamlit as st


def init_state() -> None:
    """Инициализирует session_state при первом запуске."""
    if "current_project" not in st.session_state:
        st.session_state.current_project = None
    if "page" not in st.session_state:
        st.session_state.page = "upload"


def set_project(project: dict[str, Any]) -> None:
    """Устанавливает текущий проект и переключает на страницу качества данных."""
    st.session_state.current_project = project
    st.session_state.page = "quality"


def set_page(page: str) -> None:
    """Переключает текущую страницу."""
    st.session_state.page = page


def get_current_project() -> dict[str, Any] | None:
    """Возвращает текущий проект."""
    return st.session_state.get("current_project")
