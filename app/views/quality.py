import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.api_client import get_panels_data
from app.state import get_current_project, set_page

_STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
_CHECK_NAMES = {
    "length": "Длина ряда",
    "zero_ratio": "Доля нулей",
    "cv": "Волатильность (CV)",
    "autocorrelation": "Автокорреляция",
    "stationarity": "Стационарность",
    "seasonality": "Сезонность",
    "trend": "Тренд",
}


def _render_summary(summary: dict[str, int], total_before: int, total_after: int) -> None:
    """Отображает сводные метрики качества данных."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Всего рядов (до)", total_before)
    col2.metric("После фильтрации", total_after, delta=str(total_after - total_before))
    col3.metric("🟢 Зелёных", summary.get("green", 0))
    col4.metric("🟡 Жёлтых", summary.get("yellow", 0))
    col5.metric("🔴 Красных", summary.get("red", 0))


def _render_status_chart(summary: dict[str, int]) -> None:
    """Отображает круговую диаграмму статусов."""
    df = pd.DataFrame({
        "Статус": ["Зелёный", "Жёлтый", "Красный"],
        "Количество": [summary.get("green", 0), summary.get("yellow", 0), summary.get("red", 0)],
    })
    fig = px.pie(
        df,
        values="Количество",
        names="Статус",
        color="Статус",
        color_discrete_map={"Зелёный": "#4CAF50", "Жёлтый": "#FFC107", "Красный": "#F44336"},
        hole=0.5,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        margin=dict(t=0, b=0, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_panel_charts(panels: list[dict]) -> None:
    """Отображает графики временных рядов."""
    cols = st.columns(min(len(panels), 3))
    for i, panel in enumerate(panels):
        with cols[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=panel["dates"],
                y=panel["values"],
                mode="lines+markers",
                line=dict(color="#F44336", width=2),
                marker=dict(size=4),
            ))
            fig.update_layout(
                title=f"ID: {panel['panel_id']}",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
                margin=dict(t=30, b=20, l=20, r=10),
                height=200,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_filtration_steps(steps: dict, filtered_samples: dict, project_id: str) -> None:
    """Отображает шаги фильтрации с графиками."""
    non_zero = {k: v for k, v in steps.items() if v > 0}
    if not non_zero:
        st.caption("Ни один ряд не был отфильтрован")
        return

    for step_name, count in non_zero.items():
        step_data = filtered_samples.get(step_name, {})
        reason = step_data.get("reason", step_name)
        all_ids = step_data.get("panel_ids", [])
        key = f"filtered_panels_{step_name}"

        with st.expander(f"**{reason}** — отфильтровано {count} рядов", expanded=False):
            if st.button("Показать примеры рядов", key=f"show_{step_name}") and all_ids:
                picked = random.sample(all_ids, min(3, len(all_ids)))
                with st.spinner("Загружаю данные..."):
                    try:
                        st.session_state[key] = get_panels_data(project_id, picked)
                    except Exception as e:
                        st.error(f"Ошибка загрузки: {e}")
            if st.session_state.get(key):
                _render_panel_charts(st.session_state[key])


def _render_panels_table(panels: list[dict], project_id: str) -> None:
    """Отображает таблицу с результатами диагностики по панелям."""
    rows = []
    for p in panels:
        row = {
            "": _STATUS_EMOJI.get(p["overall_status"], ""),
            "Panel ID": p["panel_id"],
            "Статус": p["overall_status"],
        }
        for check_key, check_label in _CHECK_NAMES.items():
            passed = p.get(f"{check_key}_passed")
            row[check_label] = "✅" if passed else "❌"
        rows.append(row)

    df = pd.DataFrame(rows)

    status_filter = st.selectbox(
        "Фильтр по статусу",
        ["Все", "🟢 Зелёные", "🟡 Жёлтые", "🔴 Красные"],
    )
    filter_map = {"🟢 Зелёные": "green", "🟡 Жёлтые": "yellow", "🔴 Красные": "red"}
    if status_filter != "Все":
        df = df[df["Статус"] == filter_map[status_filter]]

    display_df = df.drop(columns=["Статус"]).reset_index(drop=True)
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
        st.markdown(f"**Ряд: {panel_id}**")
        with st.spinner("Загружаю данные..."):
            try:
                data = get_panels_data(project_id, [panel_id])
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
                return
        if data:
            panel = data[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=panel["dates"],
                y=panel["values"],
                mode="lines+markers",
                line=dict(color="#7C6AF7", width=2),
                marker=dict(size=5),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
                margin=dict(t=10, b=30, l=40, r=10),
                height=250,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    """Отображает экран качества данных."""
    project = get_current_project()
    if project is None:
        st.warning("Проект не выбран")
        if st.button("← Загрузить данные"):
            set_page("upload")
        return

    result = project.get("result") or {}
    filtration = result.get("filtration", {})
    diagnostics = result.get("diagnostics", {})

    st.title("Качество данных")

    st.divider()
    _render_summary(
        summary=diagnostics.get("summary", {}),
        total_before=filtration.get("total_before", 0),
        total_after=filtration.get("total_after", 0),
    )

    st.divider()
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Распределение статусов**")
        _render_status_chart(diagnostics.get("summary", {}))

    with col_right:
        st.markdown("**Фильтрация рядов**")
        _render_filtration_steps(
            steps=filtration.get("steps", {}),
            filtered_samples=filtration.get("filtered_samples", {}),
            project_id=str(project.get("project_id", "")),
        )

    st.divider()
    st.markdown("**Детализация по панелям**")
    panels = diagnostics.get("panels", [])
    if panels:
        _render_panels_table(panels, project_id=str(project.get("project_id", "")))
    else:
        st.info("Нет данных")

    st.divider()
    if st.button("Запустить AutoML →", type="primary"):
        set_page("automl")
