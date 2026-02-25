import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import get_current_project, set_page

_STATUS_COLOR = {"green": "#4CAF50", "yellow": "#FFC107", "red": "#F44336"}
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
    col2.metric("После фильтрации", total_after, delta=f"{total_after - total_before}")
    col3.metric("🟢 Зелёных", summary.get("green", 0))
    col4.metric("🟡 Жёлтых", summary.get("yellow", 0))
    col5.metric("🔴 Красных", summary.get("red", 0))


def _render_status_chart(summary: dict[str, int]) -> None:
    """Отображает круговую диаграмму статусов."""
    data = {
        "Статус": ["Зелёный", "Жёлтый", "Красный"],
        "Количество": [summary.get("green", 0), summary.get("yellow", 0), summary.get("red", 0)],
        "Цвет": ["#4CAF50", "#FFC107", "#F44336"],
    }
    df = pd.DataFrame(data)
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
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_panels_table(panels: list[dict]) -> None:
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
        horizontal=True,
    )
    filter_map = {"🟢 Зелёные": "green", "🟡 Жёлтые": "yellow", "🔴 Красные": "red"}
    if status_filter != "Все":
        df = df[df["Статус"] == filter_map[status_filter]]

    st.dataframe(
        df.drop(columns=["Статус"]),
        use_container_width=True,
        hide_index=True,
    )


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
    st.caption(f"Проект: **{project.get('project_id', '')}**")

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
        st.markdown("**Шаги фильтрации**")
        steps = filtration.get("steps", {})
        if steps:
            steps_df = pd.DataFrame(
                [{"Шаг": k, "Отфильтровано": v} for k, v in steps.items()]
            )
            st.dataframe(steps_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Детализация по панелям**")
    panels = diagnostics.get("panels", [])
    if panels:
        _render_panels_table(panels)
    else:
        st.info("Нет данных")

    st.divider()
    if st.button("Запустить AutoML →", type="primary"):
        set_page("automl")
