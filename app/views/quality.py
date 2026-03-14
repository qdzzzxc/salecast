import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.api_client import get_panels_data
from app.state import get_current_project, set_page

_FREQ_OPTIONS: dict[str, str | None] = {
    "🔍 Авто (из данных)": None,
    "📅 Дневная (D)": "D",
    "📅 Недельная (W)": "W",
    "📅 Месячная (MS)": "MS",
    "📅 Квартальная (QS)": "QS",
}
_FREQ_SEASON: dict[str, int] = {"D": 7, "W": 52, "MS": 12, "QS": 4}
_FREQ_LABELS: dict[str, str] = {
    "D": "Дневная", "B": "Рабочие дни",
    "W": "Недельная",
    "MS": "Месячная", "ME": "Месячная", "M": "Месячная",
    "QS": "Квартальная", "Q": "Квартальная",
    "A": "Годовая", "AS": "Годовая",
}

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


def _render_ts_config(ts: dict, project_id: str) -> None:
    """Показывает определённую частоту ряда с возможностью переопределить."""
    freq_auto = ts.get("freq", "MS")
    season_auto = ts.get("season_length", 12)
    freq_sel_key = f"freq_sel_{project_id}"
    override_key = f"freq_override_{project_id}"

    # Streamlit обновляет session_state[widget_key] ДО начала рендера при изменении виджета,
    # поэтому читаем значение selectbox напрямую — метрики уже отражают текущий выбор
    sel_label = st.session_state.get(freq_sel_key, list(_FREQ_OPTIONS.keys())[0])
    current_override = _FREQ_OPTIONS.get(sel_label)  # None если "Авто"
    st.session_state[override_key] = current_override  # синхронизируем для automl.py

    effective_freq = current_override if current_override else freq_auto
    effective_season = _FREQ_SEASON.get(effective_freq, season_auto)
    freq_label = _FREQ_LABELS.get(effective_freq, effective_freq)

    col_f1, col_f2, col_f3 = st.columns([2, 1, 3])
    col_f1.metric("Частота данных", f"{freq_label} ({effective_freq})")
    col_f2.metric("Сезонный период", effective_season)
    with col_f3:
        st.selectbox("Переопределить частоту", options=list(_FREQ_OPTIONS.keys()), key=freq_sel_key)
        if current_override and current_override != freq_auto:
            st.caption(f"Автоопределено: {_FREQ_LABELS.get(freq_auto, freq_auto)} ({freq_auto}), период: {season_auto}")


_TRAIN_COLOR = "rgba(99, 149, 230, 0.15)"
_VAL_COLOR = "rgba(255, 180, 50, 0.2)"
_TEST_COLOR = "rgba(229, 100, 100, 0.2)"


def _add_split_zones(fig: go.Figure, dates: list, val_periods: int, test_periods: int) -> None:
    """Добавляет цветные зоны train/val/test на график."""
    n = len(dates)
    train_end = n - val_periods - test_periods
    val_end = n - test_periods
    if train_end > 0:
        fig.add_vrect(x0=dates[0], x1=dates[train_end] if train_end < n else dates[-1],
                      fillcolor=_TRAIN_COLOR, line_width=0,
                      annotation_text="train", annotation_position="top left")
    if 0 < train_end < val_end:
        fig.add_vrect(x0=dates[train_end], x1=dates[val_end] if val_end < n else dates[-1],
                      fillcolor=_VAL_COLOR, line_width=0,
                      annotation_text="val", annotation_position="top left")
    if val_end < n:
        fig.add_vrect(x0=dates[val_end], x1=dates[-1],
                      fillcolor=_TEST_COLOR, line_width=0,
                      annotation_text="test", annotation_position="top left")


def _render_panels_table(panels: list[dict], project_id: str, val_periods: int = 0, test_periods: int = 0) -> None:
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
            if val_periods or test_periods:
                _add_split_zones(fig, panel["dates"], val_periods, test_periods)
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
    split = result.get("split", {})
    val_periods = split.get("val_periods", 0)
    test_periods = split.get("test_periods", 0)
    project_id = str(project.get("project_id", ""))

    st.title("Качество данных")

    st.divider()
    _render_summary(
        summary=diagnostics.get("summary", {}),
        total_before=filtration.get("total_before", 0),
        total_after=filtration.get("total_after", 0),
    )

    ts = result.get("ts")
    if ts:
        st.divider()
        _render_ts_config(ts, project_id)

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
        _render_panels_table(panels, project_id=str(project.get("project_id", "")), val_periods=val_periods, test_periods=test_periods)
    else:
        st.info("Нет данных")

