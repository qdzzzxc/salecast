import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.api_client import get_cluster_data, get_job, run_clustering
from app.state import get_current_project

_CLUSTER_COLORS = px.colors.qualitative.Plotly
_OUTLIER_COLOR = "#888888"
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
)


def _render_progress(job_id: str) -> bool:
    """Отображает прогресс кластеризации. Возвращает True если завершено."""
    try:
        job = get_job(job_id)
    except Exception as e:
        st.warning(f"Задача не найдена ({e}). Можно запустить заново.")
        if "clustering_job_id" in st.session_state:
            del st.session_state["clustering_job_id"]
        return False

    steps = job.get("steps") or []
    step_labels = {
        "loading": "Загрузка данных",
        "features": "Извлечение признаков TS",
        "clustering": "Кластеризация",
        "umap": "UMAP-проекция",
        "mean_ts": "Средние ряды по кластерам",
        "saving": "Сохранение",
    }
    done_names = {s["name"] for s in steps}
    total = len(step_labels)
    n_done = sum(1 for k in step_labels if k in done_names)
    st.progress(n_done / total, text=f"{n_done} / {total} шагов")

    for key, label in step_labels.items():
        if key in done_names:
            st.markdown(f"✅ {label}")
        elif job["status"] == "running":
            current_step = steps[-1]["name"] if steps else ""
            if current_step == key:
                st.markdown(f"⏳ {label}")
            else:
                st.markdown(f"⬜ {label}")

    if job["status"] == "done":
        return True
    if job["status"] == "failed":
        st.error("Кластеризация завершилась с ошибкой")
        del st.session_state.clustering_job_id
        return False

    time.sleep(2)
    st.rerun()
    return False


def _render_umap(umap_records: list[dict], panel_col: str) -> None:
    """Отображает UMAP-scatter по кластерам с выделением выбросов."""
    df = pd.DataFrame(umap_records)
    df["cluster_id"] = df["cluster_id"].astype(int)
    has_outliers = (df["cluster_id"] == -1).any()

    # Строим цветовую карту: кластеры → Plotly palette, -1 → серый
    clusters_sorted = sorted(df[df["cluster_id"] >= 0]["cluster_id"].unique())
    color_map = {
        str(c): _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, c in enumerate(clusters_sorted)
    }
    if has_outliers:
        color_map["-1"] = _OUTLIER_COLOR

    df["cluster_label"] = df["cluster_id"].apply(lambda c: "Шум" if c == -1 else str(c))

    hover_cols = [panel_col] if panel_col in df.columns else None
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label",
        hover_data=hover_cols,
        color_discrete_map={("Шум" if k == "-1" else k): v for k, v in color_map.items()},
        category_orders={
            "cluster_label": [str(c) for c in clusters_sorted] + (["Шум"] if has_outliers else [])
        },
        title="UMAP — проекция панелей по кластерам",
        labels={"x": "UMAP-1", "y": "UMAP-2", "cluster_label": "Кластер"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))

    # Выбросы — маркер ×
    for trace in fig.data:
        if trace.name == "Шум":
            trace.marker.symbol = "x"
            trace.marker.size = 7
            trace.marker.opacity = 0.5

    fig.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0), **_DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _render_distribution(umap_records: list[dict]) -> None:
    """Отображает распределение панелей по кластерам."""
    df = pd.DataFrame(umap_records)
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["cluster_label"] = df["cluster_id"].apply(lambda c: "Шум" if c == -1 else str(c))
    counts = df.groupby("cluster_label").size().reset_index(name="count")
    # Сортировка: числовые кластеры по возрастанию, "Шум" в конце
    order = sorted(
        [x for x in counts["cluster_label"] if x != "Шум"],
        key=lambda x: int(x),
    )
    if "Шум" in counts["cluster_label"].values:
        order.append("Шум")
    counts["cluster_label"] = pd.Categorical(counts["cluster_label"], categories=order)
    counts = counts.sort_values("cluster_label")

    clusters_sorted = [x for x in order if x != "Шум"]
    color_map = {
        c: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, c in enumerate(clusters_sorted)
    }
    color_map["Шум"] = _OUTLIER_COLOR

    fig = px.bar(
        counts,
        x="cluster_label",
        y="count",
        color="cluster_label",
        color_discrete_map=color_map,
        title="Распределение панелей по кластерам",
        labels={"cluster_label": "Кластер", "count": "Панелей"},
    )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        **_DARK_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_silhouette(silhouette_scores: dict, best_k: int | None) -> None:
    """Отображает график silhouette score по числу кластеров."""
    ks = sorted(int(k) for k in silhouette_scores)
    scores = [silhouette_scores[str(k)] for k in ks]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=scores,
            mode="lines+markers",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=8),
            name="Silhouette",
        )
    )
    if best_k is not None and best_k in ks:
        best_score = silhouette_scores[str(best_k)]
        fig.add_trace(
            go.Scatter(
                x=[best_k],
                y=[best_score],
                mode="markers",
                marker=dict(size=14, color="#EF553B", symbol="star"),
                name=f"Лучшее k={best_k}",
            )
        )
    fig.update_layout(
        title="Silhouette score по количеству кластеров",
        xaxis_title="k (число кластеров)",
        yaxis_title="Silhouette score",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        **_DARK_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_mean_ts(mean_ts_records: list[dict]) -> None:
    """Отображает средние временные ряды по кластерам в сетке."""
    df = pd.DataFrame(mean_ts_records)
    df["date"] = pd.to_datetime(df["date"])
    clusters = sorted(df["cluster_id"].unique())
    n_cols = min(3, len(clusters))

    cols = st.columns(n_cols)
    for i, cluster_id in enumerate(clusters):
        cluster_df = df[df["cluster_id"] == cluster_id].sort_values("date")
        color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        fig = go.Figure()
        fill_color = (
            color.replace("rgb", "rgba").replace(")", ", 0.15)")
            if color.startswith("rgb")
            else color
        )
        fig.add_trace(
            go.Scatter(
                x=cluster_df["date"],
                y=cluster_df["mean_value"],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=fill_color,
            )
        )
        fig.update_layout(
            title=f"Кластер {cluster_id}",
            height=200,
            margin=dict(l=0, r=0, t=35, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor="#333"),
            showlegend=False,
            **_DARK_LAYOUT,
        )
        with cols[i % n_cols]:
            st.plotly_chart(fig, use_container_width=True, key=f"cluster_ts_{cluster_id}")


def render() -> None:
    """Отображает экран кластеризации."""
    project = get_current_project()
    if project is None:
        st.warning("Проект не выбран")
        return

    result = project.get("result") or {}
    project_id = str(project.get("project_id", ""))
    clustering_result = result.get("clustering")

    st.title("Кластеризация")
    st.caption(
        "Опциональный шаг — группирует панели по паттерну TS."
        " Открывает кластерные модели (CatBoost clustered, TS2Vec clustered) в AutoML."
    )

    rerun_key = f"clustering_rerun_{project_id}"

    # Если уже есть результат и не запрошен перезапуск — показываем его
    if (
        clustering_result
        and not st.session_state.get("clustering_job_id")
        and not st.session_state.get(rerun_key)
    ):
        # Метрики
        method = clustering_result["method"]
        n_outliers = clustering_result.get("n_outliers", 0)
        if n_outliers > 0:
            c1, c2, c3, c4 = st.columns(4)
            c4.metric("Шум (outliers)", n_outliers)
        else:
            c1, c2, c3 = st.columns(3)
        c1.metric("Кластеров", clustering_result["n_clusters"])
        c2.metric("Панелей", clustering_result["n_panels"])
        c3.metric("Метод", method.upper())

        # Загрузка данных
        cache_key = f"cluster_data_{project_id}"
        if cache_key not in st.session_state:
            with st.spinner("Загрузка данных..."):
                try:
                    st.session_state[cache_key] = get_cluster_data(project_id)
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")
                    return

        data = st.session_state[cache_key]
        panel_col = project.get("panel_col", "article")

        # Silhouette chart (kmeans_auto)
        if data.get("silhouette_scores"):
            _render_silhouette(data["silhouette_scores"], data.get("best_k"))

        # UMAP + distribution в две колонки
        col_umap, col_dist = st.columns([3, 2])
        with col_umap:
            _render_umap(data["umap"], panel_col)
        with col_dist:
            _render_distribution(data["umap"])

        st.markdown("**Средние ряды по кластерам**")
        _render_mean_ts(data["mean_ts"])

        st.divider()
        if st.button("🔄 Перезапустить с новыми параметрами"):
            st.session_state[rerun_key] = True
            st.session_state.pop(cache_key, None)
            st.rerun()
        return

    # Polling
    if "clustering_job_id" in st.session_state:
        job_id = st.session_state.clustering_job_id
        st.markdown("**Кластеризация...**")
        done = _render_progress(job_id)
        if "clustering_job_id" not in st.session_state:
            st.rerun()
            return
        if done:
            try:
                job = get_job(job_id)
                new_result = {**result, **job["result"]}
                st.session_state.current_project = {
                    **project,
                    "result": new_result,
                    "project_id": project_id,
                }
            except Exception:
                pass
            del st.session_state.clustering_job_id
            st.session_state.pop(rerun_key, None)
            st.rerun()
        return

    # Форма запуска
    _METHOD_LABELS = {
        "kmeans": "KMeans (фиксированное N)",
        "kmeans_auto": "KMeans Auto (подбор N по silhouette)",
        "hdbscan": "HDBSCAN (авто N, выделяет выбросы)",
    }
    _FEATURE_MODE_LABELS = {
        "all": "Все признаки TS",
        "seasonal": "Только сезонный паттерн (MSTL)",
    }
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Метод",
            list(_METHOD_LABELS.keys()),
            format_func=lambda x: _METHOD_LABELS[x],
            key="cluster_method",
        )
    with col2:
        if method == "kmeans":
            n_clusters = st.number_input(
                "Количество кластеров",
                min_value=2,
                max_value=20,
                value=5,
                key="cluster_n",
            )
        elif method == "kmeans_auto":
            n_clusters = st.number_input(
                "Максимальное K",
                min_value=3,
                max_value=20,
                value=10,
                key="cluster_n",
                help="Перебор от 2 до этого значения, выбор лучшего по silhouette score",
            )
        else:
            n_clusters = st.number_input(
                "min_cluster_size",
                min_value=2,
                max_value=50,
                value=5,
                key="cluster_n",
                help="Минимальный размер кластера для HDBSCAN",
            )

    # MSTL-признаки
    col_feat, col_mstl = st.columns(2)
    with col_feat:
        feature_mode = st.selectbox(
            "Признаки для кластеризации",
            list(_FEATURE_MODE_LABELS.keys()),
            format_func=lambda x: _FEATURE_MODE_LABELS[x],
            key="cluster_feature_mode",
        )
    with col_mstl:
        use_mstl = False
        if feature_mode == "all":
            use_mstl = st.checkbox(
                "Добавить MSTL-признаки",
                value=False,
                key="cluster_use_mstl",
                help="Добавляет seasonality_strength и trend_strength из MSTL-декомпозиции",
            )
        else:
            st.caption("MSTL-декомпозиция → нормализованный сезонный вектор для каждой панели")

    if st.button("▶ Запустить кластеризацию", type="primary", use_container_width=True):
        with st.spinner("Запускаю..."):
            try:
                job = run_clustering(
                    project_id,
                    n_clusters=int(n_clusters),
                    method=method,
                    use_mstl=use_mstl,
                    feature_mode=feature_mode,
                )
            except Exception as e:
                st.error(f"Ошибка запуска: {e}")
                return
        st.session_state.clustering_job_id = str(job["id"])
        st.session_state.pop(rerun_key, None)
        st.rerun()
