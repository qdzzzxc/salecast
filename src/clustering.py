import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ts_features import _safe_autocorr, extract_features_for_groups

logger = logging.getLogger(__name__)


def extract_panel_features(
    df: pd.DataFrame,
    panel_col: str,
    value_col: str,
    use_mstl: bool = False,
    freq: str = "MS",
) -> pd.DataFrame:
    """Извлекает нормализованные признаки TS для каждой панели.

    Расширяет extract_features_for_groups добавлением autocorr_lag12 (сезонность),
    опционально MSTL seasonality_strength / trend_strength,
    и StandardScaler-нормализацией для корректной кластеризации.

    Returns:
        DataFrame с нормализованными признаками, index = panel_id.
    """
    features_df = extract_features_for_groups(df, panel_col, value_col)

    autocorr12 = {}
    for panel_id, group in df.groupby(panel_col):
        autocorr12[panel_id] = _safe_autocorr(group[value_col].values, lag=12)
    features_df["autocorr_lag12"] = pd.Series(autocorr12)
    features_df["autocorr_lag12"] = features_df["autocorr_lag12"].fillna(0.0)

    if use_mstl:
        from src.mstl_features import extract_mstl_features
        mstl_df = extract_mstl_features(df, panel_col, value_col, freq=freq)
        # join по индексу — панели, для которых MSTL не посчитался, получат NaN → заполним 0
        features_df = features_df.join(mstl_df, how="left")
        features_df = features_df.fillna(0.0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df.values)
    return pd.DataFrame(scaled, index=features_df.index, columns=features_df.columns)


def cluster_panels(
    features_df: pd.DataFrame,
    n_clusters: int = 5,
    method: str = "kmeans",
) -> pd.Series:
    """Кластеризует панели по признакам TS.

    Args:
        features_df: нормализованные признаки, index = panel_id.
        n_clusters: количество кластеров (для KMeans) или min_cluster_size (для HDBSCAN).
        method: "kmeans" или "hdbscan".

    Returns:
        Series panel_id → cluster_id (int). HDBSCAN может вернуть -1 для шума.
    """
    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError as e:
            raise ImportError("hdbscan не установлен. Установите: uv add hdbscan") from e
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_clusters))
        labels = clusterer.fit_predict(features_df.values)
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(features_df.values)
    else:
        raise ValueError(f"Неизвестный метод кластеризации: {method!r}. Доступны: 'kmeans', 'hdbscan'")

    result = pd.Series(labels, index=features_df.index, name="cluster_id")
    n_actual = result[result >= 0].nunique()
    logger.info("Кластеризация %s: %d кластеров из %d панелей", method, n_actual, len(features_df))
    return result


def cluster_panels_auto(
    features_df: pd.DataFrame,
    max_k: int = 10,
) -> tuple[pd.Series, dict[int, float], int]:
    """Автоподбор числа кластеров KMeans по silhouette score.

    Args:
        features_df: нормализованные признаки, index = panel_id.
        max_k: максимальное число кластеров для перебора.

    Returns:
        (labels, silhouette_scores, best_k) — метки, словарь {k: score}, лучшее k.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    upper = min(max_k, len(features_df) - 1)
    if upper < 2:
        labels = pd.Series(np.zeros(len(features_df), dtype=int),
                           index=features_df.index, name="cluster_id")
        return labels, {1: 0.0}, 1

    scores: dict[int, float] = {}
    all_labels: dict[int, np.ndarray] = {}
    data = features_df.values

    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        k_labels = km.fit_predict(data)
        scores[k] = float(silhouette_score(data, k_labels))
        all_labels[k] = k_labels

    best_k = max(scores, key=scores.get)
    labels = pd.Series(all_labels[best_k], index=features_df.index, name="cluster_id")
    logger.info(
        "KMeans auto: best_k=%d, silhouette=%.3f (range 2..%d)",
        best_k, scores[best_k], upper,
    )
    return labels, scores, best_k


def compute_umap_embedding(features_df: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    """Вычисляет 2D UMAP-эмбеддинг признаков панелей для визуализации.

    Returns:
        ndarray shape (n_panels, 2).
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn не установлен. Установите: uv add umap-learn") from e

    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, len(features_df) - 1))
    return reducer.fit_transform(features_df.values)


def compute_cluster_mean_ts(
    df: pd.DataFrame,
    panel_col: str,
    date_col: str,
    value_col: str,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """Вычисляет средний временной ряд для каждого кластера.

    Args:
        df: исходный датафрейм с историей.
        cluster_labels: Series panel_id → cluster_id.

    Returns:
        DataFrame с колонками [cluster_id, date, mean_value].
    """
    merged = df.merge(
        cluster_labels.rename("cluster_id").reset_index().rename(columns={"index": panel_col}),
        on=panel_col,
        how="inner",
    )
    result = (
        merged.groupby(["cluster_id", date_col])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "mean_value"})
    )
    result["cluster_id"] = result["cluster_id"].astype(int)
    return result
