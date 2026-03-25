"""Ансамблирование предсказаний нескольких моделей."""

import numpy as np
import pandas as pd

_EPS = 1e-10


def compute_inverse_metric_weights(model_metrics: dict[str, float]) -> dict[str, float]:
    """Вычисляет нормализованные веса как обратную величину метрики (e.g. MAPE).

    Чем ниже метрика, тем выше вес. Бесконечные/nan значения получают вес 0.
    """
    if not model_metrics:
        return {}
    if len(model_metrics) == 1:
        return {k: 1.0 for k in model_metrics}

    inv: dict[str, float] = {}
    for name, metric in model_metrics.items():
        if np.isfinite(metric) and metric >= 0:
            inv[name] = 1.0 / (metric + _EPS)
        else:
            inv[name] = 0.0

    total = sum(inv.values())
    if total <= 0:
        # Все модели с inf/nan — равные веса
        n = len(model_metrics)
        return {k: 1.0 / n for k in model_metrics}

    return {k: v / total for k, v in inv.items()}


def weighted_average_predictions(
    predictions: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Комбинирует predictions (val/test) через взвешенное среднее.

    predictions: {model: DataFrame[panel_id, date, split, y_pred]}
    weights: {model: weight}
    Returns: DataFrame[panel_id, date, split, y_pred]
    """
    dfs: list[pd.DataFrame] = []
    for model_name, df in predictions.items():
        w = weights.get(model_name, 0.0)
        if w <= 0:
            continue
        tmp = df[["panel_id", "date", "split", "y_pred"]].copy()
        tmp["weighted_pred"] = tmp["y_pred"] * w
        tmp["weight"] = w
        dfs.append(tmp)

    if not dfs:
        return pd.DataFrame(columns=["panel_id", "date", "split", "y_pred"])

    merged = pd.concat(dfs, ignore_index=True)
    grouped = merged.groupby(["panel_id", "date", "split"], as_index=False).agg(
        weighted_sum=("weighted_pred", "sum"),
        weight_sum=("weight", "sum"),
    )
    grouped["y_pred"] = grouped["weighted_sum"] / grouped["weight_sum"]
    return grouped[["panel_id", "date", "split", "y_pred"]]


def select_best_model_per_panel(
    model_panel_metrics: dict[str, list[dict]],
) -> dict[str, str]:
    """Для каждой панели выбирает модель с наименьшей val метрикой.

    model_panel_metrics: {model: [{panel_id, val, test}, ...]}
    Returns: {panel_id: best_model_name}
    """
    panel_best: dict[str, tuple[str, float]] = {}  # {panel_id: (model, val_metric)}

    for model_name, metrics_list in model_panel_metrics.items():
        for pm in metrics_list:
            pid = str(pm["panel_id"])
            val = pm.get("val")
            if val is None or not np.isfinite(val):
                continue
            if pid not in panel_best or val < panel_best[pid][1]:
                panel_best[pid] = (model_name, val)

    return {pid: model for pid, (model, _) in panel_best.items()}


def best_per_panel_predictions(
    predictions: dict[str, pd.DataFrame],
    panel_best_model: dict[str, str],
) -> pd.DataFrame:
    """Для каждой панели берёт predictions из лучшей модели.

    predictions: {model: DataFrame[panel_id, date, split, y_pred]}
    panel_best_model: {panel_id: model_name}
    Returns: DataFrame[panel_id, date, split, y_pred]
    """
    parts: list[pd.DataFrame] = []
    for pid, model_name in panel_best_model.items():
        df = predictions.get(model_name)
        if df is None:
            continue
        panel_rows = df[df["panel_id"] == str(pid)]
        if not panel_rows.empty:
            parts.append(panel_rows[["panel_id", "date", "split", "y_pred"]])

    if not parts:
        return pd.DataFrame(columns=["panel_id", "date", "split", "y_pred"])

    return pd.concat(parts, ignore_index=True)


def weighted_average_forecasts(
    forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Комбинирует forecasts через взвешенное среднее.

    forecasts: {model: DataFrame[panel_id, date, forecast]}
    weights: {model: weight}
    Returns: DataFrame[panel_id, date, forecast]
    """
    dfs: list[pd.DataFrame] = []
    for model_name, df in forecasts.items():
        w = weights.get(model_name, 0.0)
        if w <= 0:
            continue
        tmp = df[["panel_id", "date", "forecast"]].copy()
        tmp["weighted_fc"] = tmp["forecast"] * w
        tmp["weight"] = w
        dfs.append(tmp)

    if not dfs:
        return pd.DataFrame(columns=["panel_id", "date", "forecast"])

    merged = pd.concat(dfs, ignore_index=True)
    grouped = merged.groupby(["panel_id", "date"], as_index=False).agg(
        weighted_sum=("weighted_fc", "sum"),
        weight_sum=("weight", "sum"),
    )
    grouped["forecast"] = grouped["weighted_sum"] / grouped["weight_sum"]
    return grouped[["panel_id", "date", "forecast"]]


def best_per_panel_forecasts(
    forecasts: dict[str, pd.DataFrame],
    panel_best_model: dict[str, str],
) -> pd.DataFrame:
    """Для каждой панели берёт forecast из лучшей модели.

    forecasts: {model: DataFrame[panel_id, date, forecast]}
    panel_best_model: {panel_id: model_name}
    Returns: DataFrame[panel_id, date, forecast]
    """
    parts: list[pd.DataFrame] = []
    for pid, model_name in panel_best_model.items():
        df = forecasts.get(model_name)
        if df is None:
            continue
        panel_rows = df[df["panel_id"] == str(pid)]
        if not panel_rows.empty:
            parts.append(panel_rows[["panel_id", "date", "forecast"]])

    if not parts:
        return pd.DataFrame(columns=["panel_id", "date", "forecast"])

    return pd.concat(parts, ignore_index=True)
