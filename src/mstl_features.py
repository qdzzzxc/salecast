"""MSTL-декомпозиция временных рядов и извлечение сезонных признаков."""

import logging

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA

logger = logging.getLogger(__name__)

_FREQ_SEASONS: dict[str, list[int]] = {
    "D": [7, 30, 365],
    "W": [4, 13, 52],
    "MS": [12],
    "ME": [12],
    "M": [12],
    "QS": [4],
    "Q": [4],
}


def _filter_season_lengths(season_lengths: list[int], n: int) -> list[int]:
    """Оставляет только периоды, для которых хватает данных (2*period + 1)."""
    return [s for s in season_lengths if n >= s * 2 + 1]


def decompose_mstl(
    values: np.ndarray,
    freq: str = "MS",
    season_lengths: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Выполняет MSTL-декомпозицию одного временного ряда.

    Args:
        values: значения ряда.
        freq: частота (D, W, MS и т.д.).
        season_lengths: сезонные периоды. None → берутся из _FREQ_SEASONS.

    Returns:
        dict с ключами: trend, seasonal (суммарная), remainder, + seasonal_i.
    """
    if season_lengths is None:
        season_lengths = _FREQ_SEASONS.get(freq, [12])

    season_lengths = _filter_season_lengths(season_lengths, len(values))
    if not season_lengths:
        return {
            "trend": values.copy(),
            "seasonal": np.zeros_like(values),
            "remainder": np.zeros_like(values),
        }

    dates = pd.date_range(start="2020-01-01", periods=len(values), freq=freq)
    df = pd.DataFrame({"unique_id": 1, "ds": dates, "y": values.astype(float)})

    models = [MSTL(season_length=season_lengths, trend_forecaster=AutoARIMA())]
    sf = StatsForecast(models=models, freq=freq)
    sf = sf.fit(df=df)

    decomp = sf.fitted_[0, 0].model_
    result: dict[str, np.ndarray] = {"trend": np.asarray(decomp["trend"])}

    seasonal_cols = [k for k in decomp if k.startswith("seasonal")]
    seasonal_sum = np.zeros(len(values))
    for col in seasonal_cols:
        arr = np.asarray(decomp[col])
        result[col] = arr
        seasonal_sum += arr

    result["seasonal"] = seasonal_sum
    result["remainder"] = np.asarray(decomp.get("remainder", values - result["trend"] - seasonal_sum))
    return result


def seasonality_strength(
    seasonal: np.ndarray,
    remainder: np.ndarray,
) -> float:
    """Вычисляет силу сезонности: 1 - Var(remainder) / Var(seasonal + remainder).

    Значение от 0 (нет сезонности) до 1 (сильная сезонность).
    Формула из Wang, Smith & Hyndman (2006).
    """
    detrended = seasonal + remainder
    var_detrended = np.var(detrended)
    if var_detrended < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - np.var(remainder) / var_detrended))


def extract_mstl_features(
    df: pd.DataFrame,
    panel_col: str,
    value_col: str,
    freq: str = "MS",
    season_lengths: list[int] | None = None,
) -> pd.DataFrame:
    """Извлекает MSTL-признаки для каждой панели.

    Returns:
        DataFrame index=panel_id, columns=[seasonality_strength, trend_strength].
    """
    records: list[dict] = []

    for panel_id, group in df.groupby(panel_col):
        values = group[value_col].values
        if len(values) < 4:
            continue

        try:
            decomp = decompose_mstl(values, freq=freq, season_lengths=season_lengths)
        except Exception:
            logger.warning("MSTL не удался для панели %s, пропускаем", panel_id)
            continue

        ss = seasonality_strength(decomp["seasonal"], decomp["remainder"])

        # Trend strength: 1 - Var(remainder) / Var(trend + remainder)
        trend_plus_rem = decomp["trend"] + decomp["remainder"]
        var_tr = np.var(trend_plus_rem)
        ts = float(max(0.0, 1.0 - np.var(decomp["remainder"]) / var_tr)) if var_tr > 1e-12 else 0.0

        records.append({
            panel_col: panel_id,
            "seasonality_strength": ss,
            "trend_strength": ts,
        })

    result = pd.DataFrame(records).set_index(panel_col)
    logger.info(
        "MSTL-признаки: %d панелей, avg seasonality=%.3f, avg trend=%.3f",
        len(result),
        result["seasonality_strength"].mean() if len(result) > 0 else 0,
        result["trend_strength"].mean() if len(result) > 0 else 0,
    )
    return result


def extract_seasonal_vectors(
    df: pd.DataFrame,
    panel_col: str,
    value_col: str,
    freq: str = "MS",
    season_lengths: list[int] | None = None,
) -> pd.DataFrame:
    """Извлекает сезонную компоненту как вектор для каждой панели.

    Для кластеризации по сезонному паттерну: нормализованный сезонный вектор
    обрезается/дополняется до одного полного сезонного цикла.

    Returns:
        DataFrame index=panel_id, columns=[s_0, s_1, ..., s_{period-1}].
    """
    if season_lengths is None:
        season_lengths = _FREQ_SEASONS.get(freq, [12])
    main_period = season_lengths[0]

    records: list[dict] = []

    for panel_id, group in df.groupby(panel_col):
        values = group[value_col].values
        if len(values) < main_period:
            continue

        try:
            decomp = decompose_mstl(values, freq=freq, season_lengths=season_lengths)
        except Exception:
            continue

        seasonal = decomp["seasonal"]
        # Берём последний полный цикл
        cycle = seasonal[-main_period:]
        # Нормализуем
        std = np.std(cycle)
        if std > 1e-12:
            cycle = (cycle - np.mean(cycle)) / std

        row = {panel_col: panel_id}
        for i, v in enumerate(cycle):
            row[f"s_{i}"] = float(v)
        records.append(row)

    return pd.DataFrame(records).set_index(panel_col)
