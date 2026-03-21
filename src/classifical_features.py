import numpy as np
import pandas as pd
from tqdm import tqdm

from src.configs.settings import Settings


def _add_lag_features(
    group: pd.DataFrame,
    target: str,
    lags: list[int],
) -> pd.DataFrame:
    """Добавляет лаговые признаки."""
    result = group.copy()
    for lag in lags:
        result[f"{target}_lag_{lag}"] = result[target].shift(lag)
    return result


def _add_rolling_features(
    group: pd.DataFrame,
    target: str,
    windows: list[int],
) -> pd.DataFrame:
    """Добавляет скользящие средние."""
    result = group.copy()
    for window in windows:
        result[f"{target}_ma_{window}"] = (
            result[target].shift(1).rolling(window, min_periods=1).mean()
        )
    return result


def _add_ema_features(
    group: pd.DataFrame,
    target: str,
    spans: list[int],
) -> pd.DataFrame:
    """Добавляет экспоненциальные скользящие средние."""
    result = group.copy()
    for span in spans:
        result[f"{target}_ema_{span}"] = (
            result[target].shift(1).ewm(span=span, min_periods=1).mean()
        )
    return result


def _add_diff_features(
    group: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Добавляет разностные признаки."""
    result = group.copy()
    result[f"{target}_diff_1"] = result[target].diff(1).shift(1)
    pct_change = result[target].pct_change(1).shift(1)
    result[f"{target}_pct_change_1"] = pct_change.replace([np.inf, -np.inf], np.nan)
    return result


def _add_panel_features(
    group: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Добавляет признаки панели."""
    result = group.copy()
    result["panel_mean"] = result[target].expanding().mean().shift(1)
    result["panel_std"] = result[target].expanding().std().shift(1)
    shifted_target = result[target].shift(1)
    result[f"{target}_vs_mean"] = (
        (shifted_target - result["panel_mean"]) / result["panel_std"]
    ).replace([np.inf, -np.inf], np.nan)
    return result


def _add_trend_features(
    group: pd.DataFrame,
    target: str,
    window: int,
) -> pd.DataFrame:
    """Добавляет признак тренда — наклон линейной регрессии на скользящем окне.

    shift(1): используем только прошлые значения, чтобы не было утечки.
    """
    result = group.copy()
    series = result[target].shift(1)

    def _slope(values: np.ndarray) -> float:
        n = len(values)
        if n < 2 or np.all(np.isnan(values)):
            return np.nan
        x = np.arange(n, dtype=float)
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan
        x_m, y_m = x[mask], values[mask]
        x_mean = x_m.mean()
        denom = ((x_m - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x_m - x_mean) * (y_m - y_m.mean())).sum() / denom)

    result[f"{target}_trend_{window}"] = (
        series.rolling(window, min_periods=2).apply(_slope, raw=True)
    )
    return result


def _add_cdf_features(
    group: pd.DataFrame,
    target: str,
    decay: float,
) -> pd.DataFrame:
    """Добавляет CDF-признак — взвешенная доля прошлых значений ≤ текущему.

    shift(1): используем только прошлые значения, чтобы не было утечки.
    decay: вес убывает как decay^(n-1-i) для более старых точек.
    """
    result = group.copy()
    series = result[target].shift(1).values
    n = len(series)
    cdf_vals = np.full(n, np.nan)

    for i in range(1, n):
        past = series[:i]
        valid = ~np.isnan(past)
        if not valid.any():
            continue
        past_valid = past[valid]
        idx = np.where(valid)[0]
        # weights: decay^(age), age=0 для самой последней точки
        ages = i - 1 - idx
        weights = decay ** ages
        w_total = weights.sum()
        if w_total == 0:
            continue
        current = series[i]
        if np.isnan(current):
            continue
        cdf_vals[i] = (weights[past_valid <= current]).sum() / w_total

    result[f"{target}_cdf"] = cdf_vals
    return result


def _add_calendar_features(
    group: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Добавляет календарные признаки."""
    result = group.copy()
    dates = pd.to_datetime(result[date_column])
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    return result


def build_ts_features(
    df: pd.DataFrame,
    settings: Settings,
    drop_na: bool = False,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    """Создаёт признаки для месячных данных с малым количеством точек."""
    df = df.copy()
    df = df.sort_values([settings.columns.id, settings.columns.date])

    target = settings.columns.main_target
    panel_col = settings.columns.id
    date_col = settings.columns.date

    lags = settings.downstream.lags
    windows = settings.downstream.windows
    ema_spans = settings.downstream.ema_spans

    # Предвычисляем MSTL seasonal если включено
    mstl_seasonal: dict[str, np.ndarray] | None = None
    if settings.downstream.use_mstl_seasonal:
        from src.mstl_features import decompose_mstl
        mstl_seasonal = {}
        for pid, grp in df.groupby(panel_col):
            vals = grp[target].values
            try:
                decomp = decompose_mstl(vals, freq=settings.ts.freq)
                mstl_seasonal[pid] = decomp["seasonal"]
            except Exception:
                mstl_seasonal[pid] = np.zeros(len(vals))

    features = []

    panels = df.groupby(panel_col)
    for pid, group in tqdm(panels, total=df[panel_col].nunique(), desc="Processing panels", disable=disable_tqdm):
        group = group.copy()

        group = _add_lag_features(group, target, lags)
        group = _add_rolling_features(group, target, windows)
        group = _add_ema_features(group, target, ema_spans)
        group = _add_diff_features(group, target)
        group = _add_panel_features(group, target)
        group = _add_calendar_features(group, date_col)
        if settings.downstream.use_trend:
            group = _add_trend_features(group, target, settings.downstream.trend_window)
        if settings.downstream.use_cdf:
            group = _add_cdf_features(group, target, settings.downstream.cdf_decay)
        if mstl_seasonal is not None and pid in mstl_seasonal:
            group[f"{target}_mstl_seasonal"] = mstl_seasonal[pid]

        features.append(group)

    result = pd.concat(features, ignore_index=True)

    if drop_na:
        result = result.dropna()

    return result