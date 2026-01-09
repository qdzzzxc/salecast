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


def build_monthly_features(
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

    features = []

    panels = df.groupby(panel_col)
    for _, group in tqdm(panels, total=df[panel_col].nunique(), desc="Processing panels", disable=disable_tqdm):
        group = group.copy()

        group = _add_lag_features(group, target, lags)
        group = _add_rolling_features(group, target, windows)
        group = _add_ema_features(group, target, ema_spans)
        group = _add_diff_features(group, target)
        group = _add_panel_features(group, target)
        group = _add_calendar_features(group, date_col)

        features.append(group)

    result = pd.concat(features, ignore_index=True)

    if drop_na:
        result = result.dropna()

    return result