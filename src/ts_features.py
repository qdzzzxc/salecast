import numpy as np
import pandas as pd
from scipy import stats


def _compute_sign_changes(series: np.ndarray) -> int:
    """Считает количество смен знака в приращениях."""
    diff = np.diff(series)
    signs = np.sign(diff)
    signs = signs[signs != 0]
    return int(np.sum(signs[:-1] != signs[1:])) if len(signs) > 1 else 0


def _compute_slope(series: np.ndarray) -> float:
    """Вычисляет наклон линейного тренда."""
    x = np.arange(len(series))
    slope, _, _, _, _ = stats.linregress(x, series)
    return slope


def _safe_autocorr(series: np.ndarray, lag: int = 1) -> float:
    """Вычисляет автокорреляцию с защитой от константных рядов."""
    if len(series) <= lag or np.std(series) == 0:
        return 0.0
    result = pd.Series(series).autocorr(lag=lag)
    return result if pd.notna(result) else 0.0


def extract_series_features(series: np.ndarray) -> dict[str, float]:
    """Извлекает фичи из одного временного ряда."""
    mean_val = np.mean(series)
    std_val = np.std(series)

    return {
        "mean": mean_val,
        "std": std_val,
        "cv": std_val / mean_val if mean_val != 0 else 0.0,
        "zero_ratio": np.mean(series == 0),
        "spike_ratio": np.mean(series > mean_val + 2 * std_val),
        "max_mean_ratio": np.max(series) / mean_val if mean_val != 0 else 0.0,
        "sign_changes": _compute_sign_changes(series),
        "slope": _compute_slope(series),
        "autocorr_lag1": _safe_autocorr(series, lag=1),
    }


def extract_features_for_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Извлекает фичи для каждой группы временного ряда."""
    features_list = []

    for group_id, group_df in df.groupby(group_col):
        series = group_df[value_col].values
        features = extract_series_features(series)
        features[group_col] = group_id
        features_list.append(features)

    return pd.DataFrame(features_list).set_index(group_col)
