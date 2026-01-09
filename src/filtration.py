import pandas as pd

from src.configs.settings import FiltrationConfig


def _filter_by_group_size(df: pd.DataFrame, group_col: str, min_size: int) -> pd.DataFrame:
    """Фильтрует датафрейм, оставляя только группы с минимальным размером."""
    group_sizes = df.groupby(group_col).size()
    valid_groups = group_sizes[group_sizes >= min_size].index
    return df[df[group_col].isin(valid_groups)]


def _filter_by_zero_std(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Удаляет группы с нулевым стандартным отклонением."""
    group_stds = df.groupby(group_col)[value_col].std()
    valid_groups = group_stds[group_stds > 0].index
    return df[df[group_col].isin(valid_groups)]


def _filter_by_min_total(
    df: pd.DataFrame, group_col: str, value_col: str, min_total: int
) -> pd.DataFrame:
    """Удаляет группы с суммой значений меньше порога."""
    group_totals = df.groupby(group_col)[value_col].sum()
    valid_groups = group_totals[group_totals >= min_total].index
    return df[df[group_col].isin(valid_groups)]


def filter_time_series(df: pd.DataFrame, config: FiltrationConfig) -> pd.DataFrame:
    """Применяет все фильтрации к временным рядам."""
    result = _filter_by_group_size(df, group_col="article", min_size=config.min_series_length)
    result = _filter_by_zero_std(result, group_col="article", value_col="sales")
    result = _filter_by_min_total(
        result, group_col="article", value_col="sales", min_total=config.min_total_sales
    )
    return result
