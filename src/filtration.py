import logging

import pandas as pd

from src.configs.settings import FiltrationConfig
from src.custom_types import FiltrationResult, FiltrationStepReport

logger = logging.getLogger(__name__)


def _track_drops(
    before: pd.DataFrame,
    after: pd.DataFrame,
    group_col: str,
    step: str,
    reason: str,
) -> FiltrationStepReport:
    """Определяет панели, отфильтрованные на данном шаге."""
    ids_before = set(before[group_col].unique())
    ids_after = set(after[group_col].unique())
    dropped = ids_before - ids_after
    logger.info("%s: dropped %d panels", step, len(dropped))
    return FiltrationStepReport(step=step, reason=reason, dropped_ids=dropped)


def _aggregate_duplicates(df: pd.DataFrame, group_col: str, date_col: str) -> pd.DataFrame:
    """Агрегирует дубликаты по панели и дате."""
    numeric_cols = df.select_dtypes("number").columns.tolist()
    agg_dict = {col: "sum" for col in numeric_cols if col not in [group_col, date_col]}
    return df.groupby([group_col, date_col], as_index=False).agg(agg_dict)


def _trim_edge_zeros(group: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Обрезает нули по краям временного ряда."""
    values = group[value_col].values
    nonzero_mask = values != 0

    if not nonzero_mask.any():
        return pd.DataFrame()

    first_idx = nonzero_mask.argmax()
    last_idx = len(values) - 1 - nonzero_mask[::-1].argmax()

    return group.iloc[first_idx : last_idx + 1]


def _filter_by_edge_zeros(
    df: pd.DataFrame, group_col: str, date_col: str, value_col: str
) -> pd.DataFrame:
    """Удаляет нулевые значения по краям временных рядов."""
    return (
        df.sort_values([group_col, date_col])
        .groupby(group_col, group_keys=False)
        .apply(_trim_edge_zeros, value_col=value_col)
        .reset_index(drop=True)
    )


def _filter_by_inner_zeros(
    df: pd.DataFrame, group_col: str, value_col: str, max_zero_ratio: float
) -> pd.DataFrame:
    """Удаляет группы с большим количеством нулей внутри ряда."""
    zero_ratios = df.groupby(group_col)[value_col].apply(lambda x: (x == 0).mean())
    valid_groups = zero_ratios[zero_ratios <= max_zero_ratio].index
    return df[df[group_col].isin(valid_groups)]


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


def filter_time_series(df: pd.DataFrame, config: FiltrationConfig) -> FiltrationResult:
    """Применяет все фильтрации к временным рядам."""
    cols = config.columns
    steps: list[FiltrationStepReport] = []

    result = _aggregate_duplicates(df, group_col=cols.id, date_col=cols.date)
    step_report = _track_drops(
        df, result, cols.id, "aggregate_duplicates", "Дубликаты агрегированы"
    )
    steps.append(step_report)

    prev = result
    result = _filter_by_edge_zeros(
        result, group_col=cols.id, date_col=cols.date, value_col=cols.main_target
    )
    steps.append(_track_drops(prev, result, cols.id, "edge_zeros", "Ряд состоит только из нулей"))

    prev = result
    result = _filter_by_inner_zeros(
        result, group_col=cols.id, value_col=cols.main_target, max_zero_ratio=config.max_zero_ratio
    )
    steps.append(
        _track_drops(prev, result, cols.id, "inner_zeros", f"Доля нулей > {config.max_zero_ratio}")
    )

    prev = result
    result = _filter_by_group_size(result, group_col=cols.id, min_size=config.min_series_length)
    steps.append(
        _track_drops(
            prev, result, cols.id, "min_length", f"Длина ряда < {config.min_series_length}"
        )
    )

    prev = result
    result = _filter_by_zero_std(result, group_col=cols.id, value_col=cols.main_target)
    steps.append(_track_drops(prev, result, cols.id, "zero_std", "Нулевое стандартное отклонение"))

    prev = result
    result = _filter_by_min_total(
        result, group_col=cols.id, value_col=cols.main_target, min_total=config.min_total_sales
    )
    steps.append(
        _track_drops(prev, result, cols.id, "min_total", f"Сумма продаж < {config.min_total_sales}")
    )

    return FiltrationResult(df=result, steps=steps)
