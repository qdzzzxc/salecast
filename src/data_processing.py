import logging
from typing import overload

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.custom_types import AggMethod, ClipBounds, ClippedSplits, PanelScalers, ScaledSplits

logger = logging.getLogger(__name__)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты с логированием."""
    prev_len = len(df)
    df = df.drop_duplicates()
    if prev_len != len(df):
        logger.info(f"Удалено дубликатов: {prev_len} -> {len(df)} (-{prev_len - len(df)})")
    return df


def fit_panel_scalers(
    df: pd.DataFrame,
    panel_col: str,
    target_columns: list[str],
    apply_log: bool = False,
) -> PanelScalers:
    """Обучает скейлеры для каждой панели и колонки."""
    df = df.copy()

    if apply_log:
        df[target_columns] = np.log1p(df[target_columns])

    scalers: PanelScalers = {col: {} for col in target_columns}

    for panel_id, group in df.groupby(panel_col):
        for col in target_columns:
            if col in group.columns:
                scaler = StandardScaler()
                scaler.fit(group[[col]])
                scalers[col][panel_id] = scaler

    return scalers


def transform_panel_columns(
    df: pd.DataFrame,
    scalers: PanelScalers,
    panel_column: str,
    target_columns: list[str],
    apply_log: bool = False,
) -> pd.DataFrame:
    """Применяет логарифм и скейлеры к колонкам по панелям."""
    df = df.copy()

    if apply_log:
        df[target_columns] = np.log1p(df[target_columns])

    for col in target_columns:
        if col not in df.columns:
            continue
        for panel_id, group_idx in df.groupby(panel_column).groups.items():
            if panel_id in scalers[col]:
                df.loc[group_idx, col] = (
                    scalers[col][panel_id].transform(df.loc[group_idx, [col]]).ravel()
                )
            else:
                raise ValueError(f"Scaler not found for panel_id={panel_id}, column={col}.")

    return df


def inverse_transform_panel_columns(
    df: pd.DataFrame,
    scalers: PanelScalers,
    panel_column: str,
    target_columns: list[str],
    apply_log: bool = False,
) -> pd.DataFrame:
    """Обратное преобразование: inverse scale и expm1."""
    df = df.copy()

    for col in target_columns:
        if col not in df.columns:
            continue
        for panel_id, group_idx in df.groupby(panel_column).groups.items():
            if panel_id in scalers[col]:
                df.loc[group_idx, col] = (
                    scalers[col][panel_id].inverse_transform(df.loc[group_idx, [col]]).ravel()
                )

    if apply_log:
        df[target_columns] = np.expm1(df[target_columns])

    return df


def scale_panel_splits(
    splits: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame],
    panel_column: str,
    target_columns: list[str],
    apply_log: bool = False,
) -> ScaledSplits:
    """Скейлит train/val/test сплиты по панелям."""
    train_df, val_df, test_df = splits

    scalers = fit_panel_scalers(train_df, panel_column, target_columns, apply_log)

    return ScaledSplits(
        train=transform_panel_columns(train_df, scalers, panel_column, target_columns, apply_log),
        val=transform_panel_columns(val_df, scalers, panel_column, target_columns, apply_log)
        if val_df is not None
        else None,
        test=transform_panel_columns(test_df, scalers, panel_column, target_columns, apply_log),
        scalers=scalers,
    )


def aggregate_by_panel_date(
    df: pd.DataFrame,
    panel_column: str,
    date_column: str,
    target_columns: list[str],
    agg_method: AggMethod = "sum",
) -> pd.DataFrame:
    """Агрегирует дубликаты по panel-date указанным методом для целевых колонок."""
    key_cols = [panel_column, date_column]
    other_cols = [col for col in df.columns if col not in key_cols + target_columns]

    agg_dict = {col: agg_method for col in target_columns} | {col: "first" for col in other_cols}

    aggregated = df.groupby(key_cols, as_index=False).agg(agg_dict)

    logger.info(
        "Aggregated duplicates by [%s, %s]: %d -> %d rows (-%d)",
        panel_column,
        date_column,
        len(df),
        len(aggregated),
        len(df) - len(aggregated),
    )

    return aggregated


def expand_to_full_panel(
    df: pd.DataFrame,
    panel_column: str = "id",
    date_column: str = "date",
) -> pd.DataFrame:
    """Расширяет панель до полной сетки panel × date, заполняя пропуски NaN."""
    full_index = pd.MultiIndex.from_product(
        [df[panel_column].unique(), df[date_column].unique()],
        names=[panel_column, date_column],
    )

    result = df.set_index([panel_column, date_column]).reindex(full_index).reset_index()

    logger.info("Expanded panel: %s -> %s", df.shape, result.shape)

    return result


def filter_sellers_by_min_periods(
    df: pd.DataFrame, panel_column: str, min_periods: int = 3
) -> pd.DataFrame:
    """Удаляет панели с количеством значений меньше указанного порога."""
    transaction_counts = df.groupby(panel_column).size()
    valid_sellers = transaction_counts[transaction_counts >= min_periods].index

    filtered_df = df[df[panel_column].isin(valid_sellers)]

    removed_count = len(df[panel_column].unique()) - len(valid_sellers)
    logger.info(f"Удалено панелей из-за малого количества данных: {removed_count}")
    logger.info(f"Осталось панелей: {len(valid_sellers)}")
    logger.info(f"Строк до фильтрации: {len(df)}, после: {len(filtered_df)}")

    return filtered_df


def sort_panel_by_date(
    df: pd.DataFrame,
    panel_column: str,
    date_column: str,
) -> pd.DataFrame:
    """Сортирует панели по возрастанию дат."""
    df_copy = df.copy()
    df_sorted = df_copy.sort_values(by=[panel_column, date_column]).reset_index(drop=True)

    logger.info(
        "Sorted %d rows by [%s, %s]",
        len(df_sorted),
        panel_column,
        date_column,
    )

    return df_sorted


@overload
def filter_panels_by_split_missing(
    splits: tuple[pd.DataFrame, None, pd.DataFrame],
    panel_column: str,
    target_columns: list[str],
    maximum_missing_ratio: float = 0.3,
    treat_zero_as_missing: bool = False,
) -> tuple[pd.DataFrame, None, pd.DataFrame]: ...


@overload
def filter_panels_by_split_missing(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    panel_column: str,
    target_columns: list[str],
    maximum_missing_ratio: float = 0.3,
    treat_zero_as_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...


def filter_panels_by_split_missing(
    splits: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame],
    panel_column: str,
    target_columns: list[str],
    maximum_missing_ratio: float = 0.3,
    treat_zero_as_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Удаляет панели, у которых в любом из сплитов доля пропусков выше порога."""
    train_df, val_df, test_df = splits
    non_null_splits = [s for s in splits if s is not None]

    all_panels = set(non_null_splits[0][panel_column].unique())
    valid_panels = all_panels.copy()

    for _, split_df in enumerate(non_null_splits):
        for col in target_columns:
            missing_mask = (
                split_df[col].isna() | (split_df[col] == 0)
                if treat_zero_as_missing
                else split_df[col].isna()
            )

            stats = (
                split_df.assign(_missing=missing_mask)
                .groupby(panel_column)
                .agg(
                    total=(col, "size"),
                    missing=("_missing", "sum"),
                )
            )
            stats["missing_ratio"] = (stats["missing"] / stats["total"]).fillna(1.0)
            valid_panels -= set(stats[stats["missing_ratio"] > maximum_missing_ratio].index)

    logger.info(
        "Filtered by split missing: %d -> %d panels (-%d)",
        len(all_panels),
        len(valid_panels),
        len(all_panels) - len(valid_panels),
    )

    return (
        train_df[train_df[panel_column].isin(valid_panels)].copy(),
        val_df[val_df[panel_column].isin(valid_panels)].copy() if val_df is not None else None,
        test_df[test_df[panel_column].isin(valid_panels)].copy(),
    )


def _compute_clip_bounds(
    df: pd.DataFrame,
    panel_col: str,
    target_col: str,
    lower_percentile: float,
    upper_percentile: float,
    min_panel_size: int,
) -> pd.DataFrame:
    """Вычисляет границы клиппинга для каждой панели."""
    panel_sizes = df.groupby(panel_col)[target_col].count()
    valid_panels = panel_sizes[panel_sizes >= min_panel_size].index

    if len(valid_panels) == 0:
        return pd.DataFrame(columns=["lower", "upper"])

    return (
        df[df[panel_col].isin(valid_panels)]
        .groupby(panel_col)[target_col]
        .agg(
            lower=lambda x: x.quantile(lower_percentile / 100),
            upper=lambda x: x.quantile(upper_percentile / 100),
        )
    )


def _apply_clip_bounds(
    df: pd.DataFrame,
    panel_col: str,
    target_col: str,
    bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Применяет границы клиппинга к датафрейму."""
    df = df.copy()
    df_with_bounds = df.join(bounds, on=panel_col)

    df[target_col] = df_with_bounds[target_col].clip(
        df_with_bounds["lower"], df_with_bounds["upper"]
    )

    return df


def clip_panel_outliers(
    splits: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame],
    panel_column: str,
    target_columns: list[str],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    min_panel_size: int = 10,
) -> ClippedSplits:
    """Клиппит выбросы по перцентилям для каждой панели."""
    train_df, val_df, test_df = splits

    train_clipped = train_df.copy()
    val_clipped = val_df.copy() if val_df is not None else None
    test_clipped = test_df.copy()

    all_bounds: ClipBounds = {}

    for col in target_columns:
        bounds = _compute_clip_bounds(
            train_df, panel_column, col, lower_percentile, upper_percentile, min_panel_size
        )

        if bounds.empty:
            logger.warning("No valid panels for column '%s', skipping", col)
            continue

        all_bounds[col] = bounds.apply(lambda r: (r["lower"], r["upper"]), axis=1).to_dict()

        train_clipped = _apply_clip_bounds(train_clipped, panel_column, col, bounds)
        if val_clipped is not None:
            val_clipped = _apply_clip_bounds(val_clipped, panel_column, col, bounds)
        test_clipped = _apply_clip_bounds(test_clipped, panel_column, col, bounds)

        logger.info(
            "Clipped outliers for '%s' using [%.1f, %.1f] percentiles",
            col,
            lower_percentile,
            upper_percentile,
        )

    return ClippedSplits(
        train=train_clipped,
        val=val_clipped,
        test=test_clipped,
        bounds=all_bounds,
    )


def find_trim_indices(revenue: pd.Series) -> tuple[int, int] | tuple[None, None]:
    """Находит позиции первого и последнего ненулевого значения."""
    non_zero_mask = revenue != 0
    if not non_zero_mask.any():
        return None, None

    first_idx = non_zero_mask.to_numpy().argmax()
    last_idx = len(revenue) - 1 - non_zero_mask[::-1].to_numpy().argmax()

    return first_idx, last_idx


def count_outliers(group: pd.Series) -> int:
    """Подсчитывает количество выбросов по правилу 3*IQR."""
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 3 * iqr

    return (group > upper_fence).sum()
