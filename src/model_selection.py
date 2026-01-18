import logging

import pandas as pd

from src.custom_types import SplitRange, Splits

logger = logging.getLogger(__name__)


def temporal_panel_train_test_split(
    df: pd.DataFrame,
    panel_column: str,
    time_column: str,
    train_ratio: float = 0.7,
    ignore_index: bool = True,
) -> Splits[pd.DataFrame]:
    """Делает temporal train/test split для панельных данных."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio должен быть в (0, 1)")

    train_dfs, test_dfs = [], []

    for panel_id in df[panel_column].unique():
        panel_df = df[df[panel_column] == panel_id].sort_values(time_column)
        if ignore_index:
            panel_df = panel_df.reset_index(drop=True)

        train_size = int(len(panel_df) * train_ratio)
        if train_size <= 0 or train_size >= len(panel_df):
            raise ValueError(f"Недостаточно данных для панели {panel_id}: {len(panel_df)} точек")

        train_dfs.append(panel_df.iloc[:train_size])
        test_dfs.append(panel_df.iloc[train_size:])

    train_df = pd.concat(train_dfs, ignore_index=ignore_index)
    test_df = pd.concat(test_dfs, ignore_index=ignore_index)

    _log_split_info(train_df, None, test_df, panel_column)

    return Splits(train=train_df, val=None, test=test_df)


def temporal_panel_train_val_test_split(
    df: pd.DataFrame,
    panel_column: str,
    time_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    ignore_index: bool = True,
) -> Splits[pd.DataFrame]:
    """Делает temporal train/val/test split для панельных данных."""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio и val_ratio должны быть в (0, 1) и их сумма < 1")

    train_dfs, val_dfs, test_dfs = [], [], []

    for panel_id in df[panel_column].unique():
        panel_df = df[df[panel_column] == panel_id].sort_values(time_column)
        if ignore_index:
            panel_df = panel_df.reset_index(drop=True)

        n = len(panel_df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        if train_size <= 0 or val_size <= 0 or train_size + val_size >= n:
            raise ValueError(f"Недостаточно данных для панели {panel_id}: {n} точек")

        train_dfs.append(panel_df.iloc[:train_size])
        val_dfs.append(panel_df.iloc[train_size : train_size + val_size])
        test_dfs.append(panel_df.iloc[train_size + val_size :])

    train_df = pd.concat(train_dfs, ignore_index=ignore_index)
    val_df = pd.concat(val_dfs, ignore_index=ignore_index)
    test_df = pd.concat(test_dfs, ignore_index=ignore_index)

    _log_split_info(train_df, val_df, test_df, panel_column)

    return Splits(train=train_df, val=val_df, test=test_df)


def temporal_panel_split(
    df: pd.DataFrame,
    panel_column: str,
    time_column: str,
    train_ratio: float = 0.7,
    val_ratio: float | None = None,
) -> Splits[pd.DataFrame]:
    """Делает temporal split для панельных данных."""
    if val_ratio:
        return temporal_panel_train_val_test_split(
            df, panel_column, time_column, train_ratio, val_ratio
        )
    return temporal_panel_train_test_split(df, panel_column, time_column, train_ratio)


def temporal_panel_split_by_date(
    df: pd.DataFrame,
    panel_column: str,
    time_column: str,
    train_range: SplitRange,
    test_range: SplitRange,
    val_range: SplitRange | None = None,
) -> Splits[pd.DataFrame]:
    """Делает temporal split по заданным временным промежуткам (одинаковым для всех панелей)."""
    dates = pd.to_datetime(df[time_column]).dt.date

    train_mask = (dates >= train_range.start) & (dates <= train_range.end)
    test_mask = (dates >= test_range.start) & (dates <= test_range.end)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    val_df = None
    if val_range is not None:
        val_mask = (dates >= val_range.start) & (dates <= val_range.end)
        val_df = df[val_mask].copy()

    _log_split_info(train_df, val_df, test_df, panel_column)

    return Splits(train=train_df, val=val_df, test=test_df)


def _log_split_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    panel_column: str,
) -> None:
    """Логирует информацию о сплитах."""
    n_panels = train_df[panel_column].nunique()
    val_info = f", Val: {len(val_df)}" if val_df is not None else ""
    logger.info(
        f"Split: {n_panels} панелей. Train: {len(train_df)}{val_info}, Test: {len(test_df)}"
    )

def temporal_panel_split_by_size(
    df: pd.DataFrame,
    panel_column: str,
    time_column: str,
    test_size: int = 2,
    val_size: int | None = None,
    ignore_index: bool = True,
) -> Splits[pd.DataFrame]:
    """Делает temporal split с фиксированным количеством последних точек в val/test."""
    if test_size < 1:
        raise ValueError("test_size должен быть >= 1")
    if val_size is not None and val_size < 1:
        raise ValueError("val_size должен быть >= 1")

    hold_out_size = test_size + (val_size or 0)
    train_dfs, val_dfs, test_dfs = [], [], []

    for panel_id in df[panel_column].unique():
        panel_df = df[df[panel_column] == panel_id].sort_values(time_column)
        if ignore_index:
            panel_df = panel_df.reset_index(drop=True)

        if len(panel_df) <= hold_out_size:
            raise ValueError(
                f"Недостаточно данных для панели {panel_id}: "
                f"{len(panel_df)} точек, нужно > {hold_out_size}"
            )

        train_dfs.append(panel_df.iloc[:-hold_out_size])
        if val_size is not None:
            val_dfs.append(panel_df.iloc[-hold_out_size:-test_size])
        test_dfs.append(panel_df.iloc[-test_size:])

    train_df = pd.concat(train_dfs, ignore_index=ignore_index)
    val_df = pd.concat(val_dfs, ignore_index=ignore_index) if val_size else None
    test_df = pd.concat(test_dfs, ignore_index=ignore_index)

    _log_split_info(train_df, val_df, test_df, panel_column)

    return Splits(train=train_df, val=val_df, test=test_df)