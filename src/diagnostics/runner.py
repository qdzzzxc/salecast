import logging

import numpy as np
import pandas as pd

from src.custom_types import DiagnosticsResult, PanelDiagnostics, QualityStatus
from src.diagnostics.checks import (
    check_autocorrelation,
    check_cv,
    check_length,
    check_seasonality,
    check_stationarity,
    check_trend,
    check_zero_ratio,
)
from src.diagnostics.config import DiagnosticsConfig

logger = logging.getLogger(__name__)

_STATUS_ORDER: dict[QualityStatus, int] = {"green": 0, "yellow": 1, "red": 2}


def _worst_status(statuses: list[QualityStatus]) -> QualityStatus:
    """Возвращает наихудший статус из списка."""
    return max(statuses, key=lambda s: _STATUS_ORDER[s])


def _diagnose_panel(
    values: np.ndarray,
    panel_id: int | str,
    config: DiagnosticsConfig,
) -> PanelDiagnostics:
    """Запускает все проверки для одной панели."""
    checks = [
        check_length(values, config),
        check_zero_ratio(values, config),
        check_cv(values, config),
        check_autocorrelation(values, config),
        check_stationarity(values, config),
        check_seasonality(values, config),
        check_trend(values, config),
    ]
    overall = _worst_status([c.status for c in checks])
    return PanelDiagnostics(panel_id=panel_id, overall_status=overall, checks=checks)


def run_diagnostics(
    df: pd.DataFrame,
    panel_col: str,
    date_col: str,
    value_col: str,
    config: DiagnosticsConfig | None = None,
) -> DiagnosticsResult:
    """Запускает диагностику качества временных рядов для всех панелей.

    Args:
        df (pd.DataFrame): Панельный датафрейм с временными рядами
        panel_col (str): Название колонки с идентификатором панели
        date_col (str): Название колонки с датой
        value_col (str): Название колонки с целевой переменной
        config (DiagnosticsConfig | None, optional): Конфигурация порогов. Defaults to None.

    Returns:
        DiagnosticsResult: Результаты диагностики по всем панелям
    """
    if config is None:
        config = DiagnosticsConfig()
    df_sorted = df.sort_values([panel_col, date_col])
    panels = []
    for panel_id, group in df_sorted.groupby(panel_col):
        values = group[value_col].to_numpy(dtype=float)
        panel_diag = _diagnose_panel(values, panel_id=panel_id, config=config)
        panels.append(panel_diag)
    logger.info("Диагностика завершена: %d панелей", len(panels))
    return DiagnosticsResult(panels=panels)
