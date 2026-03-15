from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.automl.ts_utils import next_dates
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits
from src.seasonal_naive_utilities.evaluate import evaluate_seasonal_naive
from src.seasonal_naive_utilities.train import train_seasonal_naive


class _EmptyParams(BaseModel):
    """Пустые параметры для моделей без гиперпараметров."""

    pass


class SeasonalNaiveForecastModel(BaseForecastModel):
    """Сезонная наивная модель прогнозирования."""

    name: str = "seasonal_naive"

    def __init__(self, seasonal_period: int | None = None) -> None:
        """Инициализирует модель. seasonal_period=None → берётся из settings.ts.season_length."""
        self.seasonal_period = seasonal_period

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        """Обучает сезонную наивную модель и возвращает результаты оценки."""
        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        if progress_fn:
            progress_fn("обучение...", 50.0)

        period = self.seasonal_period if self.seasonal_period is not None else settings.ts.season_length
        model = train_seasonal_naive(
            train_df=splits.train,
            val_df=splits.val,
            settings=settings,
            seasonal_period=period,
        )
        evaluation = evaluate_seasonal_naive(model, splits, settings)

        if progress_fn:
            progress_fn("готово", 100.0)

        return ModelResult(
            name=self.name,
            evaluation=evaluation,
            params=_EmptyParams(),
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Обучает модель на полных данных и строит прогноз на horizon точек вперёд."""
        from src.seasonal_naive_utilities.seasonal_naive_model import SeasonalNaiveModel

        panel_col = settings.columns.id
        date_col = settings.columns.date
        value_col = settings.columns.main_target
        period = self.seasonal_period if self.seasonal_period is not None else settings.ts.season_length

        model = SeasonalNaiveModel(seasonal_period=period)
        model.fit(full_df, panel_col, value_col)

        if on_training_done:
            on_training_done()

        rows = []
        for panel_id, group in full_df.groupby(panel_col):
            future = next_dates(group[date_col], horizon)
            rows.extend({panel_col: panel_id, date_col: d, value_col: np.nan} for d in future)

        future_df = pd.DataFrame(rows)
        preds = model.predict(future_df, panel_col, value_col, is_train=False)
        future_df["forecast"] = np.maximum(preds, 0)
        future_df["panel_id"] = future_df[panel_col].astype(str)
        future_df["date"] = pd.to_datetime(future_df[date_col]).dt.strftime("%Y-%m-%d")
        return future_df[["panel_id", "date", "forecast"]]
