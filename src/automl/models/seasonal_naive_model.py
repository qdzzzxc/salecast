import pandas as pd
from pydantic import BaseModel

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
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
