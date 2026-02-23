import pandas as pd
from pydantic import BaseModel

from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits
from src.seasonal_naive_utilities.evaluate import evaluate_seasonal_naive
from src.seasonal_naive_utilities.train import train_seasonal_naive


class _EmptyParams(BaseModel):
    """Пустые параметры для моделей без гиперпараметров."""

    pass


class SeasonalNaiveForecastModel:
    """Сезонная наивная модель прогнозирования."""

    name: str = "seasonal_naive"

    def __init__(self, seasonal_period: int = 12) -> None:
        """Инициализирует модель с заданным сезонным периодом."""
        self.seasonal_period = seasonal_period

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
    ) -> ModelResult:
        """Обучает сезонную наивную модель и возвращает результаты оценки."""
        model = train_seasonal_naive(
            train_df=splits.train,
            val_df=splits.val,
            settings=settings,
            seasonal_period=self.seasonal_period,
        )
        evaluation = evaluate_seasonal_naive(model, splits, settings)
        return ModelResult(
            name=self.name,
            evaluation=evaluation,
            params=_EmptyParams(),
        )
