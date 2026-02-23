from typing import Protocol

import pandas as pd

from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits


class ForecastModel(Protocol):
    """Протокол для всех моделей прогнозирования."""

    name: str

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
    ) -> ModelResult:
        """Обучает модель и возвращает результаты оценки."""
        ...
