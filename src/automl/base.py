from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd

from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits

# Колбэк прогресса: (сообщение, процент 0–100 или None)
ProgressFn = Callable[[str, float | None], None]
# Возвращает True если пользователь запросил пропуск модели
CancelFn = Callable[[], bool]


class ModelCancelledError(Exception):
    """Пользователь пропустил модель во время обучения."""

    def __init__(self, model_name: str) -> None:
        super().__init__(f"Модель {model_name!r} пропущена пользователем")
        self.model_name = model_name


class BaseForecastModel(ABC):
    """Базовый класс для всех моделей прогнозирования."""

    name: str

    @abstractmethod
    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        """Обучает модель и возвращает результаты оценки.

        Args:
            splits: Сплиты данных (train / val / test).
            settings: Конфигурация пайплайна.
            progress_fn: Колбэк прогресса — вызывается моделью с сообщением и % выполнения.
            cancel_fn: Колбэк отмены — если возвращает True, модель бросает ModelCancelledError.
        """
        ...
