from pydantic import BaseModel, Field

from src.custom_types import MetricType, ModelType


class AutoMLConfig(BaseModel):
    """Конфигурация AutoML системы."""

    models: list[ModelType] = Field(
        default=["seasonal_naive", "catboost"],
        description="Список моделей для сравнения",
    )
    selection_metric: MetricType = Field(
        default="mape",
        description="Метрика для выбора лучшей модели",
    )
    use_hyperopt: bool = Field(
        default=False,
        description="Использовать Optuna для подбора гиперпараметров CatBoost",
    )
    n_trials: int = Field(
        default=30,
        description="Количество попыток Optuna",
    )
    val_size: int = Field(
        default=3,
        description="Количество точек для val, если val=None в Splits",
    )
