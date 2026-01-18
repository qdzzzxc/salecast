import pandas as pd

from src.configs.settings import Settings
from src.seasonal_naive_utilities.seasonal_naive_model import SeasonalNaiveModel


def train_seasonal_naive(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    settings: Settings,
    seasonal_period: int = 12,
) -> SeasonalNaiveModel:
    """Обучает сезонную наивную модель."""
    model = SeasonalNaiveModel(seasonal_period=seasonal_period)
    model.fit(train_df, settings.columns.id, settings.columns.main_target)
    return model