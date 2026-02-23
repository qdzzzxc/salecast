import logging

import pandas as pd

from src.catboost_utilities.evaluate import evaluate_catboost
from src.catboost_utilities.train import train_catboost
from src.classifical_features import build_monthly_features
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, ModelResult, Splits
from src.data_processing import scale_panel_splits

logger = logging.getLogger(__name__)


class CatBoostForecastModel:
    """CatBoost модель прогнозирования временных рядов."""

    name: str = "catboost"

    def __init__(self, params: CatBoostParameters | None = None) -> None:
        """Инициализирует модель с заданными параметрами."""
        self.params = params or CatBoostParameters()

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
    ) -> ModelResult:
        """Обучает CatBoost и возвращает результаты оценки."""
        feature_splits = splits.apply(
            lambda df: build_monthly_features(df, settings, disable_tqdm=True)
        )

        target = settings.columns.main_target
        panel_col = settings.columns.id
        apply_log = settings.preprocessing.apply_log

        scaled = scale_panel_splits(
            splits=(feature_splits.train, feature_splits.val, feature_splits.test),
            panel_column=panel_col,
            target_columns=[target],
            apply_log=apply_log,
        )

        model = train_catboost(
            train_df=scaled.train,
            val_df=scaled.val,
            params=self.params,
            settings=settings,
        )

        evaluation = evaluate_catboost(
            model=model,
            splits=scaled,
            settings=settings,
            scalers=scaled.scalers,
        )

        return ModelResult(
            name=self.name,
            evaluation=evaluation,
            params=self.params,
        )
