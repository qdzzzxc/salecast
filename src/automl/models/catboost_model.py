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
        should_scale = not settings.downstream.round_predictions

        if should_scale:
            ready_splits = scale_panel_splits(
                splits=(feature_splits.train, feature_splits.val, feature_splits.test),
                panel_column=panel_col,
                target_columns=[target],
                apply_log=apply_log,
            )
            scalers = ready_splits.scalers
        else:
            ready_splits = feature_splits
            scalers = None

        model = train_catboost(
            train_df=ready_splits.train,
            val_df=ready_splits.val,
            params=self.params,
            settings=settings,
        )

        evaluation = evaluate_catboost(
            model=model,
            splits=ready_splits,
            settings=settings,
            scalers=scalers,
        )

        return ModelResult(
            name=self.name,
            evaluation=evaluation,
            params=self.params,
        )
