import logging

import pandas as pd

from src.automl.base import BaseForecastModel
from src.automl.config import AutoMLConfig
from src.automl.models.catboost_clustered_model import CatBoostClusteredForecastModel
from src.automl.models.catboost_model import CatBoostForecastModel, CatBoostPerPanelForecastModel
from src.automl.models.seasonal_naive_model import SeasonalNaiveForecastModel
from src.automl.models.statsforecast_model import StatsForecastModel
from src.configs.settings import Settings
from src.custom_types import (
    AutoMLResult,
    CatBoostParameters,
    MetricType,
    ModelResult,
    ModelType,
    Splits,
)
from src.model_selection import temporal_panel_split_by_size

logger = logging.getLogger(__name__)


class ModelSelector:
    """Выбирает лучшую модель из заданного списка по заданной метрике."""

    def __init__(self, config: AutoMLConfig) -> None:
        """Инициализирует селектор с конфигурацией."""
        self.config = config

    def run(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
    ) -> AutoMLResult:
        """Запускает AutoML: обучает все модели и выбирает лучшую.

        Args:
            splits (Splits[pd.DataFrame]): Сплиты данных.
            settings (Settings): Конфигурация пайплайна.

        Returns:
            AutoMLResult: Лучшая модель и все результаты.
        """
        if splits.val is None:
            splits = _create_internal_val(splits, settings, self.config.val_size)
            selection_split = "val"
        else:
            selection_split = "val"

        best_cb_params: CatBoostParameters | None = None
        if self.config.use_hyperopt and "catboost" in self.config.models:
            from src.automl.hyperopt import tune_catboost

            logger.info("Запуск гиперпоиска CatBoost с Optuna...")
            best_cb_params = tune_catboost(splits, settings, self.config.n_trials)

        all_results: list[ModelResult] = []

        for model_type in self.config.models:
            logger.info(f"Обучение модели: {model_type}")
            model = _build_model(model_type, best_cb_params)
            result = model.fit_evaluate(splits, settings)
            all_results.append(result)

        best = min(
            all_results,
            key=lambda r: _get_metric_value(r, self.config.selection_metric, selection_split),
        )

        logger.info(
            f"Лучшая модель: {best.name}, "
            f"{self.config.selection_metric}={_get_metric_value(best, self.config.selection_metric, selection_split):.4f}"
        )

        return AutoMLResult(
            best=best,
            all_results=all_results,
            selection_metric=self.config.selection_metric,
            selection_split=selection_split,
        )


def _build_model(
    model_type: ModelType,
    catboost_params: CatBoostParameters | None = None,
    cluster_labels: dict[str, int] | None = None,
) -> BaseForecastModel:
    """Создаёт модель по типу."""
    if model_type == "seasonal_naive":
        return SeasonalNaiveForecastModel()
    if model_type == "catboost":
        return CatBoostForecastModel(params=catboost_params)
    if model_type == "catboost_per_panel":
        return CatBoostPerPanelForecastModel(params=catboost_params)
    if model_type == "catboost_clustered":
        if not cluster_labels:
            raise ValueError("catboost_clustered требует cluster_labels")
        return CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=catboost_params)
    if model_type in (ModelType.autoarima, ModelType.autoets, ModelType.autotheta, ModelType.mstl):
        return StatsForecastModel(model_type=model_type)
    raise ValueError(f"Неизвестный тип модели: {model_type}")


def _get_metric_value(result: ModelResult, metric: MetricType, split: str) -> float:
    """Извлекает значение метрики из результата для заданного сплита."""
    split_evals = [s for s in result.evaluation.splits if s.split_name == split]
    if not split_evals:
        split_evals = [s for s in result.evaluation.splits if s.split_name == "test"]
    if not split_evals:
        return float("inf")

    overall = split_evals[0].overall_metrics
    return float(getattr(overall, metric))


def _create_internal_val(
    splits: Splits[pd.DataFrame],
    settings: Settings,
    val_size: int,
) -> Splits[pd.DataFrame]:
    """Откусывает последние val_size точек из train для создания val сплита."""
    cols = settings.columns
    combined_train = splits.train

    new_splits = temporal_panel_split_by_size(
        df=combined_train,
        panel_column=cols.id,
        time_column=cols.date,
        test_size=val_size,
        val_size=None,
    )

    return Splits(
        train=new_splits.train,
        val=new_splits.test,
        test=splits.test,
    )
