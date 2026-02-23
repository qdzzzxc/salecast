import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

logger = logging.getLogger(__name__)

StatsForecastModelType = Literal["autoarima", "autoets", "autotheta"]

_MODEL_COL_MAP = {
    "autoarima": "AutoARIMA",
    "autoets": "AutoETS",
    "autotheta": "AutoTheta",
}


class _EmptyParams(BaseModel):
    """Пустые параметры для моделей без гиперпараметров."""

    pass


class StatsForecastModel:
    """Модели StatsForecast: AutoARIMA, AutoETS, AutoTheta."""

    def __init__(self, model_type: StatsForecastModelType) -> None:
        """Инициализирует модель заданного типа."""
        self.model_type = model_type
        self.name = model_type

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
    ) -> ModelResult:
        """Обучает statsforecast модель и возвращает результаты оценки."""
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
        except ImportError as e:
            raise ImportError(
                "statsforecast не установлен. Установите его: uv add statsforecast"
            ) from e

        _sf_model_cls = {"autoarima": AutoARIMA, "autoets": AutoETS, "autotheta": AutoTheta}
        cols = settings.columns
        target = cols.main_target
        id_col = cols.id
        date_col = cols.date

        splits_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

        if splits.val is not None:
            val_size = splits.val[date_col].nunique()
            sf_val = StatsForecast(
                models=[_sf_model_cls[self.model_type](season_length=12)],
                freq="MS",
                verbose=False,
            )
            train_sf = _to_sf_format(splits.train, id_col, date_col, target)
            sf_val.fit(train_sf)
            val_forecast = sf_val.predict(h=val_size)
            val_preds = _align_predictions(val_forecast, _MODEL_COL_MAP[self.model_type], splits.val, id_col, date_col)
            splits_data["val"] = (splits.val[[id_col, target]].reset_index(drop=True), val_preds)

        test_size = splits.test[date_col].nunique()
        fit_df = splits.train if splits.val is None else pd.concat([splits.train, splits.val], ignore_index=True)
        sf_test = StatsForecast(
            models=[_sf_model_cls[self.model_type](season_length=12)],
            freq="MS",
            verbose=False,
        )
        fit_sf = _to_sf_format(fit_df, id_col, date_col, target)
        sf_test.fit(fit_sf)
        test_forecast = sf_test.predict(h=test_size)
        test_preds = _align_predictions(test_forecast, _MODEL_COL_MAP[self.model_type], splits.test, id_col, date_col)
        splits_data["test"] = (splits.test[[id_col, target]].reset_index(drop=True), test_preds)

        results = evaluate_multiple_splits(
            splits_data=splits_data,
            panel_column=id_col,
            target_column=target,
        )
        log_evaluation_results(results)

        return ModelResult(
            name=self.name,
            evaluation=results,
            params=_EmptyParams(),
        )


def _to_sf_format(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Конвертирует датафрейм в формат statsforecast (unique_id, ds, y)."""
    result = df[[id_col, date_col, target_col]].copy()
    result = result.rename(columns={id_col: "unique_id", date_col: "ds", target_col: "y"})
    result["ds"] = pd.to_datetime(result["ds"])
    result["unique_id"] = result["unique_id"].astype(str)
    return result.reset_index(drop=True)


def _align_predictions(
    forecast_df: pd.DataFrame,
    pred_col: str,
    target_split: pd.DataFrame,
    id_col: str,
    date_col: str,
) -> np.ndarray:
    """Выравнивает предсказания statsforecast по порядку строк target_split."""
    forecast_df = forecast_df.copy()
    forecast_df["unique_id"] = forecast_df["unique_id"].astype(str)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    target_df = target_split[[id_col, date_col]].copy().reset_index(drop=True)
    target_df["unique_id"] = target_df[id_col].astype(str)
    target_df["ds"] = pd.to_datetime(target_df[date_col])

    merged = target_df.merge(
        forecast_df[["unique_id", "ds", pred_col]],
        on=["unique_id", "ds"],
        how="left",
    )

    return merged[pred_col].fillna(0.0).values.astype(float)
