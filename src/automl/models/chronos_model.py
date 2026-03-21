import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.automl.ts_utils import next_dates
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

logger = logging.getLogger(__name__)


class _ChronosParams(BaseModel):
    """Параметры Chronos-2 для сериализации в ModelResult."""

    pass


def _get_device() -> str:
    """Возвращает 'cuda' если GPU доступен, иначе 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _predict_panel(
    pipeline,
    df: pd.DataFrame,
    horizon: int,
    id_col: str,
    date_col: str,
    target: str,
    freq: str,
) -> pd.DataFrame:
    """Прогнозирует через Chronos-2 predict_df и возвращает медианный прогноз."""
    context_df = df[[id_col, date_col, target]].copy()
    context_df = context_df.rename(columns={id_col: "id", date_col: "timestamp", target: "target"})
    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    context_df["id"] = context_df["id"].astype(str)

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    return pred_df


class ChronosForecastModel(BaseForecastModel):
    """Chronos-2 — foundation model для zero-shot прогнозирования."""

    name: str = "chronos"

    def __init__(self) -> None:
        pass

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        """Zero-shot прогноз на val и test."""
        try:
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting не установлен. Установите: "
                "uv sync --extra neural"
            ) from e

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        cols = settings.columns
        target = cols.main_target
        id_col = cols.id
        date_col = cols.date
        freq = settings.ts.freq

        device = _get_device()
        if progress_fn:
            progress_fn(f"загрузка модели ({device})...", 5.0)

        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map=device,
        )

        splits_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

        # Val прогноз: обучение на train
        if splits.val is not None:
            if cancel_fn and cancel_fn():
                raise ModelCancelledError(self.name)
            if progress_fn:
                progress_fn("прогноз val...", 20.0)

            val_size = splits.val[date_col].nunique()
            pred_df = _predict_panel(
                pipeline, splits.train, val_size, id_col, date_col, target, freq,
            )
            val_preds = _align_chronos_predictions(
                pred_df, splits.val, id_col, date_col,
            )
            splits_data["val"] = (
                splits.val[[id_col, target]].reset_index(drop=True),
                val_preds,
            )

        # Test прогноз: обучение на train+val
        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)
        if progress_fn:
            progress_fn("прогноз test...", 60.0)

        test_size = splits.test[date_col].nunique()
        fit_df = splits.train if splits.val is None else pd.concat(
            [splits.train, splits.val], ignore_index=True,
        )
        pred_df = _predict_panel(
            pipeline, fit_df, test_size, id_col, date_col, target, freq,
        )
        test_preds = _align_chronos_predictions(
            pred_df, splits.test, id_col, date_col,
        )
        splits_data["test"] = (
            splits.test[[id_col, target]].reset_index(drop=True),
            test_preds,
        )

        if progress_fn:
            progress_fn("вычисление метрик...", 95.0)

        results = evaluate_multiple_splits(
            splits_data=splits_data,
            panel_column=id_col,
            target_column=target,
        )
        log_evaluation_results(results)

        return ModelResult(
            name=self.name,
            evaluation=results,
            params=_ChronosParams(),
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Zero-shot прогноз на horizon шагов вперёд."""
        try:
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError("chronos-forecasting не установлен") from e

        cols = settings.columns
        device = _get_device()

        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map=device,
        )

        if on_training_done:
            on_training_done()

        pred_df = _predict_panel(
            pipeline, full_df, horizon,
            cols.id, cols.date, cols.main_target, settings.ts.freq,
        )

        # Извлекаем медианный прогноз
        forecast_rows = []
        for _, row in pred_df.iterrows():
            forecast_rows.append({
                "panel_id": str(row["id"]),
                "date": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d"),
                "forecast": max(0.0, float(row["0.5"])),
            })

        return pd.DataFrame(forecast_rows)


def _align_chronos_predictions(
    pred_df: pd.DataFrame,
    target_split: pd.DataFrame,
    id_col: str,
    date_col: str,
) -> np.ndarray:
    """Выравнивает предсказания Chronos по порядку строк target_split."""
    pred = pred_df.copy()
    pred["id"] = pred["id"].astype(str)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])

    target = target_split[[id_col, date_col]].copy().reset_index(drop=True)
    target["_id"] = target[id_col].astype(str)
    target["_ts"] = pd.to_datetime(target[date_col])

    merged = target.merge(
        pred[["id", "timestamp", "0.5"]],
        left_on=["_id", "_ts"],
        right_on=["id", "timestamp"],
        how="left",
    )

    return merged["0.5"].fillna(0.0).clip(lower=0).values.astype(float)
