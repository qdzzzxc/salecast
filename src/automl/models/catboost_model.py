import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.catboost_utilities.evaluate import evaluate_catboost
from src.catboost_utilities.train import train_catboost
from src.classifical_features import build_monthly_features
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, ModelResult, Splits
from src.data_processing import scale_panel_splits

logger = logging.getLogger(__name__)


class _TrainingCallback:
    """CatBoost callback для прогресса и отмены."""

    def __init__(
        self,
        progress_fn: ProgressFn | None,
        cancel_fn: CancelFn | None,
        n_iterations: int,
    ) -> None:
        self.progress_fn = progress_fn
        self.cancel_fn = cancel_fn
        self.n_iterations = n_iterations
        self.cancelled = False

    def after_iteration(self, info) -> bool:  # type: ignore[return]
        """Вызывается CatBoost после каждой итерации. Возвращает False чтобы остановить."""
        if self.cancel_fn and self.cancel_fn():
            self.cancelled = True
            return False
        if self.progress_fn:
            pct = min(info.iteration / max(self.n_iterations, 1) * 90, 90.0)
            self.progress_fn(f"итерация {info.iteration}/{self.n_iterations}", pct)
        return True


class CatBoostForecastModel(BaseForecastModel):
    """CatBoost модель прогнозирования временных рядов."""

    name: str = "catboost"

    def __init__(self, params: CatBoostParameters | None = None) -> None:
        """Инициализирует модель с заданными параметрами."""
        self.params = params or CatBoostParameters()

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        """Обучает CatBoost и возвращает результаты оценки."""
        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        if progress_fn:
            progress_fn("построение признаков...", 2.0)

        # Строим фичи на полном df, чтобы lag-фичи на границах train/val/test были корректными.
        # Если строить фичи на каждом сплите отдельно, lag_1 первой точки val = NaN
        # вместо реального последнего значения train → провал на графике.
        _SPLIT_COL = "_split"
        parts = [splits.train.copy().assign(**{_SPLIT_COL: "train"})]
        if splits.val is not None:
            parts.append(splits.val.copy().assign(**{_SPLIT_COL: "val"}))
        parts.append(splits.test.copy().assign(**{_SPLIT_COL: "test"}))

        full_features = build_monthly_features(
            pd.concat(parts, ignore_index=True), settings, disable_tqdm=True
        )

        train_feat = full_features[full_features[_SPLIT_COL] == "train"].drop(columns=[_SPLIT_COL])
        val_feat = (
            full_features[full_features[_SPLIT_COL] == "val"].drop(columns=[_SPLIT_COL])
            if splits.val is not None
            else None
        )
        test_feat = full_features[full_features[_SPLIT_COL] == "test"].drop(columns=[_SPLIT_COL])
        feature_splits = Splits(train=train_feat, val=val_feat, test=test_feat)

        target = settings.columns.main_target
        panel_col = settings.columns.id
        apply_log = settings.preprocessing.apply_log
        should_scale = not settings.downstream.round_predictions

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        if should_scale:
            if progress_fn:
                progress_fn("масштабирование...", 5.0)
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

        callback = _TrainingCallback(progress_fn, cancel_fn, self.params.iterations)

        if progress_fn:
            progress_fn("обучение...", 8.0)

        model = train_catboost(
            train_df=ready_splits.train,
            val_df=ready_splits.val,
            params=self.params,
            settings=settings,
            callbacks=[callback],
        )

        if callback.cancelled:
            raise ModelCancelledError(self.name)

        if progress_fn:
            progress_fn("оценка качества...", 92.0)

        evaluation = evaluate_catboost(
            model=model,
            splits=ready_splits,
            settings=settings,
            scalers=scalers,
        )

        if progress_fn:
            progress_fn("готово", 100.0)

        return ModelResult(
            name=self.name,
            evaluation=evaluation,
            params=self.params,
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Обучает CatBoost на полных данных и строит прогноз на horizon точек вперёд."""
        from src.automl.ts_utils import next_dates

        # Отключаем скейлинг — прогноз строится в исходном масштабе
        forecast_settings = settings.model_copy(
            update={"downstream": settings.downstream.model_copy(update={"scale": False})}
        )
        panel_col = forecast_settings.columns.id
        date_col = forecast_settings.columns.date
        value_col = forecast_settings.columns.main_target

        features_df = build_monthly_features(full_df, forecast_settings, disable_tqdm=True)
        drop_cols = {value_col, panel_col, date_col}
        feature_cols = [c for c in features_df.columns if c not in drop_cols]

        model = train_catboost(
            train_df=features_df,
            val_df=None,
            params=self.params,
            settings=forecast_settings,
        )

        if on_training_done:
            on_training_done()

        _FUTURE = "_future"
        running_df = full_df.copy()
        all_preds = []

        for step_i in range(horizon):
            if on_forecast_step:
                on_forecast_step(step_i + 1, horizon)

            next_rows = []
            for pid, group in running_df.groupby(panel_col):
                nd = next_dates(group[date_col], 1)[0]
                next_rows.append({panel_col: pid, date_col: nd, value_col: 0.0, _FUTURE: True})

            next_df = pd.DataFrame(next_rows)
            extended = pd.concat(
                [running_df.assign(**{_FUTURE: False}), next_df],
                ignore_index=True,
            )
            feat = build_monthly_features(extended, forecast_settings, disable_tqdm=True)
            future_feat = feat[feat[_FUTURE].fillna(False)][feature_cols].reset_index(drop=True)
            preds = np.maximum(model.predict(future_feat), 0)

            next_df = next_df.reset_index(drop=True)
            next_df[value_col] = preds

            for i, row in next_df.iterrows():
                all_preds.append({
                    "panel_id": str(row[panel_col]),
                    "date": pd.Timestamp(row[date_col]).strftime("%Y-%m-%d"),
                    "forecast": float(preds[i]),
                })

            running_df = pd.concat([running_df, next_df.drop(columns=[_FUTURE])], ignore_index=True)

        return pd.DataFrame(all_preds)
