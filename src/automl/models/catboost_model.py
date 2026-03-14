import logging

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
