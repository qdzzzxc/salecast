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
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

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


class CatBoostPerPanelForecastModel(BaseForecastModel):
    """CatBoost модель, обучающая отдельный регрессор для каждой панели."""

    name: str = "catboost_per_panel"

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
        """Обучает отдельный CatBoost на каждую панель и агрегирует метрики."""
        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        panel_col = settings.columns.id
        target = settings.columns.main_target
        apply_log = settings.preprocessing.apply_log
        should_scale = not settings.downstream.round_predictions

        # Строим фичи на полном df — корректные лаги на границах сплитов
        _SPLIT_COL = "_split"
        parts = [splits.train.copy().assign(**{_SPLIT_COL: "train"})]
        if splits.val is not None:
            parts.append(splits.val.copy().assign(**{_SPLIT_COL: "val"}))
        parts.append(splits.test.copy().assign(**{_SPLIT_COL: "test"}))

        full_features = build_monthly_features(
            pd.concat(parts, ignore_index=True), settings, disable_tqdm=True
        )

        panels = splits.train[panel_col].unique()
        n_panels = len(panels)
        # splits_data: накапливаем (df_с_таргетом, y_pred) по каждому сплиту
        splits_acc: dict[str, list[tuple[pd.DataFrame, np.ndarray]]] = {
            "train": [], "val": [], "test": []
        }

        for i, panel_id in enumerate(panels):
            if cancel_fn and cancel_fn():
                raise ModelCancelledError(self.name)

            if progress_fn:
                progress_fn(f"панель {i + 1}/{n_panels}", i / n_panels * 95)

            panel_mask = full_features[panel_col] == panel_id

            train_feat = full_features[
                panel_mask & (full_features[_SPLIT_COL] == "train")
            ].drop(columns=[_SPLIT_COL])

            val_feat = None
            if splits.val is not None:
                v = full_features[
                    panel_mask & (full_features[_SPLIT_COL] == "val")
                ].drop(columns=[_SPLIT_COL])
                val_feat = v if len(v) > 0 else None

            test_feat = full_features[
                panel_mask & (full_features[_SPLIT_COL] == "test")
            ].drop(columns=[_SPLIT_COL])

            panel_splits: Splits[pd.DataFrame] = Splits(
                train=train_feat, val=val_feat, test=test_feat
            )

            if should_scale:
                ready = scale_panel_splits(
                    splits=(panel_splits.train, panel_splits.val, panel_splits.test),
                    panel_column=panel_col,
                    target_columns=[target],
                    apply_log=apply_log,
                )
                scalers = ready.scalers
            else:
                ready = panel_splits
                scalers = None

            model = train_catboost(
                train_df=ready.train,
                val_df=ready.val,
                params=self.params,
                settings=settings,
            )

            for split_name, split_df in ready.splits:
                if split_df is None or len(split_df) == 0:
                    continue
                from src.catboost_utilities.evaluate import _prepare_predictions
                result_df, y_pred = _prepare_predictions(model, split_df, settings, scalers)
                splits_acc[split_name].append((result_df, y_pred))

        splits_data = {
            split_name: (
                pd.concat([p[0] for p in preds], ignore_index=True),
                np.concatenate([p[1] for p in preds]),
            )
            for split_name, preds in splits_acc.items()
            if preds
        }

        evaluation = evaluate_multiple_splits(
            splits_data=splits_data,
            panel_column=panel_col,
            target_column=target,
        )
        log_evaluation_results(evaluation)

        if progress_fn:
            progress_fn("готово", 100.0)

        return ModelResult(name=self.name, evaluation=evaluation, params=self.params)

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Обучает отдельный CatBoost на каждую панель и строит прогноз на horizon шагов."""
        from src.automl.ts_utils import next_dates

        forecast_settings = settings.model_copy(
            update={"downstream": settings.downstream.model_copy(update={"scale": False})}
        )
        panel_col = forecast_settings.columns.id
        date_col = forecast_settings.columns.date
        value_col = forecast_settings.columns.main_target

        panels = full_df[panel_col].unique()
        n_panels = len(panels)

        if on_training_done:
            on_training_done()

        all_preds: list[dict] = []

        for panel_i, panel_id in enumerate(panels):
            if on_forecast_step:
                on_forecast_step(panel_i + 1, n_panels)

            panel_df = full_df[full_df[panel_col] == panel_id].copy()

            features_df = build_monthly_features(panel_df, forecast_settings, disable_tqdm=True)
            drop_cols = {value_col, panel_col, date_col}
            feature_cols = [c for c in features_df.columns if c not in drop_cols]

            model = train_catboost(
                train_df=features_df,
                val_df=None,
                params=self.params,
                settings=forecast_settings,
            )

            _FUTURE = "_future"
            running_df = panel_df.copy()

            for step_i in range(horizon):
                nd = next_dates(running_df[date_col], 1)[0]
                next_row = pd.DataFrame(
                    [{panel_col: panel_id, date_col: nd, value_col: 0.0, _FUTURE: True}]
                )
                extended = pd.concat(
                    [running_df.assign(**{_FUTURE: False}), next_row],
                    ignore_index=True,
                )
                feat = build_monthly_features(extended, forecast_settings, disable_tqdm=True)
                future_feat = feat[feat[_FUTURE].fillna(False)][feature_cols].reset_index(drop=True)
                pred = float(np.maximum(model.predict(future_feat)[0], 0))

                all_preds.append({
                    "panel_id": str(panel_id),
                    "date": pd.Timestamp(nd).strftime("%Y-%m-%d"),
                    "forecast": pred,
                })

                next_row = next_row.drop(columns=[_FUTURE])
                next_row[value_col] = pred
                running_df = pd.concat([running_df, next_row], ignore_index=True)

        return pd.DataFrame(all_preds)
