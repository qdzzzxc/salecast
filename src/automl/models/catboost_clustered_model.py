import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.catboost_utilities.evaluate import _prepare_predictions
from src.catboost_utilities.train import train_catboost
from src.classifical_features import build_ts_features
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, ModelResult
from src.data_processing import scale_panel_splits
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

logger = logging.getLogger(__name__)


class CatBoostClusteredForecastModel(BaseForecastModel):
    """CatBoost модель, обучающая отдельный регрессор для каждого кластера панелей."""

    name: str = "catboost_clustered"

    def __init__(
        self,
        cluster_labels: dict[str, int],
        params: CatBoostParameters | None = None,
    ) -> None:
        """Инициализирует модель.

        Args:
            cluster_labels: словарь panel_id → cluster_id.
            params: параметры CatBoost.
        """
        self.cluster_labels = cluster_labels
        self.params = params or CatBoostParameters()

    def fit_evaluate(
        self,
        splits,
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        """Обучает CatBoost на каждый кластер и агрегирует метрики."""
        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        panel_col = settings.columns.id
        target = settings.columns.main_target
        apply_log = settings.preprocessing.apply_log
        should_scale = not settings.downstream.round_predictions

        # Строим фичи на полном concat — корректные лаги на границах сплитов
        _SPLIT_COL = "_split"
        parts = [splits.train.copy().assign(**{_SPLIT_COL: "train"})]
        if splits.val is not None:
            parts.append(splits.val.copy().assign(**{_SPLIT_COL: "val"}))
        parts.append(splits.test.copy().assign(**{_SPLIT_COL: "test"}))
        full_features = build_ts_features(
            pd.concat(parts, ignore_index=True), settings, disable_tqdm=True
        )

        clusters = sorted({int(v) for v in self.cluster_labels.values() if int(v) >= 0})
        n_clusters = len(clusters)
        splits_acc: dict[str, list[tuple[pd.DataFrame, np.ndarray]]] = {
            "train": [], "val": [], "test": []
        }
        importance_acc: dict[str, list[float]] = {}

        for i, cluster_id in enumerate(clusters):
            if cancel_fn and cancel_fn():
                raise ModelCancelledError(self.name)

            if progress_fn:
                progress_fn(f"кластер {i + 1}/{n_clusters}", i / n_clusters * 95)

            # Панели этого кластера
            cluster_panels = {pid for pid, cid in self.cluster_labels.items() if int(cid) == cluster_id}
            if not cluster_panels:
                continue

            panel_mask = full_features[panel_col].astype(str).isin(cluster_panels)

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

            if len(train_feat) == 0 or len(test_feat) == 0:
                logger.warning("Кластер %d: нет данных в train/test, пропускаем", cluster_id)
                continue

            from src.custom_types import Splits
            panel_splits = Splits(train=train_feat, val=val_feat, test=test_feat)

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

            for fname, imp in zip(model.feature_names_, model.get_feature_importance().tolist()):
                importance_acc.setdefault(fname, []).append(imp)

            for split_name, split_df in ready.splits:
                if split_df is None or len(split_df) == 0:
                    continue
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

        feature_importance = sorted(
            [(f, sum(vs) / len(vs)) for f, vs in importance_acc.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        if progress_fn:
            progress_fn("готово", 100.0)

        return ModelResult(
            name=self.name, evaluation=evaluation, params=self.params,
            feature_importance=feature_importance,
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Обучает CatBoost на каждый кластер и строит прогноз на horizon шагов."""
        from src.automl.ts_utils import next_dates

        forecast_settings = settings.model_copy(
            update={"downstream": settings.downstream.model_copy(update={"scale": False})}
        )
        panel_col = forecast_settings.columns.id
        date_col = forecast_settings.columns.date
        value_col = forecast_settings.columns.main_target

        clusters = sorted({int(v) for v in self.cluster_labels.values() if int(v) >= 0})
        n_clusters = len(clusters)

        if on_training_done:
            on_training_done()

        all_preds: list[dict] = []

        for step_i, cluster_id in enumerate(clusters):
            if on_forecast_step:
                on_forecast_step(step_i + 1, n_clusters)

            cluster_panels = {pid for pid, cid in self.cluster_labels.items() if int(cid) == cluster_id}
            cluster_df = full_df[full_df[panel_col].astype(str).isin(cluster_panels)].copy()
            if len(cluster_df) == 0:
                continue

            features_df = build_ts_features(cluster_df, forecast_settings, disable_tqdm=True)
            drop_cols = {value_col, panel_col, date_col}
            feature_cols = [c for c in features_df.columns if c not in drop_cols]

            model = train_catboost(
                train_df=features_df,
                val_df=None,
                params=self.params,
                settings=forecast_settings,
            )

            _FUTURE = "_future"
            running_df = cluster_df.copy()

            for _ in range(horizon):
                next_rows = []
                for pid, group in running_df.groupby(panel_col):
                    nd = next_dates(group[date_col], 1)[0]
                    next_rows.append({panel_col: pid, date_col: nd, value_col: 0.0, _FUTURE: True})

                next_df = pd.DataFrame(next_rows)
                extended = pd.concat(
                    [running_df.assign(**{_FUTURE: False}), next_df],
                    ignore_index=True,
                )
                feat = build_ts_features(extended, forecast_settings, disable_tqdm=True)
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
