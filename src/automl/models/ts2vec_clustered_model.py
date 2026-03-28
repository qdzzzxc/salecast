"""TS2Vec + CatBoost модель, обучающая отдельный энкодер и регрессор для каждого кластера."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.automl.models.ts2vec_model import (
    TS2VecParameters,
    _add_embedding_features,
    _encode_panels,
    _get_device,
    _reshape_panel_to_3d,
    _train_ts2vec_encoder,
)
from src.automl.ts_utils import next_dates
from src.catboost_utilities.evaluate import _prepare_predictions
from src.catboost_utilities.train import train_catboost
from src.classifical_features import build_ts_features
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, ModelResult, Splits
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

logger = logging.getLogger(__name__)


class TS2VecClusteredForecastModel(BaseForecastModel):
    """TS2Vec encoder + CatBoost, обучаемые отдельно для каждого кластера панелей."""

    name: str = "ts2vec_clustered"

    def __init__(
        self,
        cluster_labels: dict[str, int],
        params: TS2VecParameters | None = None,
    ) -> None:
        self.cluster_labels = cluster_labels
        self.params = params or TS2VecParameters()

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        cols = settings.columns
        target = cols.main_target
        panel_col = cols.id
        date_col = cols.date
        device = _get_device()

        clusters = sorted({int(v) for v in self.cluster_labels.values() if int(v) >= 0})
        n_clusters = len(clusters)

        splits_acc: dict[str, list[tuple[pd.DataFrame, np.ndarray]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        importance_acc: dict[str, list[tuple[float, int]]] = {}  # {fname: [(imp, n_panels)]}
        all_loss_histories: list[list[tuple[int, float]]] = []

        for ci, cluster_id in enumerate(clusters):
            if cancel_fn and cancel_fn():
                raise ModelCancelledError(self.name)

            cluster_pct_base = ci / n_clusters * 95

            if progress_fn:
                progress_fn(f"кластер {ci + 1}/{n_clusters}: подготовка", cluster_pct_base)

            # Панели кластера
            cluster_panels = {
                pid for pid, cid in self.cluster_labels.items() if int(cid) == cluster_id
            }
            if not cluster_panels:
                continue

            # Фильтрация splits по панелям кластера
            def _filter_cluster(df: pd.DataFrame) -> pd.DataFrame:
                return df[df[panel_col].isin(cluster_panels)]

            c_train = _filter_cluster(splits.train)
            c_val = _filter_cluster(splits.val) if splits.val is not None else None
            c_test = _filter_cluster(splits.test)

            if len(c_train) == 0 or len(c_test) == 0:
                logger.warning("Кластер %d: нет данных в train/test, пропускаем", cluster_id)
                continue

            # TS2Vec encoder на данных кластера
            train_array, _ = _reshape_panel_to_3d(c_train, panel_col, target)
            if train_array.shape[0] == 0:
                continue

            if progress_fn:
                progress_fn(
                    f"кластер {ci + 1}/{n_clusters}: обучение TS2Vec ({device})",
                    cluster_pct_base + 2,
                )

            # Оборачиваем progress_fn для передачи эпох с loss в Redis Stream
            def _cluster_progress(
                msg: str,
                pct: float | None = None,
                _ci: int = ci,
                _n: int = n_clusters,
                _base: float = cluster_pct_base,
            ) -> None:
                if progress_fn:
                    # Перемапируем прогресс внутри кластера на общий диапазон
                    overall_pct = _base + (pct or 0) / _n if pct is not None else None
                    progress_fn(msg, overall_pct)

            ts2vec_model, cluster_loss = _train_ts2vec_encoder(
                train_array, self.params, device, _cluster_progress if progress_fn else None
            )
            if cluster_loss:
                all_loss_histories.append(cluster_loss)

            # Concat с _split колонкой
            _SPLIT_COL = "_split"
            parts = [c_train.copy().assign(**{_SPLIT_COL: "train"})]
            if c_val is not None and len(c_val) > 0:
                parts.append(c_val.copy().assign(**{_SPLIT_COL: "val"}))
            parts.append(c_test.copy().assign(**{_SPLIT_COL: "test"}))
            cluster_full = pd.concat(parts, ignore_index=True)

            # Encode + embed + features
            panel_embeddings = _encode_panels(ts2vec_model, cluster_full, panel_col, target)
            cluster_with_emb = _add_embedding_features(
                cluster_full, panel_embeddings, panel_col, self.params.output_dims
            )
            cluster_features = build_ts_features(cluster_with_emb, settings, disable_tqdm=True)

            train_feat = cluster_features[cluster_features[_SPLIT_COL] == "train"].drop(
                columns=[_SPLIT_COL]
            )
            val_feat = None
            if c_val is not None:
                v = cluster_features[cluster_features[_SPLIT_COL] == "val"].drop(
                    columns=[_SPLIT_COL]
                )
                val_feat = v if len(v) > 0 else None
            test_feat = cluster_features[cluster_features[_SPLIT_COL] == "test"].drop(
                columns=[_SPLIT_COL]
            )

            feature_cols = [
                c for c in train_feat.columns if c not in {panel_col, date_col, target}
            ]
            train_clean = train_feat.dropna(subset=feature_cols)

            if len(train_clean) == 0:
                logger.warning("Кластер %d: все train строки NaN, пропускаем", cluster_id)
                continue

            if progress_fn:
                progress_fn(
                    f"кластер {ci + 1}/{n_clusters}: CatBoost",
                    cluster_pct_base + (95 / n_clusters * 0.7),
                )

            cb_params = CatBoostParameters(iterations=500, verbose=0)
            val_for_train = train_clean if val_feat is None else val_feat
            cb_model = train_catboost(train_clean, val_for_train, cb_params, settings)

            # Feature importance: embedding-фичи агрегируются в одну запись
            n_panels = len(cluster_panels)
            emb_imp_sum = 0.0
            for fname, imp in zip(
                cb_model.feature_names_, cb_model.get_feature_importance().tolist()
            ):
                if fname.startswith("ts2vec_emb_"):
                    emb_imp_sum += imp
                else:
                    importance_acc.setdefault(fname, []).append((imp, n_panels))
            if emb_imp_sum > 0:
                importance_acc.setdefault("ts2vec_embeddings", []).append((emb_imp_sum, n_panels))

            # Predictions per split
            cluster_splits = Splits(train=train_feat, val=val_feat, test=test_feat)
            for split_name, split_df in cluster_splits.splits:
                if split_df is None or len(split_df) == 0:
                    continue
                result_df, y_pred = _prepare_predictions(cb_model, split_df, settings, None)
                splits_acc[split_name].append((result_df, y_pred))

        # Aggregate
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

        # Взвешенное среднее по размеру кластера
        feature_importance = []
        for fname, entries in importance_acc.items():
            total_panels = sum(n for _, n in entries)
            if total_panels > 0:
                weighted = sum(imp * n for imp, n in entries) / total_panels
            else:
                weighted = 0.0
            feature_importance.append((fname, weighted))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        if progress_fn:
            progress_fn("готово", 100.0)

        return ModelResult(
            name=self.name,
            evaluation=evaluation,
            params=self.params,
            feature_importance=feature_importance,
            loss_histories=all_loss_histories if all_loss_histories else None,
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        cols = settings.columns
        target = cols.main_target
        panel_col = cols.id
        date_col = cols.date
        device = _get_device()

        clusters = sorted({int(v) for v in self.cluster_labels.values() if int(v) >= 0})
        n_clusters = len(clusters)

        if on_training_done:
            on_training_done()

        all_preds: list[dict] = []

        for step_i, cluster_id in enumerate(clusters):
            if on_forecast_step:
                on_forecast_step(step_i + 1, n_clusters)

            cluster_panels = {
                pid for pid, cid in self.cluster_labels.items() if int(cid) == cluster_id
            }
            cluster_df = full_df[full_df[panel_col].isin(cluster_panels)].copy()
            if len(cluster_df) == 0:
                continue

            # TS2Vec encoder на кластере
            train_array, _ = _reshape_panel_to_3d(cluster_df, panel_col, target)
            if train_array.shape[0] == 0:
                continue
            ts2vec_model, _ = _train_ts2vec_encoder(train_array, self.params, device)
            panel_embeddings = _encode_panels(ts2vec_model, cluster_df, panel_col, target)

            cluster_with_emb = _add_embedding_features(
                cluster_df, panel_embeddings, panel_col, self.params.output_dims
            )
            features_df = build_ts_features(cluster_with_emb, settings, disable_tqdm=True)

            drop_cols = {target, panel_col, date_col}
            feature_cols = [c for c in features_df.columns if c not in drop_cols]
            train_clean = features_df.dropna(subset=feature_cols)

            cb_params = CatBoostParameters(iterations=500, verbose=0)
            cb_model = train_catboost(train_clean, None, cb_params, settings)

            # Iterative forecast
            _FUTURE = "_future"
            running_df = cluster_df.copy()

            for _ in range(horizon):
                next_rows = []
                for pid, group in running_df.groupby(panel_col):
                    nd = next_dates(group[date_col], 1)[0]
                    next_rows.append(
                        {panel_col: pid, date_col: nd, target: 0.0, _FUTURE: True}
                    )

                next_df = pd.DataFrame(next_rows)
                extended = pd.concat(
                    [running_df.assign(**{_FUTURE: False}), next_df],
                    ignore_index=True,
                )

                ext_with_emb = _add_embedding_features(
                    extended, panel_embeddings, panel_col, self.params.output_dims
                )
                feat = build_ts_features(ext_with_emb, settings, disable_tqdm=True)
                future_feat = feat[feat[_FUTURE].fillna(False)][feature_cols].reset_index(
                    drop=True
                )

                if len(future_feat) == 0:
                    break

                preds = np.maximum(cb_model.predict(future_feat.fillna(0)), 0)

                next_df = next_df.reset_index(drop=True)
                next_df[target] = preds

                for i, row in next_df.iterrows():
                    all_preds.append(
                        {
                            "panel_id": str(row[panel_col]),
                            "date": pd.Timestamp(row[date_col]).strftime("%Y-%m-%d"),
                            "forecast": float(preds[i]),
                        }
                    )

                running_df = pd.concat(
                    [running_df, next_df.drop(columns=[_FUTURE])], ignore_index=True
                )

        return pd.DataFrame(all_preds)
