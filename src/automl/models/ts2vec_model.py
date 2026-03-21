import logging
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.automl.ts_utils import next_dates
from src.classifical_features import build_ts_features
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits
from src.evaluation import log_evaluation_results

logger = logging.getLogger(__name__)


class TS2VecParameters(BaseModel):
    output_dims: int = Field(default=320, description="Размерность эмбеддингов")
    hidden_dims: int = Field(default=64, description="Скрытые каналы энкодера")
    depth: int = Field(default=10, description="Количество dilated conv блоков")
    n_epochs: int = Field(default=50, description="Эпохи обучения энкодера")
    lr: float = Field(default=1e-3, description="Learning rate")
    batch_size: int = Field(default=16, description="Batch size")
    downstream: str = Field(default="catboost", description="Downstream: catboost")


def _get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _reshape_panel_to_3d(
    df: pd.DataFrame,
    panel_col: str,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    unique_panels = df[panel_col].unique()
    groups = []
    valid_panels = []
    for pid in unique_panels:
        series = df.loc[df[panel_col] == pid, value_col].values.astype(np.float32)
        groups.append(series)
        valid_panels.append(pid)

    if not groups:
        return np.zeros((0, 0, 1), dtype=np.float32), np.array([])

    max_len = max(len(g) for g in groups)
    result = np.full((len(groups), max_len, 1), np.nan, dtype=np.float32)
    for i, g in enumerate(groups):
        result[i, : len(g), 0] = g

    return result, np.array(valid_panels)


def _encode_panels(
    model,
    df: pd.DataFrame,
    panel_col: str,
    value_col: str,
) -> dict[str, np.ndarray]:
    data, panel_ids = _reshape_panel_to_3d(df, panel_col, value_col)
    if data.shape[0] == 0:
        return {}
    embeddings = model.encode(data, encoding_window="full_series")
    return dict(zip(panel_ids, embeddings))


def _add_embedding_features(
    df: pd.DataFrame,
    panel_embeddings: dict[str, np.ndarray],
    panel_col: str,
    output_dims: int,
) -> pd.DataFrame:
    emb_cols = [f"ts2vec_emb_{i}" for i in range(output_dims)]
    default = np.zeros(output_dims, dtype=np.float32)
    emb_values = np.array([panel_embeddings.get(pid, default) for pid in df[panel_col].values])
    emb_df = pd.DataFrame(emb_values, columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1)


def _train_ts2vec_encoder(
    train_array: np.ndarray,
    params: TS2VecParameters,
    device: str,
    progress_fn: ProgressFn | None = None,
    cancel_fn: CancelFn | None = None,
):
    from src.automl.ts2vec import TS2Vec

    model = TS2Vec(
        input_dims=train_array.shape[-1],
        output_dims=params.output_dims,
        hidden_dims=params.hidden_dims,
        depth=params.depth,
        device=device,
        lr=params.lr,
        batch_size=min(params.batch_size, train_array.shape[0]),
    )

    best_state = None
    best_loss = float("inf")

    def _epoch_callback(m, epoch_loss):
        nonlocal best_state, best_loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = deepcopy(m._net.state_dict())
        if progress_fn:
            pct = min(80.0, 10.0 + 70.0 * m.n_epochs / params.n_epochs)
            progress_fn(f"TS2Vec: epoch {m.n_epochs}/{params.n_epochs}, loss={epoch_loss:.4f}", pct)
        if cancel_fn and cancel_fn():
            raise ModelCancelledError("ts2vec")

    model.after_epoch_callback = _epoch_callback

    model.fit(train_array, n_epochs=params.n_epochs)

    if best_state is not None:
        model._net.load_state_dict(best_state)

    return model


class TS2VecForecastModel(BaseForecastModel):
    """TS2Vec encoder + CatBoost downstream."""

    name: str = "ts2vec"

    def __init__(self, params: TS2VecParameters | None = None) -> None:
        self.params = params or TS2VecParameters()

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        from src.catboost_utilities.evaluate import evaluate_catboost
        from src.catboost_utilities.train import train_catboost
        from src.custom_types import CatBoostParameters

        cols = settings.columns
        target = cols.main_target
        panel_col = cols.id
        date_col = cols.date

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        device = _get_device()
        if progress_fn:
            progress_fn(f"TS2Vec: подготовка данных ({device})...", 5.0)

        train_array, _ = _reshape_panel_to_3d(splits.train, panel_col, target)
        if train_array.shape[0] == 0:
            raise ValueError("Нет данных для обучения TS2Vec")

        ts2vec_model = _train_ts2vec_encoder(
            train_array,
            self.params,
            device,
            progress_fn,
            cancel_fn,
        )

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)
        if progress_fn:
            progress_fn("TS2Vec: кодирование панелей...", 82.0)

        full_df = pd.concat(
            [s for _, s in splits.splits],
            ignore_index=True,
        )
        panel_embeddings = _encode_panels(ts2vec_model, full_df, panel_col, target)

        if progress_fn:
            progress_fn("TS2Vec: построение признаков...", 85.0)

        full_with_emb = _add_embedding_features(
            full_df,
            panel_embeddings,
            panel_col,
            self.params.output_dims,
        )
        full_features = build_ts_features(full_with_emb, settings, disable_tqdm=True)

        train_len = len(splits.train)
        val_len = len(splits.val) if splits.val is not None else 0
        train_feat = full_features.iloc[:train_len]
        val_feat = full_features.iloc[train_len : train_len + val_len] if val_len > 0 else None
        test_feat = full_features.iloc[train_len + val_len :]

        feature_cols = [c for c in train_feat.columns if c not in {panel_col, date_col, target}]
        train_feat_clean = train_feat.dropna(subset=feature_cols)

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)
        if progress_fn:
            progress_fn("TS2Vec: обучение CatBoost...", 88.0)

        cb_params = CatBoostParameters(iterations=500, verbose=0)
        val_feat_for_train = train_feat_clean if val_feat is None else val_feat
        cb_model = train_catboost(train_feat_clean, val_feat_for_train, cb_params, settings)

        if progress_fn:
            progress_fn("TS2Vec: вычисление метрик...", 95.0)

        feature_splits = Splits(
            train=train_feat,
            val=val_feat,
            test=test_feat,
        )
        eval_results = evaluate_catboost(cb_model, feature_splits, settings)
        log_evaluation_results(eval_results)

        importance = list(
            zip(
                feature_cols,
                cb_model.get_feature_importance().tolist(),
            )
        )
        importance.sort(key=lambda x: x[1], reverse=True)

        return ModelResult(
            name=self.name,
            evaluation=eval_results,
            params=self.params,
            feature_importance=importance[:20],
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        from src.catboost_utilities.train import train_catboost
        from src.custom_types import CatBoostParameters

        cols = settings.columns
        target = cols.main_target
        panel_col = cols.id
        date_col = cols.date

        device = _get_device()

        train_array, _ = _reshape_panel_to_3d(full_df, panel_col, target)
        ts2vec_model = _train_ts2vec_encoder(train_array, self.params, device)

        panel_embeddings = _encode_panels(ts2vec_model, full_df, panel_col, target)

        full_with_emb = _add_embedding_features(
            full_df,
            panel_embeddings,
            panel_col,
            self.params.output_dims,
        )
        full_features = build_ts_features(full_with_emb, settings, disable_tqdm=True)

        feature_cols = [c for c in full_features.columns if c not in {panel_col, date_col, target}]
        train_clean = full_features.dropna(subset=feature_cols)

        cb_params = CatBoostParameters(iterations=500, verbose=0)
        cb_model = train_catboost(train_clean, None, cb_params, settings)

        if on_training_done:
            on_training_done()

        panels = full_df[panel_col].unique()
        future_dates = next_dates(full_df[date_col], horizon)
        forecast_rows = []

        for step in range(horizon):
            if on_forecast_step:
                on_forecast_step(step, horizon)

            for pid in panels:
                panel_data = full_df[full_df[panel_col] == pid].copy()
                panel_with_emb = _add_embedding_features(
                    panel_data,
                    panel_embeddings,
                    panel_col,
                    self.params.output_dims,
                )
                panel_features = build_ts_features(panel_with_emb, settings, disable_tqdm=True)

                if len(panel_features) == 0:
                    forecast_rows.append(
                        {
                            "panel_id": str(pid),
                            "date": future_dates[step].strftime("%Y-%m-%d"),
                            "forecast": 0.0,
                        }
                    )
                    continue

                last_row = panel_features.iloc[[-1]]
                feat_values = last_row[feature_cols]

                if feat_values.isna().any(axis=1).iloc[0]:
                    feat_values = feat_values.fillna(0)

                pred = float(cb_model.predict(feat_values)[0])
                pred = max(0.0, pred)

                forecast_rows.append(
                    {
                        "panel_id": str(pid),
                        "date": future_dates[step].strftime("%Y-%m-%d"),
                        "forecast": pred,
                    }
                )

                new_row = panel_data.iloc[[-1]].copy()
                new_row[date_col] = future_dates[step]
                new_row[target] = pred
                full_df = pd.concat([full_df, new_row], ignore_index=True)

        return pd.DataFrame(forecast_rows)
