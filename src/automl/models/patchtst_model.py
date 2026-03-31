import logging
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.automl.base import BaseForecastModel, CancelFn, ModelCancelledError, ProgressFn
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

logger = logging.getLogger(__name__)

_REPORT_EVERY = 20  # репортим progress каждые N шагов


def _patch_logging_proxy() -> None:
    """neuralforecast проверяет sys.stdout.encoding при импорте.

    Celery подменяет sys.stdout/stderr на LoggingProxy без атрибута encoding.
    Добавляем атрибут если он отсутствует, чтобы импорт не падал.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and not hasattr(stream, "encoding"):
            try:
                stream.encoding = "utf-8"  # type: ignore[misc]
            except (AttributeError, TypeError):
                pass


class PatchTSTParameters(BaseModel):
    """Параметры PatchTST через neuralforecast."""

    input_size: int = Field(default=24, description="Длина lookback окна (шагов)")
    patch_len: int = Field(default=16, description="Длина патча")
    stride: int = Field(default=8, description="Шаг между патчами")
    hidden_size: int = Field(default=64, description="Размер скрытого слоя")
    n_heads: int = Field(default=4, description="Число голов внимания")
    encoder_layers: int = Field(default=3, description="Число слоёв трансформера")
    batch_size: int = Field(default=16, description="Серий в батче")
    windows_batch_size: int = Field(default=256, description="Окон в батче")
    max_steps: int = Field(default=200, description="Шагов обучения")
    learning_rate: float = Field(default=1e-3, description="Learning rate")

    model_config = {"frozen": True}


def _get_device() -> str:
    """Возвращает 'cuda' если GPU доступен, иначе 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _to_nixtla(df: pd.DataFrame, id_col: str, date_col: str, target: str) -> pd.DataFrame:
    """Конвертирует panel DataFrame в формат neuralforecast [unique_id, ds, y]."""
    out = df[[id_col, date_col, target]].copy()
    out = out.rename(columns={id_col: "unique_id", date_col: "ds", target: "y"})
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _horizon(df: pd.DataFrame, date_col: str) -> int:
    """Количество уникальных дат в сплите."""
    return int(df[date_col].nunique())


def _effective_params(params: PatchTSTParameters, season_length: int) -> dict:
    """Вычисляет эффективные гиперпараметры с учётом сезонности.

    input_size  ≥ 3 × season_length — минимум 3 полных цикла в lookback окне.
    patch_len   = season_length     — один патч = один сезонный период.
    stride      = season_length // 2 — перекрытие 50%.
    """
    sl = max(season_length, 1)
    input_size = max(params.input_size, 3 * sl)
    patch_len = min(params.patch_len, sl) if sl < params.patch_len else sl
    stride = max(patch_len // 2, 1)
    return {
        "input_size": input_size,
        "patch_len": patch_len,
        "stride": stride,
    }


def _build_nf(
    h: int,
    params: PatchTSTParameters,
    freq: str,
    season_length: int = 1,
    callbacks: list | None = None,
):
    """Создаёт NeuralForecast(PatchTST(...)).

    В neuralforecast 3.x trainer kwargs передаются напрямую как **kwargs модели.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST

    accelerator = "gpu" if _get_device() == "cuda" else "cpu"
    ep = _effective_params(params, season_length)

    model = PatchTST(
        h=h,
        input_size=ep["input_size"],
        patch_len=ep["patch_len"],
        stride=ep["stride"],
        hidden_size=params.hidden_size,
        n_heads=params.n_heads,
        encoder_layers=params.encoder_layers,
        batch_size=params.batch_size,
        windows_batch_size=params.windows_batch_size,
        max_steps=params.max_steps,
        learning_rate=params.learning_rate,
        scaler_type="standard",
        start_padding_enabled=True,
        # trainer kwargs передаются как **kwargs в neuralforecast 3.x
        accelerator=accelerator,
        enable_progress_bar=False,
        enable_model_summary=False,
        **{"callbacks": callbacks} if callbacks else {},
    )
    return NeuralForecast(models=[model], freq=freq)


def _make_progress_callback(
    params: PatchTSTParameters,
    progress_fn: ProgressFn,
    pct_start: float,
    pct_end: float,
    cancel_fn: CancelFn | None,
    model_name: str,
    epoch_offset: int = 0,
):
    """Создаёт Lightning callback для репорта прогресса.

    epoch_offset смещает номера шагов чтобы val и test образовали одну непрерывную линию.
    """
    try:
        from pytorch_lightning.callbacks import Callback
    except ImportError:
        return None

    max_steps = params.max_steps
    pct_range = pct_end - pct_start
    best_loss: list[float] = []

    class _CB(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if cancel_fn and cancel_fn():
                raise ModelCancelledError(model_name)
            step = trainer.global_step
            if step % _REPORT_EVERY != 0 and step != max_steps:
                return
            loss = None
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = float(outputs["loss"])
            elif hasattr(outputs, "item"):
                loss = float(outputs)
            pct = pct_start + min(step / max_steps, 1.0) * pct_range
            global_step = step + epoch_offset
            total_steps = max_steps * (2 if epoch_offset else 1)
            msg = f"PatchTST: step {global_step}/{total_steps}"
            if loss is not None:
                if not best_loss or loss < best_loss[0]:
                    best_loss[:] = [loss]
                msg += f"||loss={loss:.6f}||best_loss={best_loss[0]:.6f}||epoch={global_step}"
            progress_fn(msg, pct)

    return _CB()


def _align(
    pred_df: pd.DataFrame,
    target_split: pd.DataFrame,
    id_col: str,
    date_col: str,
) -> np.ndarray:
    """Выравнивает прогнозы по порядку строк target_split."""
    pred = pred_df[["unique_id", "ds", "PatchTST"]].copy()
    pred["unique_id"] = pred["unique_id"].astype(str)
    pred["ds"] = pd.to_datetime(pred["ds"])

    target = target_split[[id_col, date_col]].copy().reset_index(drop=True)
    target["_uid"] = target[id_col].astype(str)
    target["_ds"] = pd.to_datetime(target[date_col])

    merged = target.merge(
        pred,
        left_on=["_uid", "_ds"],
        right_on=["unique_id", "ds"],
        how="left",
    )
    return merged["PatchTST"].fillna(0.0).clip(lower=0).values.astype(float)


class PatchTSTForecastModel(BaseForecastModel):
    """PatchTST — patch-based трансформер для прогнозирования временных рядов."""

    name: str = "patchtst"

    def __init__(self, params: PatchTSTParameters | None = None) -> None:
        self.params = params or PatchTSTParameters()

    def fit_evaluate(
        self,
        splits: Splits[pd.DataFrame],
        settings: Settings,
        progress_fn: ProgressFn | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> ModelResult:
        """Обучает PatchTST и оценивает на val/test."""
        _patch_logging_proxy()
        try:
            from neuralforecast import NeuralForecast  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "neuralforecast не установлен. Установите: uv sync --extra neural"
            ) from e

        cols = settings.columns
        id_col, date_col, target = cols.id, cols.date, cols.main_target
        freq = settings.ts.freq or "MS"
        season_length = settings.ts.season_length or 1

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        if progress_fn:
            progress_fn(f"PatchTST: подготовка данных ({_get_device()})...", 3.0)

        splits_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

        # --- Val ---
        if splits.val is not None:
            if cancel_fn and cancel_fn():
                raise ModelCancelledError(self.name)

            h_val = _horizon(splits.val, date_col)
            train_nixtla = _to_nixtla(splits.train, id_col, date_col, target)

            if progress_fn:
                progress_fn("PatchTST: обучение (val)...", 5.0)
            cb = (
                _make_progress_callback(self.params, progress_fn, 5.0, 42.0, cancel_fn, self.name)
                if progress_fn
                else None
            )
            nf_val = _build_nf(h_val, self.params, freq, season_length, [cb] if cb else None)
            nf_val.fit(df=train_nixtla)
            if progress_fn:
                progress_fn("PatchTST: прогноз val...", 43.0)
            pred_val = nf_val.predict(df=train_nixtla)

            splits_data["val"] = (
                splits.val[[id_col, target]].reset_index(drop=True),
                _align(pred_val, splits.val, id_col, date_col),
            )

        if cancel_fn and cancel_fn():
            raise ModelCancelledError(self.name)

        # --- Test ---
        h_test = _horizon(splits.test, date_col)
        fit_df = (
            splits.train
            if splits.val is None
            else pd.concat([splits.train, splits.val], ignore_index=True)
        )
        train_test_nixtla = _to_nixtla(fit_df, id_col, date_col, target)

        pct_start = 46.0 if splits.val is not None else 5.0
        if progress_fn:
            progress_fn("PatchTST: обучение (test)...", pct_start)
        epoch_offset = self.params.max_steps if splits.val is not None else 0
        cb_test = (
            _make_progress_callback(
                self.params, progress_fn, pct_start, 90.0, cancel_fn, self.name, epoch_offset
            )
            if progress_fn
            else None
        )
        nf_test = _build_nf(h_test, self.params, freq, season_length, [cb_test] if cb_test else None)
        nf_test.fit(df=train_test_nixtla)
        if progress_fn:
            progress_fn("PatchTST: прогноз test...", 91.0)
        pred_test = nf_test.predict(df=train_test_nixtla)

        # Извлекаем loss history из train_trajectories модели (шаг, loss)
        raw_trajectories = getattr(nf_test.models[0], "train_trajectories", [])
        loss_history: list[tuple[int, float]] = [(int(s), float(lv)) for s, lv in raw_trajectories]

        splits_data["test"] = (
            splits.test[[id_col, target]].reset_index(drop=True),
            _align(pred_test, splits.test, id_col, date_col),
        )


        if progress_fn:
            progress_fn("PatchTST: вычисление метрик...", 95.0)

        results = evaluate_multiple_splits(
            splits_data=splits_data,
            panel_column=id_col,
            target_column=target,
        )
        log_evaluation_results(results)

        return ModelResult(
            name=self.name,
            evaluation=results,
            params=self.params,
            loss_history=loss_history if loss_history else None,
        )

    def forecast_future(
        self,
        full_df: pd.DataFrame,
        horizon: int,
        settings: Settings,
        on_training_done: Callable[[], None] | None = None,
        on_forecast_step: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Обучает на всех данных и прогнозирует horizon шагов вперёд."""
        _patch_logging_proxy()
        try:
            from neuralforecast import NeuralForecast  # noqa: F401
        except ImportError as e:
            raise ImportError("neuralforecast не установлен") from e

        cols = settings.columns
        id_col, date_col, target = cols.id, cols.date, cols.main_target
        freq = settings.ts.freq or "MS"
        season_length = settings.ts.season_length or 1

        full_nixtla = _to_nixtla(full_df, id_col, date_col, target)
        nf = _build_nf(horizon, self.params, freq, season_length)
        nf.fit(df=full_nixtla)

        if on_training_done:
            on_training_done()

        pred_df = nf.predict(df=full_nixtla)

        forecast_rows = []
        for _, row in pred_df.iterrows():
            forecast_rows.append(
                {
                    "panel_id": str(row["unique_id"]),
                    "date": pd.Timestamp(row["ds"]).strftime("%Y-%m-%d"),
                    "forecast": max(0.0, float(row["PatchTST"])),
                }
            )
        return pd.DataFrame(forecast_rows)
