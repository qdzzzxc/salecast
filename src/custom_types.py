import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

AggMethod: TypeAlias = Literal["sum", "mean", "first", "last", "min", "max"] | Callable
ClipBounds: TypeAlias = dict[str, dict[int | str, tuple[float, float]]]
PanelScalers: TypeAlias = dict[str, dict[int, StandardScaler]]
MetricType: TypeAlias = Literal["mape", "rmse", "mae", "r2"]
ModelType: TypeAlias = Literal["seasonal_naive", "catboost", "catboost_per_panel", "catboost_clustered", "autoarima", "autoets", "autotheta", "mstl"]
QualityStatus: TypeAlias = Literal["green", "yellow", "red"]

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Splits(Generic[T]):
    """Базовый класс для train/val/test сплитов."""

    train: T
    val: T | None
    test: T

    def apply(self, func: Callable[[T], R]) -> "Splits[R]":
        """Применяет функцию к каждому сплиту."""
        return Splits(
            train=func(self.train),
            val=func(self.val) if self.val is not None else None,
            test=func(self.test),
        )

    @property
    def splits(self) -> list[tuple[str, T]]:
        """Возвращает список непустых сплитов с их названиями."""
        result = [("train", self.train)]
        if self.val is not None:
            result.append(("val", self.val))
        result.append(("test", self.test))
        return result


@dataclass
class SplitsWithoutTrain(Generic[T]):
    """Сплиты без train для foundation моделей, которые не требуют обучения.

    Используется для оценки предобученных моделей (Chronos, TimesFM и др.),
    где train split служит только контекстом для генерации предсказаний,
    а метрики имеют смысл только на val/test.
    """

    val: T | None
    test: T

    def apply(self, func: Callable[[T], R]) -> "SplitsWithoutTrain[R]":
        """Применяет функцию к каждому сплиту."""
        return SplitsWithoutTrain(
            val=func(self.val) if self.val is not None else None,
            test=func(self.test),
        )

    @property
    def splits(self) -> list[tuple[str, T]]:
        """Возвращает список непустых сплитов с их названиями."""
        result = []
        if self.val is not None:
            result.append(("val", self.val))
        result.append(("test", self.test))
        return result

    @classmethod
    def from_splits(cls, splits: Splits[T]) -> "SplitsWithoutTrain[T]":
        """Конвертирует Splits в SplitsWithoutTrain, отбрасывая train."""
        return cls(val=splits.val, test=splits.test)


@dataclass
class ScaledSplits(Splits[pd.DataFrame]):
    """Результат скейлинга сплитов."""

    scalers: PanelScalers


@dataclass
class ClippedSplits(Splits[pd.DataFrame]):
    """Результат клиппинга сплитов."""

    bounds: ClipBounds


@dataclass
class FeatureResult(Splits[pd.DataFrame]):
    """Результат пайплайна подготовки признаков."""

    original_df: pd.DataFrame
    scalers: PanelScalers
    clip_bounds: ClipBounds


@dataclass
class EvaluationState:
    """Состояние для отслеживания лучшей модели."""

    val_loss_history: list[float] = field(default_factory=list)
    test_loss_history: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_model_state: dict | None = None


@dataclass
class SplitRange:
    """Временной промежуток для сплита."""

    start: dt.date
    end: dt.date


@dataclass
class RegressionMetrics:
    """Метрики регрессии."""

    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    explained_variance: float
    nrmse: float
    nmae: float
    cv_rmse: float
    nrmse_std: float

    def to_dict(self) -> dict[str, float]:
        """Преобразует метрики в словарь."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "explained_variance": self.explained_variance,
            "nrmse": self.nrmse,
            "nmae": self.nmae,
            "cv_rmse": self.cv_rmse,
            "nrmse_std": self.nrmse_std,
        }

    def get_scale_invariant_metrics(self) -> dict[str, float]:
        """Возвращает только метрики, инвариантные к масштабу."""
        return {
            "r2": self.r2,
            "mape": self.mape,
            "explained_variance": self.explained_variance,
            "nrmse": self.nrmse,
            "nmae": self.nmae,
            "cv_rmse": self.cv_rmse,
            "nrmse_std": self.nrmse_std,
        }


@dataclass
class PanelMetrics:
    """Метрики для одной панели."""

    panel_id: int | str
    split: str
    metrics: RegressionMetrics
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class SplitEvaluation:
    """Результаты оценки для одного сплита."""

    split_name: str
    overall_metrics: RegressionMetrics
    panel_metrics: list[PanelMetrics]
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class EvaluationResults:
    """Результаты оценки модели на всех сплитах."""

    splits: list[SplitEvaluation]

    def get_overall_metrics_df(self) -> pd.DataFrame:
        """Возвращает общие метрики в виде DataFrame."""
        data = []
        for split_eval in self.splits:
            row = {"split": split_eval.split_name}
            row.update(split_eval.overall_metrics.to_dict())
            data.append(row)
        return pd.DataFrame(data)

    def get_panel_metrics_df(self) -> pd.DataFrame:
        """Возвращает метрики по панелям в виде DataFrame."""
        data = []
        for split_eval in self.splits:
            for panel_metric in split_eval.panel_metrics:
                row = {
                    "split": panel_metric.split,
                    "panel_id": panel_metric.panel_id,
                }
                row.update(panel_metric.metrics.to_dict())
                data.append(row)
        return pd.DataFrame(data)

    def get_predictions(self) -> dict[str, np.ndarray]:
        """Возвращает предсказания в виде словаря."""
        predictions = {}
        for split_eval in self.splits:
            predictions[f"{split_eval.split_name}_true"] = split_eval.y_true
            predictions[f"{split_eval.split_name}_pred"] = split_eval.y_pred
        return predictions


@dataclass
class PanelPredictions:
    """Предсказания для одной панели."""

    panel_id: int | str
    y_true: np.ndarray
    y_pred: np.ndarray
    split: str


@dataclass
class FiltrationStepReport:
    """Отчёт об одном шаге фильтрации."""

    step: str
    reason: str
    dropped_ids: set[int | str]


@dataclass
class FiltrationResult:
    """Результат фильтрации с отслеживанием причин отфильтровки."""

    df: pd.DataFrame
    steps: list[FiltrationStepReport]

    def to_report_df(self) -> pd.DataFrame:
        """Возвращает таблицу с отфильтрованными панелями и причинами."""
        rows: list[dict[str, int | str]] = []
        for step_report in self.steps:
            rows.extend(
                {
                    "panel_id": panel_id,
                    "step": step_report.step,
                    "reason": step_report.reason,
                }
                for panel_id in step_report.dropped_ids
            )
        return pd.DataFrame(rows, columns=["panel_id", "step", "reason"])

    @property
    def total_dropped(self) -> int:
        """Возвращает общее количество уникальных отфильтрованных панелей."""
        all_dropped: set[int | str] = set()
        for step_report in self.steps:
            all_dropped |= step_report.dropped_ids
        return len(all_dropped)

    def summary(self) -> dict[str, int]:
        """Возвращает сводку по шагам: имя шага -> количество отфильтрованных."""
        return {step_report.step: len(step_report.dropped_ids) for step_report in self.steps}


@dataclass
class ModelResult:
    """Результат обучения и оценки одной модели."""

    name: str
    evaluation: "EvaluationResults"
    params: BaseModel
    feature_importance: list[tuple[str, float]] | None = None


@dataclass
class AutoMLResult:
    """Результат AutoML: лучшая модель и все результаты."""

    best: ModelResult
    all_results: list[ModelResult]
    selection_metric: MetricType
    selection_split: str


@dataclass
class CheckResult:
    """Результат одной диагностической проверки."""

    name: str
    status: QualityStatus
    message: str
    value: float | None

    @property
    def passed(self) -> bool:
        """Возвращает True если проверка пройдена (статус green)."""
        return self.status == "green"


@dataclass
class PanelDiagnostics:
    """Результат диагностики одной панели."""

    panel_id: int | str
    overall_status: QualityStatus
    checks: list[CheckResult]


@dataclass
class DiagnosticsResult:
    """Результат диагностики всего датасета."""

    panels: list[PanelDiagnostics]

    def to_df(self) -> pd.DataFrame:
        """Возвращает результаты диагностики в виде широкого DataFrame."""
        rows: list[dict[str, object]] = []
        for panel in self.panels:
            row: dict[str, object] = {
                "panel_id": panel.panel_id,
                "overall_status": panel.overall_status,
            }
            for check in panel.checks:
                row[f"{check.name}_passed"] = check.passed
                row[f"{check.name}_value"] = check.value
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, int]:
        """Возвращает количество панелей по статусам."""
        counts: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        for panel in self.panels:
            counts[panel.overall_status] += 1
        return counts


class CatBoostParameters(BaseModel):
    """Параметры для CatBoost регрессии."""

    iterations: int = Field(default=1000, description="Количество деревьев")
    learning_rate: float = Field(default=0.03, description="Learning rate")
    depth: int = Field(default=6, description="Глубина деревьев")
    l2_leaf_reg: float = Field(default=3.0, description="L2 регуляризация")
    subsample: float = Field(default=0.8, description="Доля samples для обучения")
    rsm: float = Field(default=0.8, description="Доля features для обучения")
    random_seed: int = Field(default=420, description="Random seed")
    verbose: int | bool = Field(default=100, description="Частота вывода логов")
    loss_function: str = Field(default="RMSE", description="Функция потерь")

    model_config = {"frozen": True}
