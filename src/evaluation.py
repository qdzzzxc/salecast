import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm

from src.configs.settings import Settings
from src.custom_types import (
    EvaluationResults,
    PanelMetrics,
    PanelPredictions,
    RegressionMetrics,
    SplitEvaluation,
    Splits,
    SplitsWithoutTrain,
)

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Вычисляет метрики регрессии с учетом масштаба"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    else:
        mape = float("inf")

    mean_y = np.mean(y_true)
    if mean_y != 0:
        nrmse = rmse / mean_y
        nmae = mae / mean_y
    else:
        nrmse = float("inf")
        nmae = float("inf")

    cv_rmse = rmse / mean_y if mean_y != 0 else float("inf")

    std_y = np.std(y_true)
    if std_y != 0:
        nrmse_std = rmse / std_y
    else:
        nrmse_std = float("inf")

    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mape=mape,
        explained_variance=explained_var,
        nrmse=nrmse,
        nmae=nmae,
        cv_rmse=cv_rmse,
        nrmse_std=nrmse_std,
    )


def _compute_panel_metrics(panel_pred: PanelPredictions) -> PanelMetrics:
    """Вычисляет метрики для одной панели."""
    metrics = compute_regression_metrics(panel_pred.y_true, panel_pred.y_pred)
    return PanelMetrics(
        panel_id=panel_pred.panel_id,
        split=panel_pred.split,
        metrics=metrics,
        y_true=panel_pred.y_true,
        y_pred=panel_pred.y_pred,
    )


def evaluate_split(
    df: pd.DataFrame,
    predictions: np.ndarray,
    panel_column: str,
    target_column: str,
    split_name: str,
) -> SplitEvaluation:
    """Вычисляет метрики для одного сплита."""
    panel_ids = df[panel_column].unique()
    panel_predictions = []

    for panel_id in panel_ids:
        mask = df[panel_column] == panel_id
        y_true = df.loc[mask, target_column].values
        y_pred = predictions[mask]

        if len(y_true) == 0:
            continue

        panel_pred = PanelPredictions(
            panel_id=panel_id,
            y_true=y_true,
            y_pred=y_pred,
            split=split_name,
        )
        panel_predictions.append(panel_pred)

    panel_metrics = [
        _compute_panel_metrics(pp)
        for pp in tqdm(panel_predictions, desc=f"Evaluating {split_name} panels")
    ]

    y_true_all = df[target_column].values
    overall_metrics = compute_regression_metrics(y_true_all, predictions)

    return SplitEvaluation(
        split_name=split_name,
        overall_metrics=overall_metrics,
        panel_metrics=panel_metrics,
        y_true=y_true_all,
        y_pred=predictions,
    )


def evaluate_multiple_splits(
    splits_data: dict[str, tuple[pd.DataFrame, np.ndarray]],
    panel_column: str,
    target_column: str,
) -> EvaluationResults:
    """Вычисляет метрики для нескольких сплитов."""
    split_evaluations = []

    for split_name, (df, predictions) in splits_data.items():
        split_eval = evaluate_split(
            df=df,
            predictions=predictions,
            panel_column=panel_column,
            target_column=target_column,
            split_name=split_name,
        )
        split_evaluations.append(split_eval)

    return EvaluationResults(splits=split_evaluations)


def log_evaluation_results(results: EvaluationResults) -> None:
    """Логирует результаты оценки."""
    for split_eval in results.splits:
        metrics = split_eval.overall_metrics
        logger.info(f"\n=== {split_eval.split_name.upper()} OVERALL METRICS ===")
        logger.info(f"MAPE: {metrics.mape:.4f}")
        logger.info(f"RMSE: {metrics.rmse:.4f}")
        logger.info(f"MAE: {metrics.mae:.4f}")
        logger.info(f"R²: {metrics.r2:.4f}")

    logger.info("\n=== PANEL METRICS SUMMARY ===")
    panel_df = results.get_panel_metrics_df()
    for split_name in panel_df["split"].unique():
        split_metrics = panel_df[panel_df["split"] == split_name]
        logger.info(f"\n{split_name.upper()}:")
        logger.info(
            f"  MAPE - Mean: {split_metrics['mape'].mean():.4f}, "
            f"Median: {split_metrics['mape'].median():.4f}"
        )
        logger.info(
            f"  RMSE - Mean: {split_metrics['rmse'].mean():.4f}, "
            f"Median: {split_metrics['rmse'].median():.4f}"
        )


def get_panel_metrics_wide(
    panel_metrics_df: pd.DataFrame,
    sort_metric: str = "mape",
) -> dict[str, pd.DataFrame]:
    """Преобразует длинный формат метрик в широкий (по сплитам)."""
    panel_metrics_wide = {}

    for split_name in panel_metrics_df["split"].unique():
        split_data = panel_metrics_df[panel_metrics_df["split"] == split_name].copy()
        split_data = split_data.drop(columns=["split"])
        split_data = split_data.sort_values(by=sort_metric, ascending=False)
        panel_metrics_wide[split_name] = split_data

    return panel_metrics_wide


def combine_panel_results(panel_results: list[EvaluationResults]) -> EvaluationResults:
    """Объединяет результаты по панелям в общий результат."""
    combined_splits = []

    for split_name in ["train", "val", "test"]:
        all_split_evals = []
        for result in panel_results:
            split_eval = next((s for s in result.splits if s.split_name == split_name), None)
            if split_eval:
                all_split_evals.append(split_eval)

        if all_split_evals:
            all_panels: list[PanelMetrics] = []
            all_y_true: list[float] = []
            all_y_pred: list[float] = []

            for split_eval in all_split_evals:
                all_panels.extend(split_eval.panel_metrics)
                all_y_true.extend(split_eval.y_true)
                all_y_pred.extend(split_eval.y_pred)

            overall_metrics = compute_regression_metrics(np.array(all_y_true), np.array(all_y_pred))

            combined_split = SplitEvaluation(
                split_name=split_name,
                overall_metrics=overall_metrics,
                panel_metrics=all_panels,
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
            )
            combined_splits.append(combined_split)

    return EvaluationResults(splits=combined_splits)


def evaluate_from_predictions(
    pred_df: pd.DataFrame,
    splits: Splits[pd.DataFrame] | SplitsWithoutTrain[pd.DataFrame],
    settings: Settings,
    prediction_column: str = "prediction",
) -> EvaluationResults:
    """Общая функция оценки по готовым предсказаниям"""
    cols = settings.columns
    target = cols.main_target

    splits_data = {}

    for split_name, split_df in splits.splits:
        split_dates = split_df[cols.date].unique()
        split_ids = split_df[cols.id].unique()

        split_pred_df = pred_df[
            (pred_df[cols.date].isin(split_dates)) & (pred_df[cols.id].isin(split_ids))
        ].copy()

        result_df = split_pred_df[[cols.id, target]].copy()
        y_pred = split_pred_df[prediction_column].values

        splits_data[split_name] = (result_df, y_pred)

    results = evaluate_multiple_splits(
        splits_data=splits_data,
        panel_column=cols.id,
        target_column=target,
    )

    log_evaluation_results(results)

    return results
