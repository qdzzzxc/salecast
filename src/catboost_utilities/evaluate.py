import logging

import catboost as cb
import numpy as np
import pandas as pd

from src.configs.settings import Settings
from src.custom_types import EvaluationResults, PanelScalers, Splits
from src.data_processing import inverse_transform_panel_columns
from src.evaluation import evaluate_multiple_splits, log_evaluation_results

logger = logging.getLogger(__name__)


def evaluate_catboost(
    model: cb.CatBoostRegressor,
    splits: Splits[pd.DataFrame],
    settings: Settings,
    scalers: PanelScalers | None = None,
) -> EvaluationResults:
    """Оценивает CatBoost на всех сплитах с метриками по панелям."""
    cols = settings.columns
    target = cols.main_target

    splits_data = {}

    for split_name, split_df in splits.splits:
        logger.info(f"Evaluating {split_name} split")
        result_df, y_pred = _prepare_predictions(model, split_df, settings, scalers)
        splits_data[split_name] = (result_df, y_pred)

    logger.info("Computing metrics")
    results = evaluate_multiple_splits(
        splits_data=splits_data,
        panel_column=cols.id,
        target_column=target,
    )

    log_evaluation_results(results)

    return results


def _prepare_predictions(
    model: cb.CatBoostRegressor,
    df: pd.DataFrame,
    settings: Settings,
    scalers: PanelScalers | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Подготавливает предсказания для одного сплита."""
    target = settings.columns.main_target
    cols = settings.columns
    prep = settings.preprocessing
    downstream = settings.downstream

    drop_cols = [c for c in [target, cols.id, cols.date] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y_pred_scaled = model.predict(X)

    result_df = df[[cols.id, target]].copy().reset_index(drop=True)

    if downstream.inverse and scalers is not None:
        y_pred = _inverse_predictions(
            predictions=y_pred_scaled,
            df=df,
            scalers=scalers,
            target=target,
            panel_column=cols.id,
            apply_log=prep.apply_log,
        )
        result_df[target] = _inverse_predictions(
            predictions=df[target].values,
            df=df,
            scalers=scalers,
            target=target,
            panel_column=cols.id,
            apply_log=prep.apply_log,
        )
    else:
        y_pred = y_pred_scaled

    if downstream.round_predictions:
        y_pred = np.round(y_pred)

    return result_df, y_pred


def _inverse_predictions(
    predictions: np.ndarray,
    df: pd.DataFrame,
    scalers: PanelScalers,
    target: str,
    panel_column: str,
    apply_log: bool,
) -> np.ndarray:
    """Обратное преобразование предсказаний."""
    pred_df = df[[panel_column]].copy().reset_index(drop=True)
    pred_df[target] = predictions

    inversed = inverse_transform_panel_columns(
        pred_df,
        scalers,
        panel_column,
        [target],
        apply_log,
    )

    return inversed[target].values
