import pandas as pd

from src.configs.settings import Settings
from src.custom_types import EvaluationResults, Splits
from src.evaluation import evaluate_multiple_splits, log_evaluation_results
from src.seasonal_naive_utilities.seasonal_naive_model import SeasonalNaiveModel


def evaluate_seasonal_naive(
    model: SeasonalNaiveModel,
    splits: Splits[pd.DataFrame],
    settings: Settings,
    skip_train_warmup: bool = True,
) -> EvaluationResults:
    """Оценивает сезонную наивную модель на всех сплитах."""
    cols = settings.columns
    target = cols.main_target

    splits_data = {}

    for split_name, split_df in splits.splits:
        is_train = split_name == "train"
        
        if is_train and skip_train_warmup:
            split_df = _filter_warmup_period(split_df, cols.id, model.seasonal_period)
        
        split_df = split_df.reset_index(drop=True)
        result_df = split_df[[cols.id, target]].copy()
        y_pred = model.predict(split_df, cols.id, target, is_train=is_train)
        splits_data[split_name] = (result_df, y_pred)

    results = evaluate_multiple_splits(
        splits_data=splits_data,
        panel_column=cols.id,
        target_column=target,
    )

    log_evaluation_results(results)

    return results


def _filter_warmup_period(df: pd.DataFrame, panel_column: str, warmup: int) -> pd.DataFrame:
    """Удаляет первые warmup точек для каждой панели."""
    return (
        df.groupby(panel_column, group_keys=False)
        .apply(lambda g: g.iloc[warmup:])
    )