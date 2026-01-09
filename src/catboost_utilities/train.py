import logging

import catboost as cb
import pandas as pd

from src.configs.settings import Settings
from src.custom_types import CatBoostParameters

logger = logging.getLogger(__name__)


def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    params: CatBoostParameters,
    settings: Settings,
) -> cb.CatBoostRegressor:
    """Обучает CatBoost модель."""
    target = settings.columns.main_target
    cols = settings.columns

    drop_cols = [c for c in [target, cols.id, cols.date] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]

    train_pool = cb.Pool(X_train, y_train)

    eval_set = None
    if val_df is not None:
        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df[target]
        eval_set = cb.Pool(X_val, y_val)

    model = cb.CatBoostRegressor(**params.model_dump())
    model.fit(
        train_pool,
        eval_set=eval_set,
        use_best_model=eval_set is not None,
        early_stopping_rounds=params.iterations // 10,
    )

    logger.info(f"CatBoost trained with {model.tree_count_} trees")
    return model
