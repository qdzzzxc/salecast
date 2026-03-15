import logging

import pandas as pd

from src.automl.models.catboost_model import CatBoostForecastModel
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, Splits

logger = logging.getLogger(__name__)


def tune_catboost(
    splits: Splits[pd.DataFrame],
    settings: Settings,
    n_trials: int = 30,
    timeout: int | None = None,
) -> CatBoostParameters:
    """Подбирает гиперпараметры CatBoost с помощью Optuna, минимизируя MAPE на val.

    Args:
        splits (Splits[pd.DataFrame]): Сплиты данных, val обязателен.
        settings (Settings): Конфигурация пайплайна.
        n_trials (int): Количество попыток Optuna. Defaults to 30.

    Returns:
        CatBoostParameters: Оптимальные гиперпараметры.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError("optuna не установлен. Установите его: uv add optuna") from e

    if splits.val is None:
        raise ValueError("Для гиперпоиска требуется val split. Splits.val не должен быть None.")

    def objective(trial: "optuna.Trial") -> float:
        params = CatBoostParameters(
            iterations=trial.suggest_int("iterations", 200, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            verbose=False,
        )
        model = CatBoostForecastModel(params=params)
        result = model.fit_evaluate(splits, settings)

        val_evals = [s for s in result.evaluation.splits if s.split_name == "val"]
        if not val_evals:
            return float("inf")

        return val_evals[0].overall_metrics.mape

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Optuna best MAPE: {study.best_value:.4f}, params: {best}")

    return CatBoostParameters(
        iterations=best["iterations"],
        learning_rate=best["learning_rate"],
        depth=best["depth"],
        l2_leaf_reg=best["l2_leaf_reg"],
        subsample=best["subsample"],
    )
