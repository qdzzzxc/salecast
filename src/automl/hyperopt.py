import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from src.automl.models.catboost_model import CatBoostForecastModel
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, Splits

logger = logging.getLogger(__name__)

TrialCallback = Callable[[int, int, float, float], None]
"""(trial_number, n_trials, trial_value, best_value) -> None"""

# Дефолтные диапазоны поиска
DEFAULT_SEARCH_SPACE: dict[str, dict] = {
    "iterations": {"type": "int", "low": 200, "high": 1000, "step": 50},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "depth": {"type": "int", "low": 3, "high": 8},
    "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 50.0, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
}


@dataclass
class HyperoptResult:
    """Результат подбора гиперпараметров."""

    best_params: CatBoostParameters
    best_value: float
    trials: list[dict] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    param_importance: dict[str, float] = field(default_factory=dict)


def tune_catboost(
    splits: Splits[pd.DataFrame],
    settings: Settings,
    n_trials: int = 30,
    timeout: int | None = None,
    on_trial_done: TrialCallback | None = None,
    search_space: dict[str, dict] | None = None,
) -> HyperoptResult:
    """Подбирает гиперпараметры CatBoost с помощью Optuna, минимизируя MAPE на val.

    Args:
        splits: Сплиты данных, val обязателен.
        settings: Конфигурация пайплайна.
        n_trials: Количество попыток Optuna.
        timeout: Таймаут в секундах.
        on_trial_done: Callback после каждого trial'а.

    Returns:
        HyperoptResult с лучшими параметрами и историей.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError("optuna не установлен. Установите его: uv add optuna") from e

    if splits.val is None:
        raise ValueError("Для гиперпоиска требуется val split. Splits.val не должен быть None.")

    ss = {**DEFAULT_SEARCH_SPACE, **(search_space or {})}

    def _suggest(trial: "optuna.Trial", name: str, spec: dict) -> int | float:
        if spec["type"] == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

    def objective(trial: "optuna.Trial") -> float:
        params = CatBoostParameters(
            iterations=_suggest(trial, "iterations", ss["iterations"]),
            learning_rate=_suggest(trial, "learning_rate", ss["learning_rate"]),
            depth=_suggest(trial, "depth", ss["depth"]),
            l2_leaf_reg=_suggest(trial, "l2_leaf_reg", ss["l2_leaf_reg"]),
            subsample=_suggest(trial, "subsample", ss["subsample"]),
            verbose=False,
        )
        model = CatBoostForecastModel(params=params)
        result = model.fit_evaluate(splits, settings)

        val_evals = [s for s in result.evaluation.splits if s.split_name == "val"]
        if not val_evals:
            return float("inf")

        return val_evals[0].overall_metrics.mape

    best_so_far = float("inf")

    def _trial_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        nonlocal best_so_far
        if trial.value is not None and trial.value < best_so_far:
            best_so_far = trial.value
        if on_trial_done:
            on_trial_done(
                trial.number + 1,
                n_trials,
                trial.value if trial.value is not None else float("inf"),
                best_so_far,
            )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        callbacks=[_trial_callback],
    )

    best = study.best_params
    logger.info("Optuna best MAPE: %.4f, params: %s", study.best_value, best)

    param_names = ["iterations", "learning_rate", "depth", "l2_leaf_reg", "subsample"]
    trials = []
    for t in study.trials:
        if t.value is None:
            continue
        trial_data: dict = {"number": t.number, "value": round(t.value, 6)}
        for p in param_names:
            trial_data[p] = t.params.get(p)
        trials.append(trial_data)

    # Importance пока study ещё жив
    param_importance: dict[str, float] = {}
    try:
        param_importance = optuna.importance.get_param_importances(study)
    except Exception:
        logger.warning("Не удалось вычислить param importances")

    return HyperoptResult(
        best_params=CatBoostParameters(
            iterations=best["iterations"],
            learning_rate=best["learning_rate"],
            depth=best["depth"],
            l2_leaf_reg=best["l2_leaf_reg"],
            subsample=best["subsample"],
        ),
        best_value=study.best_value,
        trials=trials,
        param_names=param_names,
        param_importance=param_importance,
    )
