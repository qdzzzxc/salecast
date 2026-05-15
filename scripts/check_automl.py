"""Проверка AutoML пайплайна на синтетических данных."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.automl.config import AutoMLConfig
from src.automl.selector import ModelSelector
from src.configs.settings import Settings
from src.custom_types import Splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_synthetic_splits() -> Splits[pd.DataFrame]:
    """Создаёт синтетические сплиты с 5 артикулами и сезонностью."""
    rng = np.random.default_rng(42)
    articles = [f"A{i}" for i in range(5)]
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")

    rows = []
    for article in articles:
        base = rng.integers(50, 200)
        season = np.sin(2 * np.pi * np.arange(36) / 12) * base * 0.3
        noise = rng.normal(0, base * 0.05, size=36)
        values = np.clip(base + season + noise, 1, None)
        for i, date in enumerate(dates):
            rows.append({"article": article, "date": date, "sales": float(values[i])})

    df = pd.DataFrame(rows)
    train_df = df[df["date"] < "2023-07-01"].copy().reset_index(drop=True)
    val_df = df[(df["date"] >= "2023-07-01") & (df["date"] < "2023-10-01")].copy().reset_index(drop=True)
    test_df = df[df["date"] >= "2023-10-01"].copy().reset_index(drop=True)
    return Splits(train=train_df, val=val_df, test=test_df)


def main() -> None:
    """Запускает AutoML на синтетических данных и выводит результаты."""
    splits = _make_synthetic_splits()
    settings = Settings()
    config = AutoMLConfig(
        models=["seasonal_naive", "catboost"],
        selection_metric="mape",
        use_hyperopt=False,
    )

    logger.info("Запуск AutoML...")
    selector = ModelSelector(config)
    result = selector.run(splits, settings)

    logger.info(f"\n{'='*50}")
    logger.info(f"Лучшая модель: {result.best.name}")
    logger.info(f"Метрика выбора: {result.selection_metric} на {result.selection_split}")

    for model_result in result.all_results:
        val_evals = [s for s in model_result.evaluation.splits if s.split_name == result.selection_split]
        test_evals = [s for s in model_result.evaluation.splits if s.split_name == "test"]
        val_mape = val_evals[0].overall_metrics.mape if val_evals else float("nan")
        test_mape = test_evals[0].overall_metrics.mape if test_evals else float("nan")
        logger.info(
            f"  {model_result.name}: val_mape={val_mape:.4f}, test_mape={test_mape:.4f}"
        )

    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
