"""Проверка filter_time_series() на синтетических данных."""

import logging
from pathlib import Path

import pandas as pd

from src.configs.settings import FiltrationConfig
from src.filtration import filter_time_series

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "sample_ts.csv"


def main() -> None:
    """Загружает данные и запускает фильтрацию с дефолтным конфигом."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    logger.info("Загружено %d строк, %d артикулов", len(df), df["article"].nunique())

    config = FiltrationConfig()
    result = filter_time_series(df, config)

    logger.info("--- summary ---")
    for step, count in result.summary().items():
        logger.info("  %s: %d dropped", step, count)

    logger.info("total_dropped: %d", result.total_dropped)
    logger.info("Осталось артикулов: %d", result.df["article"].nunique())

    report = result.to_report_df()
    logger.info("--- report ---\n%s", report.to_string(index=False))


if __name__ == "__main__":
    main()
