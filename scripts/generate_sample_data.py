"""Генератор синтетических данных для ручной проверки модулей."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "sample_ts.csv"

SEED = 42
N_DATES = 24


def _make_date_range() -> pd.DatetimeIndex:
    """Возвращает месячный диапазон дат."""
    return pd.date_range("2023-01-01", periods=N_DATES, freq="MS")


def _normal_series(rng: np.random.Generator) -> np.ndarray:
    """Нормальный ряд, проходящий все фильтры."""
    return rng.integers(5, 50, size=N_DATES).astype(float)


def _all_zeros() -> np.ndarray:
    """Ряд из одних нулей — отсеется на edge_zeros."""
    return np.zeros(N_DATES)


def _high_inner_zeros(rng: np.random.Generator) -> np.ndarray:
    """Ряд с высокой долей внутренних нулей — отсеется на inner_zeros."""
    values = np.zeros(N_DATES)
    values[0] = 10.0
    values[-1] = 10.0
    nonzero_count = 3
    inner_idx = rng.choice(range(1, N_DATES - 1), size=nonzero_count, replace=False)
    values[inner_idx] = rng.integers(1, 10, size=nonzero_count).astype(float)
    return values


def _short_series(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Короткий ряд < 18 точек — отсеется на min_length."""
    length = 10
    return rng.integers(5, 30, size=length).astype(float), length


def _constant_series() -> np.ndarray:
    """Константный ненулевой ряд — отсеется на zero_std."""
    return np.full(N_DATES, 7.0)


def _small_total_series() -> np.ndarray:
    """Ряд с маленькой суммой < 10 — отсеется на min_total."""
    values = np.zeros(N_DATES)
    values[0] = 1.0
    values[5] = 2.0
    values[10] = 3.0
    return values


def _edge_zeros_series(rng: np.random.Generator) -> np.ndarray:
    """Ряд с нулями по краям, но живой внутри — проходит фильтры после тримминга."""
    values = np.zeros(N_DATES)
    values[3:20] = rng.integers(5, 40, size=17).astype(float)
    return values


def generate_sample_data() -> pd.DataFrame:
    """Генерирует синтетический датасет для проверки фильтрации."""
    rng = np.random.default_rng(SEED)
    dates = _make_date_range()
    rows: list[pd.DataFrame] = []

    cases: list[tuple[str, np.ndarray, pd.DatetimeIndex]] = [
        ("normal_1", _normal_series(rng), dates),
        ("normal_2", _normal_series(rng), dates),
        ("normal_3", _normal_series(rng), dates),
        ("all_zeros", _all_zeros(), dates),
        ("high_inner_zeros", _high_inner_zeros(rng), dates),
        ("constant", _constant_series(), dates),
        ("small_total", _small_total_series(), dates),
        ("edge_zeros_ok", _edge_zeros_series(rng), dates),
    ]

    short_values, short_len = _short_series(rng)
    cases.append(("short", short_values, dates[:short_len]))

    for article, values, case_dates in cases:
        rows.append(
            pd.DataFrame({"article": article, "date": case_dates, "sales": values})
        )

    dup_values = _normal_series(rng)
    dup_df = pd.DataFrame({"article": "has_duplicates", "date": dates, "sales": dup_values})
    extra_rows = pd.DataFrame({
        "article": "has_duplicates",
        "date": dates[:5],
        "sales": rng.integers(1, 10, size=5).astype(float),
    })
    rows.append(pd.concat([dup_df, extra_rows], ignore_index=True))

    df = pd.concat(rows, ignore_index=True)
    logger.info("Сгенерировано %d строк, %d артикулов", len(df), df["article"].nunique())
    return df


def main() -> None:
    """Точка входа: генерирует и сохраняет sample_ts.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_sample_data()
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Сохранено в %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
