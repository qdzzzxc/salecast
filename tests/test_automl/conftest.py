import numpy as np
import pandas as pd
import pytest

from src.configs.settings import Settings
from src.custom_types import Splits


@pytest.fixture()
def sample_splits() -> Splits[pd.DataFrame]:
    """Синтетические Splits с 5 артикулами × 36 месяцев (train=30, val=3, test=3)."""
    rng = np.random.default_rng(42)
    articles = [f"A{i}" for i in range(5)]
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")

    rows = []
    for article in articles:
        base = rng.integers(10, 100)
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


@pytest.fixture()
def sample_settings() -> Settings:
    """Дефолтный Settings с отключённым обратным преобразованием для тестов."""
    return Settings()


@pytest.fixture()
def full_df(sample_splits: Splits[pd.DataFrame]) -> pd.DataFrame:
    """Полный датафрейм (train+val+test) для тестов forecast_future."""
    return pd.concat(
        [sample_splits.train, sample_splits.val, sample_splits.test], ignore_index=True
    )
