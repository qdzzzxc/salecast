"""Тесты общего качества build_ts_features: колонки, строки, inf, drop_na."""

import numpy as np
import pandas as pd
import pytest

from src.classifical_features import build_ts_features
from src.configs.settings import Settings


@pytest.fixture
def two_panel_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    a1 = pd.DataFrame(
        {
            "article": "A1",
            "date": dates,
            "sales": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
        }
    )
    a2 = pd.DataFrame(
        {
            "article": "A2",
            "date": dates,
            "sales": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
            ],
        }
    )
    return pd.concat([a1, a2], ignore_index=True)


class TestFeatureQuality:
    def test_no_inf_in_features(self, two_panel_df) -> None:
        settings = Settings()
        result = build_ts_features(two_panel_df, settings, disable_tqdm=True)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.any(np.isinf(numeric.values[~np.isnan(numeric.values)]))

    def test_expected_columns_present(self, two_panel_df) -> None:
        settings = Settings()
        result = build_ts_features(two_panel_df, settings, disable_tqdm=True)
        expected = {
            "sales_lag_1",
            "sales_lag_2",
            "sales_lag_3",
            "sales_ma_2",
            "sales_ma_3",
            "sales_ema_2",
            "sales_ema_3",
            "sales_diff_1",
            "sales_pct_change_1",
            "panel_mean",
            "panel_std",
            "sales_vs_mean",
            "month",
            "quarter",
            "month_sin",
            "month_cos",
        }
        assert expected.issubset(set(result.columns))

    def test_row_count_preserved(self, two_panel_df) -> None:
        settings = Settings()
        result = build_ts_features(two_panel_df, settings, disable_tqdm=True)
        assert len(result) == len(two_panel_df)

    def test_drop_na_reduces_rows(self, two_panel_df) -> None:
        settings = Settings()
        full = build_ts_features(two_panel_df, settings, disable_tqdm=True, drop_na=False)
        dropped = build_ts_features(two_panel_df, settings, disable_tqdm=True, drop_na=True)
        assert len(dropped) < len(full)
        numeric = dropped.select_dtypes(include=[np.number])
        assert not numeric.isna().any().any()
