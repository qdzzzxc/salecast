"""Тесты отдельных функций построения признаков из classifical_features."""

import numpy as np
import pandas as pd
import pytest

from src.classifical_features import (
    _add_calendar_features,
    _add_diff_features,
    _add_ema_features,
    _add_lag_features,
    _add_panel_features,
    _add_rolling_features,
)


@pytest.fixture
def panel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article": "A1",
            "date": pd.date_range("2023-01-01", periods=12, freq="MS"),
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


class TestAddLagFeatures:
    def test_lag_columns_added(self, panel_df) -> None:
        result = _add_lag_features(panel_df, "sales", lags=[1, 2, 3])
        assert "sales_lag_1" in result.columns
        assert "sales_lag_2" in result.columns
        assert "sales_lag_3" in result.columns

    def test_lag_1_values(self, panel_df) -> None:
        result = _add_lag_features(panel_df, "sales", lags=[1])
        assert np.isnan(result["sales_lag_1"].iloc[0])
        assert result["sales_lag_1"].iloc[1] == 10.0
        assert result["sales_lag_1"].iloc[2] == 20.0
        assert result["sales_lag_1"].iloc[5] == 50.0

    def test_lag_3_first_values_nan(self, panel_df) -> None:
        result = _add_lag_features(panel_df, "sales", lags=[3])
        assert result["sales_lag_3"].iloc[:3].isna().all()
        assert result["sales_lag_3"].iloc[3] == 10.0

    def test_no_future_leakage(self, panel_df) -> None:
        result = _add_lag_features(panel_df, "sales", lags=[1])
        for i in range(1, len(result)):
            assert result["sales_lag_1"].iloc[i] == result["sales"].iloc[i - 1]


class TestAddRollingFeatures:
    def test_columns_added(self, panel_df) -> None:
        result = _add_rolling_features(panel_df, "sales", windows=[2, 3])
        assert "sales_ma_2" in result.columns
        assert "sales_ma_3" in result.columns

    def test_ma_uses_shifted_values(self, panel_df) -> None:
        result = _add_rolling_features(panel_df, "sales", windows=[2])
        assert np.isnan(result["sales_ma_2"].iloc[0])
        assert result["sales_ma_2"].iloc[1] == pytest.approx(10.0)
        assert result["sales_ma_2"].iloc[2] == pytest.approx(15.0)

    def test_no_future_leakage(self, panel_df) -> None:
        result = _add_rolling_features(panel_df, "sales", windows=[3])
        assert result["sales_ma_3"].iloc[3] == pytest.approx(20.0)
        assert result["sales_ma_3"].iloc[3] != pytest.approx(30.0)


class TestAddEmaFeatures:
    def test_columns_added(self, panel_df) -> None:
        result = _add_ema_features(panel_df, "sales", spans=[2, 3])
        assert "sales_ema_2" in result.columns
        assert "sales_ema_3" in result.columns

    def test_first_value_nan(self, panel_df) -> None:
        result = _add_ema_features(panel_df, "sales", spans=[2])
        assert np.isnan(result["sales_ema_2"].iloc[0])

    def test_no_future_leakage(self, panel_df) -> None:
        result = _add_ema_features(panel_df, "sales", spans=[2])
        assert result["sales_ema_2"].iloc[1] == pytest.approx(10.0)
        assert result["sales_ema_2"].iloc[2] < 30.0


class TestAddDiffFeatures:
    def test_columns_added(self, panel_df) -> None:
        result = _add_diff_features(panel_df, "sales")
        assert "sales_diff_1" in result.columns
        assert "sales_pct_change_1" in result.columns

    def test_diff_values(self, panel_df) -> None:
        result = _add_diff_features(panel_df, "sales")
        assert np.isnan(result["sales_diff_1"].iloc[0])
        assert np.isnan(result["sales_diff_1"].iloc[1])
        assert result["sales_diff_1"].iloc[2] == pytest.approx(10.0)

    def test_no_inf_in_pct_change(self, panel_df) -> None:
        df = panel_df.copy()
        df.loc[df.index[0], "sales"] = 0.0
        result = _add_diff_features(df, "sales")
        assert not np.any(np.isinf(result["sales_pct_change_1"].dropna()))


class TestAddPanelFeatures:
    def test_columns_added(self, panel_df) -> None:
        result = _add_panel_features(panel_df, "sales")
        assert "panel_mean" in result.columns
        assert "panel_std" in result.columns
        assert "sales_vs_mean" in result.columns

    def test_first_value_nan(self, panel_df) -> None:
        result = _add_panel_features(panel_df, "sales")
        assert np.isnan(result["panel_mean"].iloc[0])

    def test_expanding_mean_is_correct(self, panel_df) -> None:
        result = _add_panel_features(panel_df, "sales")
        assert result["panel_mean"].iloc[1] == pytest.approx(10.0)
        assert result["panel_mean"].iloc[2] == pytest.approx(15.0)

    def test_no_inf_in_vs_mean(self, panel_df) -> None:
        result = _add_panel_features(panel_df, "sales")
        valid = result["sales_vs_mean"].dropna()
        assert not np.any(np.isinf(valid))


class TestAddCalendarFeatures:
    def test_columns_added(self, panel_df) -> None:
        result = _add_calendar_features(panel_df, "date")
        assert "month" in result.columns
        assert "quarter" in result.columns
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

    def test_month_values(self, panel_df) -> None:
        result = _add_calendar_features(panel_df, "date")
        assert result["month"].iloc[0] == 1
        assert result["month"].iloc[5] == 6
        assert result["quarter"].iloc[0] == 1
        assert result["quarter"].iloc[5] == 2

    def test_sin_cos_bounded(self, panel_df) -> None:
        result = _add_calendar_features(panel_df, "date")
        assert (result["month_sin"] >= -1).all()
        assert (result["month_sin"] <= 1).all()
        assert (result["month_cos"] >= -1).all()
        assert (result["month_cos"] <= 1).all()

    def test_january_december_sin_close(self) -> None:
        df = pd.DataFrame(
            {
                "article": "A1",
                "date": pd.to_datetime(["2023-01-01", "2023-12-01"]),
                "sales": [1.0, 1.0],
            }
        )
        result = _add_calendar_features(df, "date")
        assert abs(result["month_sin"].iloc[0]) < 1.0
        assert abs(result["month_sin"].iloc[1]) < 1.0
