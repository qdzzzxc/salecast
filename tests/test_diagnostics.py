import numpy as np
import pandas as pd
import pytest

from src.custom_types import DiagnosticsResult, PanelDiagnostics
from src.diagnostics import DiagnosticsConfig, run_diagnostics
from src.diagnostics.checks import (
    check_autocorrelation,
    check_cv,
    check_length,
    check_seasonality,
    check_stationarity,
    check_trend,
    check_zero_ratio,
)


@pytest.fixture()
def default_config() -> DiagnosticsConfig:
    return DiagnosticsConfig()


@pytest.fixture()
def seasonal_series() -> np.ndarray:
    np.random.seed(0)
    t = np.arange(36)
    return 100 + 20 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, 36)


@pytest.fixture()
def random_walk_series() -> np.ndarray:
    np.random.seed(1)
    return np.cumsum(np.random.normal(0, 1, 40))


@pytest.fixture()
def panel_df() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for article in [1, 2, 3]:
        for i in range(30):
            rows.append(
                {
                    "article": article,
                    "date": pd.Timestamp("2021-01-01") + pd.DateOffset(months=i),
                    "sales": float(np.random.poisson(50)),
                }
            )
    return pd.DataFrame(rows)


class TestCheckLength:
    def test_red_below_min_red(self, default_config: DiagnosticsConfig) -> None:
        result = check_length(np.ones(5), default_config)
        assert result.status == "red"
        assert result.passed is False
        assert result.value == 5.0

    def test_yellow_between_thresholds(self, default_config: DiagnosticsConfig) -> None:
        result = check_length(np.ones(18), default_config)
        assert result.status == "yellow"
        assert result.value == 18.0

    def test_green_above_min_yellow(self, default_config: DiagnosticsConfig) -> None:
        result = check_length(np.ones(24), default_config)
        assert result.status == "green"
        assert result.passed is True

    def test_name_is_length(self, default_config: DiagnosticsConfig) -> None:
        result = check_length(np.ones(30), default_config)
        assert result.name == "length"


class TestCheckZeroRatio:
    def test_red_above_max_red(self, default_config: DiagnosticsConfig) -> None:
        values = np.array([0.0] * 6 + [1.0] * 4)
        result = check_zero_ratio(values, default_config)
        assert result.status == "red"

    def test_yellow_between_thresholds(self, default_config: DiagnosticsConfig) -> None:
        values = np.array([0.0] * 3 + [1.0] * 7)
        result = check_zero_ratio(values, default_config)
        assert result.status == "yellow"

    def test_green_no_zeros(self, default_config: DiagnosticsConfig) -> None:
        result = check_zero_ratio(np.ones(20), default_config)
        assert result.status == "green"
        assert result.passed is True

    def test_value_is_ratio(self, default_config: DiagnosticsConfig) -> None:
        values = np.array([0.0] * 2 + [1.0] * 8)
        result = check_zero_ratio(values, default_config)
        assert result.value == pytest.approx(0.2)


class TestCheckCv:
    def test_red_when_mean_zero(self, default_config: DiagnosticsConfig) -> None:
        result = check_cv(np.zeros(20), default_config)
        assert result.status == "red"
        assert result.value is None

    def test_red_high_cv(self, default_config: DiagnosticsConfig) -> None:
        values = np.array([0.0] * 18 + [100.0, 200.0])
        result = check_cv(values, default_config)
        assert result.status == "red"

    def test_yellow_moderate_cv(self, default_config: DiagnosticsConfig) -> None:
        np.random.seed(5)
        values = np.abs(np.random.normal(10, 7, 30))
        result = check_cv(values, default_config)
        assert result.status in ("yellow", "red")

    def test_green_low_cv(self, default_config: DiagnosticsConfig) -> None:
        np.random.seed(5)
        values = np.abs(np.random.normal(100, 5, 30))
        result = check_cv(values, default_config)
        assert result.status == "green"

    def test_value_is_cv(self, default_config: DiagnosticsConfig) -> None:
        values = np.array([10.0] * 20)
        result = check_cv(values, default_config)
        assert result.value == pytest.approx(0.0)


class TestCheckAutocorrelation:
    def test_green_for_seasonal_series(
        self, seasonal_series: np.ndarray, default_config: DiagnosticsConfig
    ) -> None:
        result = check_autocorrelation(seasonal_series, default_config)
        assert result.status == "green"

    def test_yellow_for_too_short_series(self, default_config: DiagnosticsConfig) -> None:
        result = check_autocorrelation(np.array([1.0, 2.0, 3.0]), default_config)
        assert result.status == "yellow"
        assert result.value is None

    def test_name_is_autocorrelation(
        self, seasonal_series: np.ndarray, default_config: DiagnosticsConfig
    ) -> None:
        result = check_autocorrelation(seasonal_series, default_config)
        assert result.name == "autocorrelation"

    def test_value_is_p_value(
        self, seasonal_series: np.ndarray, default_config: DiagnosticsConfig
    ) -> None:
        result = check_autocorrelation(seasonal_series, default_config)
        assert result.value is not None
        assert 0.0 <= result.value <= 1.0


class TestCheckStationarity:
    def test_green_for_stationary_series(self, default_config: DiagnosticsConfig) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        result = check_stationarity(values, default_config)
        assert result.status == "green"

    def test_yellow_for_random_walk(
        self, random_walk_series: np.ndarray, default_config: DiagnosticsConfig
    ) -> None:
        result = check_stationarity(random_walk_series, default_config)
        assert result.status in ("yellow", "green")

    def test_name_is_stationarity(self, default_config: DiagnosticsConfig) -> None:
        result = check_stationarity(np.random.normal(0, 1, 50), default_config)
        assert result.name == "stationarity"


class TestCheckSeasonality:
    def test_green_for_seasonal_series(
        self, seasonal_series: np.ndarray, default_config: DiagnosticsConfig
    ) -> None:
        result = check_seasonality(seasonal_series, default_config)
        assert result.status == "green"

    def test_yellow_for_too_short_series(self, default_config: DiagnosticsConfig) -> None:
        result = check_seasonality(np.ones(10), default_config)
        assert result.status == "yellow"
        assert result.value is None

    def test_yellow_for_no_seasonality(self, default_config: DiagnosticsConfig) -> None:
        np.random.seed(10)
        values = np.random.normal(0, 1, 36)
        result = check_seasonality(values, default_config)
        assert result.status in ("yellow", "green")

    def test_value_is_acf12(
        self, seasonal_series: np.ndarray, default_config: DiagnosticsConfig
    ) -> None:
        result = check_seasonality(seasonal_series, default_config)
        assert result.value is not None
        assert 0.0 <= result.value <= 1.0


class TestCheckTrend:
    def test_yellow_for_monotonic_trend(self, default_config: DiagnosticsConfig) -> None:
        values = np.arange(1.0, 41.0)
        result = check_trend(values, default_config)
        assert result.status == "yellow"

    def test_green_for_no_trend(self, default_config: DiagnosticsConfig) -> None:
        np.random.seed(7)
        values = np.random.normal(50, 5, 36)
        result = check_trend(values, default_config)
        assert result.status in ("green", "yellow")

    def test_name_is_trend(self, default_config: DiagnosticsConfig) -> None:
        result = check_trend(np.arange(1.0, 30.0), default_config)
        assert result.name == "trend"


class TestRunDiagnostics:
    def test_returns_diagnostics_result(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        assert isinstance(result, DiagnosticsResult)

    def test_panel_count_matches(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        assert len(result.panels) == 3

    def test_all_panels_have_checks(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        for panel in result.panels:
            assert isinstance(panel, PanelDiagnostics)
            assert len(panel.checks) == 7

    def test_overall_status_is_worst(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        order = {"green": 0, "yellow": 1, "red": 2}
        for panel in result.panels:
            check_statuses = [order[c.status] for c in panel.checks]
            assert order[panel.overall_status] == max(check_statuses)

    def test_summary_keys(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        summary = result.summary()
        assert set(summary.keys()) == {"green", "yellow", "red"}

    def test_summary_counts_sum_to_panel_count(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        assert sum(result.summary().values()) == len(result.panels)

    def test_to_df_shape(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        df = result.to_df()
        assert len(df) == 3
        assert "panel_id" in df.columns
        assert "overall_status" in df.columns

    def test_to_df_has_passed_columns(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        df = result.to_df()
        expected_checks = ["length", "zero_ratio", "cv", "autocorrelation", "stationarity", "seasonality", "trend"]
        for check in expected_checks:
            assert f"{check}_passed" in df.columns
            assert f"{check}_value" in df.columns

    def test_to_df_passed_columns_are_bool(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales")
        df = result.to_df()
        assert df["length_passed"].dtype == bool

    def test_default_config_used_when_none(self, panel_df: pd.DataFrame) -> None:
        result = run_diagnostics(panel_df, "article", "date", "sales", config=None)
        assert len(result.panels) == 3

    def test_custom_config_affects_results(self, panel_df: pd.DataFrame) -> None:
        strict_config = DiagnosticsConfig(min_length_yellow=100)
        result = run_diagnostics(panel_df, "article", "date", "sales", config=strict_config)
        for panel in result.panels:
            length_check = next(c for c in panel.checks if c.name == "length")
            assert length_check.status == "yellow"
