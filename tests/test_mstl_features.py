import numpy as np
import pandas as pd
import pytest

from src.mstl_features import (
    decompose_mstl,
    extract_mstl_features,
    extract_seasonal_vectors,
    seasonality_strength,
)


@pytest.fixture
def seasonal_series() -> np.ndarray:
    """Синтетический ряд с сезонностью period=12."""
    rng = np.random.default_rng(42)
    t = np.arange(36)
    return 100 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 36)


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """3 панели по 36 точек с разной сезонностью."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    rows = []
    for i in range(3):
        amp = (i + 1) * 10
        for j, d in enumerate(dates):
            val = 100 + amp * np.sin(2 * np.pi * j / 12) + rng.normal(0, 1)
            rows.append({"article": f"A{i}", "date": d, "sales": max(0.1, val)})
    return pd.DataFrame(rows)


class TestDecomposeMstl:
    def test_returns_dict(self, seasonal_series) -> None:
        result = decompose_mstl(seasonal_series, freq="MS")
        assert isinstance(result, dict)

    def test_has_required_keys(self, seasonal_series) -> None:
        result = decompose_mstl(seasonal_series, freq="MS")
        assert "trend" in result
        assert "seasonal" in result
        assert "remainder" in result

    def test_components_sum_to_original(self, seasonal_series) -> None:
        result = decompose_mstl(seasonal_series, freq="MS")
        reconstructed = result["trend"] + result["seasonal"] + result["remainder"]
        np.testing.assert_allclose(reconstructed, seasonal_series, atol=1e-6)

    def test_short_series_returns_trivial(self) -> None:
        short = np.array([1.0, 2.0, 3.0])
        result = decompose_mstl(short, freq="MS", season_lengths=[12])
        assert np.allclose(result["seasonal"], 0)


class TestSeasonalityStrength:
    def test_strong_seasonality(self) -> None:
        t = np.arange(48)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        remainder = np.random.default_rng(0).normal(0, 0.1, 48)
        ss = seasonality_strength(seasonal, remainder)
        assert ss > 0.9

    def test_no_seasonality(self) -> None:
        remainder = np.random.default_rng(0).normal(0, 1, 48)
        seasonal = np.zeros(48)
        ss = seasonality_strength(seasonal, remainder)
        assert ss < 0.1

    def test_bounded(self) -> None:
        seasonal = np.random.default_rng(0).normal(0, 1, 48)
        remainder = np.random.default_rng(1).normal(0, 1, 48)
        ss = seasonality_strength(seasonal, remainder)
        assert 0.0 <= ss <= 1.0


class TestExtractMstlFeatures:
    def test_returns_dataframe(self, panel_df) -> None:
        result = extract_mstl_features(panel_df, "article", "sales", freq="MS")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, panel_df) -> None:
        result = extract_mstl_features(panel_df, "article", "sales", freq="MS")
        assert "seasonality_strength" in result.columns
        assert "trend_strength" in result.columns

    def test_index_is_panel_ids(self, panel_df) -> None:
        result = extract_mstl_features(panel_df, "article", "sales", freq="MS")
        assert set(result.index) == {"A0", "A1", "A2"}

    def test_values_bounded(self, panel_df) -> None:
        result = extract_mstl_features(panel_df, "article", "sales", freq="MS")
        assert (result["seasonality_strength"] >= 0).all()
        assert (result["seasonality_strength"] <= 1).all()


class TestExtractSeasonalVectors:
    def test_returns_dataframe(self, panel_df) -> None:
        result = extract_seasonal_vectors(panel_df, "article", "sales", freq="MS")
        assert isinstance(result, pd.DataFrame)

    def test_vector_length_equals_period(self, panel_df) -> None:
        result = extract_seasonal_vectors(
            panel_df,
            "article",
            "sales",
            freq="MS",
            season_lengths=[12],
        )
        assert result.shape[1] == 12

    def test_index_is_panel_ids(self, panel_df) -> None:
        result = extract_seasonal_vectors(panel_df, "article", "sales", freq="MS")
        assert set(result.index) == {"A0", "A1", "A2"}

    def test_normalized_vectors(self, panel_df) -> None:
        result = extract_seasonal_vectors(panel_df, "article", "sales", freq="MS")
        # После нормализации mean ≈ 0
        for _, row in result.iterrows():
            assert abs(row.mean()) < 0.5
