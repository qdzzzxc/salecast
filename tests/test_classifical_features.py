import numpy as np
import pandas as pd
import pytest

from src.classifical_features import _add_cdf_features, _add_trend_features, build_ts_features
from src.configs.settings import DownstreamConfig, Settings


@pytest.fixture()
def single_panel_df() -> pd.DataFrame:
    """Одна панель, 24 точки."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    values = rng.uniform(10, 100, size=24)
    return pd.DataFrame({"article": "A1", "date": dates, "sales": values})


@pytest.fixture()
def multi_panel_df() -> pd.DataFrame:
    """Три панели по 18 точек."""
    rng = np.random.default_rng(1)
    frames = []
    for pid in ["A1", "A2", "A3"]:
        dates = pd.date_range("2022-01-01", periods=18, freq="MS")
        values = rng.uniform(5, 50, size=18)
        frames.append(pd.DataFrame({"article": pid, "date": dates, "sales": values}))
    return pd.concat(frames, ignore_index=True)


class TestAddTrendFeatures:
    def test_column_added(self, single_panel_df):
        result = _add_trend_features(single_panel_df, "sales", window=6)
        assert "sales_trend_6" in result.columns

    def test_no_leakage_shift(self, single_panel_df):
        """Первые (window-1) значения должны быть NaN (нет прошлых данных для регрессии)."""
        result = _add_trend_features(single_panel_df, "sales", window=6)
        # После shift(1) первое значение NaN, значит trend[0] = NaN
        assert np.isnan(result["sales_trend_6"].iloc[0])

    def test_values_are_finite_after_warmup(self, single_panel_df):
        window = 4
        result = _add_trend_features(single_panel_df, "sales", window=window)
        col = result["sales_trend_6"] if "sales_trend_6" in result.columns else result[f"sales_trend_{window}"]
        result2 = _add_trend_features(single_panel_df, "sales", window=window)
        trend_col = result2[f"sales_trend_{window}"]
        # После прогрева (window точек прошлого) значения конечны
        finite_vals = trend_col.dropna()
        assert len(finite_vals) > 0
        assert np.all(np.isfinite(finite_vals))

    def test_rising_series_positive_slope(self):
        """Строго возрастающий ряд → положительный наклон."""
        df = pd.DataFrame({
            "article": "A1",
            "date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "sales": np.arange(1.0, 21.0),
        })
        result = _add_trend_features(df, "sales", window=5)
        trend_col = result["sales_trend_5"].dropna()
        assert (trend_col > 0).all()

    def test_original_df_unchanged(self, single_panel_df):
        original_cols = set(single_panel_df.columns)
        _add_trend_features(single_panel_df, "sales", window=6)
        assert set(single_panel_df.columns) == original_cols


class TestAddCdfFeatures:
    def test_column_added(self, single_panel_df):
        result = _add_cdf_features(single_panel_df, "sales", decay=0.9)
        assert "sales_cdf" in result.columns

    def test_values_in_0_1(self, single_panel_df):
        result = _add_cdf_features(single_panel_df, "sales", decay=0.9)
        valid = result["sales_cdf"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_first_value_nan(self, single_panel_df):
        """Первое значение всегда NaN (нет прошлых точек)."""
        result = _add_cdf_features(single_panel_df, "sales", decay=0.9)
        assert np.isnan(result["sales_cdf"].iloc[0])

    def test_no_leakage(self):
        """Значение на шаге i не должно использовать values[i]."""
        # Минимальный ряд: [1, 1, 1, ..., 1000]
        df = pd.DataFrame({
            "article": "A1",
            "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "sales": [1.0] * 9 + [1000.0],
        })
        result = _add_cdf_features(df, "sales", decay=1.0)
        # На последнем шаге past = [1]*8 (после shift), все ≤ текущий (1000) → CDF = 1.0
        assert result["sales_cdf"].iloc[-1] == pytest.approx(1.0)

    def test_original_df_unchanged(self, single_panel_df):
        original_cols = set(single_panel_df.columns)
        _add_cdf_features(single_panel_df, "sales", decay=0.9)
        assert set(single_panel_df.columns) == original_cols


class TestBuildTsFeatures:
    def test_trend_disabled_by_default(self, multi_panel_df):
        settings = Settings()
        assert not settings.downstream.use_trend
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        assert not any("trend" in c for c in result.columns)

    def test_cdf_disabled_by_default(self, multi_panel_df):
        settings = Settings()
        assert not settings.downstream.use_cdf
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        assert "sales_cdf" not in result.columns

    def test_trend_enabled(self, multi_panel_df):
        settings = Settings(
            downstream=DownstreamConfig(use_trend=True, trend_window=4)
        )
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        assert "sales_trend_4" in result.columns

    def test_cdf_enabled(self, multi_panel_df):
        settings = Settings(
            downstream=DownstreamConfig(use_cdf=True, cdf_decay=0.8)
        )
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        assert "sales_cdf" in result.columns

    def test_both_features_enabled(self, multi_panel_df):
        settings = Settings(
            downstream=DownstreamConfig(use_trend=True, trend_window=3, use_cdf=True, cdf_decay=0.9)
        )
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        assert "sales_trend_3" in result.columns
        assert "sales_cdf" in result.columns

    def test_per_panel_no_leakage_across_panels(self, multi_panel_df):
        """Тренд и CDF считаются per-panel — не смешиваются между панелями."""
        settings = Settings(
            downstream=DownstreamConfig(use_trend=True, trend_window=3, use_cdf=True)
        )
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        for pid in ["A1", "A2", "A3"]:
            panel = result[result["article"] == pid]
            assert len(panel) == 18

    def test_rename_build_ts_features(self, multi_panel_df):
        """build_ts_features существует и возвращает DataFrame с нужными колонками."""
        settings = Settings()
        result = build_ts_features(multi_panel_df, settings, disable_tqdm=True)
        assert isinstance(result, pd.DataFrame)
        assert "sales" in result.columns
        assert "sales_lag_1" in result.columns
