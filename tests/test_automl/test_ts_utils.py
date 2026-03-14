import pandas as pd
import pytest

from src.automl.ts_utils import _normalize_freq, get_downstream_lags, infer_ts_config
from src.configs.settings import TimeSeriesConfig


def _make_df(freq: str, periods: int = 30) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=periods, freq=freq)})


class TestNormalizeFreq:
    def test_monthly_start(self) -> None:
        assert _normalize_freq("MS") == "MS"

    def test_monthly_end(self) -> None:
        assert _normalize_freq("ME") == "MS"

    def test_monthly_legacy(self) -> None:
        assert _normalize_freq("M") == "MS"

    def test_weekly_variants(self) -> None:
        assert _normalize_freq("W-SUN") == "W"
        assert _normalize_freq("W-MON") == "W"
        assert _normalize_freq("W") == "W"

    def test_quarterly_variants(self) -> None:
        assert _normalize_freq("QS") == "Q"
        assert _normalize_freq("QE") == "Q"
        assert _normalize_freq("QS-JAN") == "Q"

    def test_annual_variants(self) -> None:
        assert _normalize_freq("AS") == "A"
        assert _normalize_freq("YS") == "A"
        assert _normalize_freq("A") == "A"

    def test_daily(self) -> None:
        assert _normalize_freq("D") == "D"

    def test_business_daily(self) -> None:
        assert _normalize_freq("B") == "B"


class TestInferTsConfig:
    def test_returns_time_series_config(self) -> None:
        df = _make_df("MS")
        result = infer_ts_config(df, "date")
        assert isinstance(result, TimeSeriesConfig)

    def test_monthly_season_length(self) -> None:
        df = _make_df("MS")
        result = infer_ts_config(df, "date")
        assert result.season_length == 12

    def test_daily_season_length(self) -> None:
        df = _make_df("D", periods=365)
        result = infer_ts_config(df, "date")
        assert result.season_length == 7

    def test_weekly_season_length(self) -> None:
        df = _make_df("W", periods=52)
        result = infer_ts_config(df, "date")
        assert result.season_length == 52

    def test_quarterly_season_length(self) -> None:
        df = _make_df("QS", periods=12)
        result = infer_ts_config(df, "date")
        assert result.season_length == 4

    def test_returns_default_on_irregular_dates(self) -> None:
        """Нерегулярные даты → дефолтный MS/12."""
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-03", "2020-01-07", "2020-02-15"]})
        result = infer_ts_config(df, "date")
        assert result.freq == "MS"
        assert result.season_length == 12

    def test_freq_field_is_set(self) -> None:
        df = _make_df("MS")
        result = infer_ts_config(df, "date")
        assert result.freq != ""


class TestGetDownstreamLags:
    def test_monthly_lags(self) -> None:
        lags = get_downstream_lags("MS")
        assert 12 in lags
        assert 1 in lags

    def test_daily_lags(self) -> None:
        lags = get_downstream_lags("D")
        assert 7 in lags
        assert 365 in lags

    def test_weekly_lags(self) -> None:
        lags = get_downstream_lags("W")
        assert 52 in lags

    def test_quarterly_lags(self) -> None:
        lags = get_downstream_lags("Q")
        assert 4 in lags

    def test_unknown_freq_returns_defaults(self) -> None:
        lags = get_downstream_lags("UNKNOWN")
        assert isinstance(lags, list)
        assert len(lags) > 0

    def test_lags_are_sorted_ascending(self) -> None:
        for freq in ("D", "W", "MS", "Q"):
            lags = get_downstream_lags(freq)
            assert lags == sorted(lags)
