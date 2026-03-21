import pandas as pd
import pytest

from src.configs.settings import FiltrationConfig
from src.filtration import filter_time_series


def _make_series(article: str, values: list[float], start: str = "2023-01-01") -> pd.DataFrame:
    """Создаёт однопанельный датафрейм с месячными датами."""
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"article": article, "date": dates, "sales": values})


def _concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Объединяет датафреймы и сбрасывает индекс."""
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def config() -> FiltrationConfig:
    """Дефолтный конфиг фильтрации."""
    return FiltrationConfig()


def _normal_values(n: int = 24) -> list[float]:
    """Нормальные значения без нулей."""
    return [float(i % 10 + 1) for i in range(n)]


class TestEdgeZeros:
    def test_all_zeros_dropped(self, config: FiltrationConfig) -> None:
        """Ряд из одних нулей отфильтровывается."""
        df = _make_series("a", [0.0] * 24)
        result = filter_time_series(df, config)
        assert "a" not in result.df["article"].values

    def test_all_zeros_reported_at_edge_zeros_step(self, config: FiltrationConfig) -> None:
        """Ряд из нулей падает на шаге edge_zeros."""
        df = _make_series("a", [0.0] * 24)
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert "a" in report[report["step"] == "edge_zeros"]["panel_id"].values

    def test_leading_trailing_zeros_trimmed_and_series_survives(
        self, config: FiltrationConfig
    ) -> None:
        """Нули по краям обрезаются; ряд с 18+ живыми точками проходит."""
        inner = _normal_values(20)
        values = [0.0, 0.0] + inner + [0.0, 0.0]
        df = _make_series("a", values)
        result = filter_time_series(df, config)
        assert "a" in result.df["article"].values

    def test_trimming_reduces_length_below_min_causes_drop(self, config: FiltrationConfig) -> None:
        """Если после трима длина < min_series_length, ряд отфильтровывается."""
        inner = _normal_values(10)
        values = [0.0] * 5 + inner + [0.0] * 5
        df = _make_series("a", values)
        result = filter_time_series(df, config)
        assert "a" not in result.df["article"].values


class TestInnerZeros:
    def test_high_inner_zero_ratio_dropped(self, config: FiltrationConfig) -> None:
        """Ряд с долей нулей > max_zero_ratio отфильтровывается на inner_zeros."""
        values = [10.0] + [0.0] * 20 + [10.0] + [0.0] + [5.0]
        df = _make_series("a", values)
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert "a" in report[report["step"] == "inner_zeros"]["panel_id"].values

    def test_acceptable_inner_zero_ratio_survives(self, config: FiltrationConfig) -> None:
        """Ряд с долей нулей <= max_zero_ratio проходит фильтр."""
        values = _normal_values(20) + [0.0] * 3 + [5.0]
        df = _make_series("a", values)
        result = filter_time_series(df, config)
        assert "a" in result.df["article"].values


class TestMinLength:
    def test_short_series_dropped(self, config: FiltrationConfig) -> None:
        """Ряд короче min_series_length отфильтровывается."""
        df = _make_series("a", _normal_values(10))
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert "a" in report[report["step"] == "min_length"]["panel_id"].values

    def test_series_at_exact_min_length_survives(self, config: FiltrationConfig) -> None:
        """Ряд длиной ровно min_series_length проходит фильтр."""
        df = _make_series("a", _normal_values(config.min_series_length))
        result = filter_time_series(df, config)
        assert "a" in result.df["article"].values


class TestZeroStd:
    def test_constant_series_dropped(self, config: FiltrationConfig) -> None:
        """Константный ненулевой ряд отфильтровывается на zero_std."""
        df = _make_series("a", [5.0] * 24)
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert "a" in report[report["step"] == "zero_std"]["panel_id"].values

    def test_varying_series_survives(self, config: FiltrationConfig) -> None:
        """Ряд с ненулевым std проходит фильтр."""
        df = _make_series("a", _normal_values(24))
        result = filter_time_series(df, config)
        assert "a" in result.df["article"].values


class TestMinTotal:
    def test_small_total_dropped(self, config: FiltrationConfig) -> None:
        """Ряд с суммой < min_total_sales отфильтровывается."""
        values = [0.2, 0.4] * 12
        df = _make_series("a", values)
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert "a" in report[report["step"] == "min_total"]["panel_id"].values

    def test_sufficient_total_survives(self, config: FiltrationConfig) -> None:
        """Ряд с суммой >= min_total_sales проходит фильтр."""
        df = _make_series("a", _normal_values(24))
        result = filter_time_series(df, config)
        assert "a" in result.df["article"].values


class TestAggregateDuplicates:
    def test_duplicate_rows_are_summed(self, config: FiltrationConfig) -> None:
        """Дубликаты по (article, date) суммируются."""
        base = _make_series("a", _normal_values(24))
        extra = _make_series("a", [1.0] * 3)
        df = _concat(base, extra)
        result = filter_time_series(df, config)
        assert "a" in result.df["article"].values
        first_date = result.df[result.df["article"] == "a"].sort_values("date").iloc[0]
        assert first_date["sales"] == pytest.approx(_normal_values(24)[0] + 1.0)

    def test_duplicate_rows_do_not_expand_series_length(self, config: FiltrationConfig) -> None:
        """После агрегации дубликатов количество точек не увеличивается."""
        base = _make_series("a", _normal_values(24))
        extra = _make_series("a", [1.0] * 3)
        df = _concat(base, extra)
        result = filter_time_series(df, config)
        n_rows = len(result.df[result.df["article"] == "a"])
        assert n_rows == 24


class TestFiltrationResult:
    def test_total_dropped_counts_unique_panels(self, config: FiltrationConfig) -> None:
        """total_dropped считает уникальные отфильтрованные панели."""
        df = _concat(
            _make_series("ok", _normal_values(24)),
            _make_series("zeros", [0.0] * 24),
            _make_series("short", _normal_values(5)),
        )
        result = filter_time_series(df, config)
        assert result.total_dropped == 2

    def test_summary_returns_step_counts(self, config: FiltrationConfig) -> None:
        """summary() возвращает словарь шаг -> количество дропов."""
        df = _concat(
            _make_series("ok", _normal_values(24)),
            _make_series("zeros", [0.0] * 24),
        )
        result = filter_time_series(df, config)
        summary = result.summary()
        assert summary["edge_zeros"] == 1
        assert summary["inner_zeros"] == 0

    def test_to_report_df_columns(self, config: FiltrationConfig) -> None:
        """to_report_df() содержит нужные колонки."""
        df = _make_series("a", _normal_values(24))
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert set(report.columns) == {"panel_id", "step", "reason"}

    def test_passing_series_not_in_report(self, config: FiltrationConfig) -> None:
        """Прошедшие фильтрацию панели не появляются в отчёте."""
        df = _make_series("ok", _normal_values(24))
        result = filter_time_series(df, config)
        report = result.to_report_df()
        assert "ok" not in report["panel_id"].values
