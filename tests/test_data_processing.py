import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    aggregate_by_panel_date,
    clip_panel_outliers,
    count_outliers,
    drop_duplicates,
    expand_to_full_panel,
    filter_panels_by_split_missing,
    filter_sellers_by_min_periods,
    find_trim_indices,
    fit_panel_scalers,
    inverse_transform_panel_columns,
    scale_panel_splits,
    sort_panel_by_date,
    transform_panel_columns,
)


def _make_panel(n_panels: int = 3, n_periods: int = 12, start: str = "2021-01-01") -> pd.DataFrame:
    """Создаёт панельный датафрейм с тремя панелями и месячными датами."""
    ids = [f"P{i}" for i in range(n_panels)]
    dates = pd.date_range(start, periods=n_periods, freq="MS")
    rows = [
        {"article": pid, "date": d, "sales": float(i % 5 + 1)}
        for pid in ids
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


class TestDropDuplicates:
    def test_removes_exact_duplicates(self) -> None:
        """Точные дубликаты строк удаляются."""
        df = _make_panel()
        df_with_dupes = pd.concat([df, df.iloc[:5]], ignore_index=True)
        result = drop_duplicates(df_with_dupes)
        assert len(result) == len(df)

    def test_no_dupes_returns_same_length(self) -> None:
        """При отсутствии дубликатов длина датафрейма не меняется."""
        df = _make_panel()
        result = drop_duplicates(df)
        assert len(result) == len(df)


class TestAggregateByPanelDate:
    def test_sums_duplicate_rows(self) -> None:
        """Дублирующиеся строки суммируются по целевой колонке."""
        df = _make_panel(n_panels=1, n_periods=6)
        extra = df.copy()
        combined = pd.concat([df, extra], ignore_index=True)
        result = aggregate_by_panel_date(combined, "article", "date", ["sales"])
        expected_sales = df.set_index(["article", "date"])["sales"] * 2
        result_sales = result.set_index(["article", "date"])["sales"].sort_index()
        expected_sales = expected_sales.sort_index()
        pd.testing.assert_series_equal(result_sales, expected_sales, check_names=False)

    def test_deduplicates_rows(self) -> None:
        """После агрегации количество строк не превышает число уникальных (panel, date) пар."""
        df = _make_panel(n_panels=2, n_periods=6)
        extra = df.iloc[:3].copy()
        combined = pd.concat([df, extra], ignore_index=True)
        result = aggregate_by_panel_date(combined, "article", "date", ["sales"])
        assert len(result) == len(df)


class TestExpandToFullPanel:
    def test_fills_missing_combinations(self) -> None:
        """Пропущенные комбинации panel × date заполняются NaN."""
        df = _make_panel(n_panels=2, n_periods=6)
        sparse_df = df.iloc[:-3].copy()
        result = expand_to_full_panel(sparse_df, panel_column="article", date_column="date")
        n_panels = sparse_df["article"].nunique()
        n_dates = sparse_df["date"].nunique()
        assert len(result) == n_panels * n_dates

    def test_no_new_rows_for_full_panel(self) -> None:
        """Полный панельный датафрейм не получает лишних строк."""
        df = _make_panel(n_panels=2, n_periods=6)
        result = expand_to_full_panel(df, panel_column="article", date_column="date")
        assert len(result) == len(df)


class TestFilterSellersByMinPeriods:
    def test_removes_panels_with_too_few_rows(self) -> None:
        """Панели с количеством строк ниже порога удаляются."""
        df = _make_panel(n_panels=2, n_periods=10)
        small_panel = pd.DataFrame(
            [{"article": "SMALL", "date": pd.Timestamp("2021-01-01"), "sales": 1.0}]
        )
        combined = pd.concat([df, small_panel], ignore_index=True)
        result = filter_sellers_by_min_periods(combined, panel_column="article", min_periods=5)
        assert "SMALL" not in result["article"].values

    def test_keeps_panels_with_enough_rows(self) -> None:
        """Панели с достаточным количеством строк остаются."""
        df = _make_panel(n_panels=3, n_periods=10)
        result = filter_sellers_by_min_periods(df, panel_column="article", min_periods=5)
        assert result["article"].nunique() == 3


class TestSortPanelByDate:
    def test_sorts_correctly(self) -> None:
        """Датафрейм сортируется по (panel, date) в возрастающем порядке."""
        df = _make_panel(n_panels=2, n_periods=6)
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        result = sort_panel_by_date(shuffled, panel_column="article", date_column="date")
        for panel_id in result["article"].unique():
            panel_dates = result[result["article"] == panel_id]["date"].reset_index(drop=True)
            assert panel_dates.is_monotonic_increasing


class TestFilterPanelsBySplitMissing:
    def test_drops_panel_with_high_nan_ratio(self) -> None:
        """Панель с долей NaN выше порога удаляется из всех сплитов."""
        good = _make_panel(n_panels=1, n_periods=12)
        good["article"] = "GOOD"

        bad_train = pd.DataFrame(
            [{"article": "BAD", "date": pd.Timestamp("2021-01-01"), "sales": np.nan}]
        )
        bad_test = pd.DataFrame(
            [
                {"article": "BAD", "date": pd.Timestamp("2021-07-01"), "sales": np.nan},
                {"article": "BAD", "date": pd.Timestamp("2021-08-01"), "sales": np.nan},
            ]
        )
        train = pd.concat([good.iloc[:6], bad_train], ignore_index=True)
        test = pd.concat([good.iloc[6:], bad_test], ignore_index=True)

        result_train, result_val, result_test = filter_panels_by_split_missing(
            (train, None, test), "article", ["sales"], maximum_missing_ratio=0.3
        )
        assert "BAD" not in result_train["article"].values
        assert "BAD" not in result_test["article"].values

    def test_keeps_panel_without_nans(self) -> None:
        """Панель без NaN сохраняется после фильтрации."""
        df = _make_panel(n_panels=2, n_periods=12)
        train = df.iloc[:12].copy()
        test = df.iloc[12:].copy()
        result_train, _, result_test = filter_panels_by_split_missing(
            (train, None, test), "article", ["sales"]
        )
        assert len(result_train["article"].unique()) > 0

    def test_treat_zero_as_missing(self) -> None:
        """При treat_zero_as_missing=True нули считаются пропусками."""
        good = pd.DataFrame(
            [
                {"article": "GOOD", "date": pd.Timestamp(f"2021-{m:02d}-01"), "sales": float(m)}
                for m in range(1, 7)
            ]
        )
        bad = pd.DataFrame(
            [
                {"article": "ZEROS", "date": pd.Timestamp(f"2021-{m:02d}-01"), "sales": 0.0}
                for m in range(1, 7)
            ]
        )
        train = pd.concat([good, bad], ignore_index=True)
        test = pd.concat([good, bad], ignore_index=True)

        result_train, _, result_test = filter_panels_by_split_missing(
            (train, None, test),
            "article",
            ["sales"],
            maximum_missing_ratio=0.5,
            treat_zero_as_missing=True,
        )
        assert "ZEROS" not in result_train["article"].values


class TestClipPanelOutliers:
    def test_clips_extreme_values(self) -> None:
        """Экстремальные значения обрезаются до диапазона перцентилей."""
        df = _make_panel(n_panels=2, n_periods=30)
        df.loc[0, "sales"] = 9999.0
        train = df.iloc[:20].copy()
        test = df.iloc[20:].copy()
        result = clip_panel_outliers(
            (train, None, test),
            panel_column="article",
            target_columns=["sales"],
            lower_percentile=1.0,
            upper_percentile=99.0,
            min_panel_size=5,
        )
        assert result.train["sales"].max() < 9999.0

    def test_values_within_bounds(self) -> None:
        """После клиппинга все значения находятся в допустимом диапазоне."""
        df = _make_panel(n_panels=2, n_periods=30)
        df.loc[0, "sales"] = 9999.0
        train = df.iloc[:20].copy()
        test = df.iloc[20:].copy()
        result = clip_panel_outliers(
            (train, None, test),
            panel_column="article",
            target_columns=["sales"],
            min_panel_size=5,
        )
        assert result.train["sales"].max() < 9999.0


class TestFindTrimIndices:
    def test_returns_correct_indices_for_mixed_series(self) -> None:
        """Возвращает правильные индексы первого и последнего ненулевого значения."""
        s = pd.Series([0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0])
        first, last = find_trim_indices(s)
        assert first == 2
        assert last == 5

    def test_returns_none_none_for_all_zero_series(self) -> None:
        """Для полностью нулевого ряда возвращается (None, None)."""
        s = pd.Series([0.0, 0.0, 0.0])
        first, last = find_trim_indices(s)
        assert first is None
        assert last is None

    def test_single_nonzero_value(self) -> None:
        """Единственное ненулевое значение: first == last."""
        s = pd.Series([0.0, 5.0, 0.0])
        first, last = find_trim_indices(s)
        assert first == last == 1


class TestCountOutliers:
    def test_returns_zero_for_uniform_data(self) -> None:
        """Для однородных данных без выбросов возвращается 0."""
        s = pd.Series([5.0] * 20)
        assert count_outliers(s) == 0

    def test_returns_positive_for_data_with_outliers(self) -> None:
        """Для данных с явными выбросами возвращается положительное число."""
        normal = pd.Series([1.0, 2.0, 1.5, 2.0, 1.0] * 10)
        with_outlier = pd.concat([normal, pd.Series([10000.0])], ignore_index=True)
        assert count_outliers(with_outlier) > 0


class TestFitPanelScalers:
    def test_returns_scaler_per_panel(self) -> None:
        """fit_panel_scalers возвращает скейлер для каждой панели."""
        df = _make_panel(n_panels=3, n_periods=12)
        scalers = fit_panel_scalers(df, "article", ["sales"])
        assert set(scalers["sales"].keys()) == set(df["article"].unique())

    def test_scaler_mean_close_to_zero_after_transform(self) -> None:
        """После обучения и применения среднее по панели ≈ 0."""
        df = _make_panel(n_panels=2, n_periods=24)
        scalers = fit_panel_scalers(df, "article", ["sales"])
        transformed = transform_panel_columns(df, scalers, "article", ["sales"])
        for panel_id, group in transformed.groupby("article"):
            assert abs(group["sales"].mean()) < 0.1

    def test_apply_log_changes_scale(self) -> None:
        """apply_log=True логарифмирует данные перед масштабированием."""
        df = _make_panel(n_panels=2, n_periods=12)
        scalers_no_log = fit_panel_scalers(df, "article", ["sales"], apply_log=False)
        scalers_log = fit_panel_scalers(df, "article", ["sales"], apply_log=True)
        panel_id = df["article"].iloc[0]
        mean_no_log = scalers_no_log["sales"][panel_id].mean_[0]
        mean_log = scalers_log["sales"][panel_id].mean_[0]
        assert mean_no_log != mean_log


class TestTransformPanelColumns:
    def test_raises_for_unknown_panel(self) -> None:
        """ValueError при наличии в df панели, отсутствующей в scalers."""
        train = _make_panel(n_panels=2, n_periods=12)
        scalers = fit_panel_scalers(train, "article", ["sales"])
        new_panel = pd.DataFrame(
            [{"article": "UNKNOWN", "date": pd.Timestamp("2021-01-01"), "sales": 5.0}]
        )
        with pytest.raises(ValueError, match="Scaler not found"):
            transform_panel_columns(new_panel, scalers, "article", ["sales"])

    def test_does_not_mutate_original(self) -> None:
        """transform_panel_columns не изменяет оригинальный датафрейм."""
        df = _make_panel(n_panels=2, n_periods=12)
        scalers = fit_panel_scalers(df, "article", ["sales"])
        original_values = df["sales"].copy()
        transform_panel_columns(df, scalers, "article", ["sales"])
        pd.testing.assert_series_equal(df["sales"], original_values)


class TestInverseTransformPanelColumns:
    def test_roundtrip_restores_original(self) -> None:
        """transform + inverse_transform восстанавливает исходные значения."""
        df = _make_panel(n_panels=3, n_periods=12)
        scalers = fit_panel_scalers(df, "article", ["sales"])
        transformed = transform_panel_columns(df, scalers, "article", ["sales"])
        restored = inverse_transform_panel_columns(transformed, scalers, "article", ["sales"])
        pd.testing.assert_series_equal(
            df["sales"].reset_index(drop=True),
            restored["sales"].reset_index(drop=True),
            check_names=False,
            atol=1e-6,
        )

    def test_roundtrip_with_log(self) -> None:
        """Логарифмический round-trip восстанавливает исходные значения."""
        df = _make_panel(n_panels=2, n_periods=12)
        scalers = fit_panel_scalers(df, "article", ["sales"], apply_log=True)
        transformed = transform_panel_columns(df, scalers, "article", ["sales"], apply_log=True)
        restored = inverse_transform_panel_columns(
            transformed, scalers, "article", ["sales"], apply_log=True
        )
        pd.testing.assert_series_equal(
            df["sales"].reset_index(drop=True),
            restored["sales"].reset_index(drop=True),
            check_names=False,
            atol=1e-6,
        )


class TestScalePanelSplits:
    def _split_by_date(self, df: pd.DataFrame, train_end: str, val_end: str | None = None):
        train = df[df["date"] < train_end].copy()
        if val_end:
            val = df[(df["date"] >= train_end) & (df["date"] < val_end)].copy()
            test = df[df["date"] >= val_end].copy()
            return train, val, test
        test = df[df["date"] >= train_end].copy()
        return train, None, test

    def test_returns_scaled_splits(self) -> None:
        """scale_panel_splits возвращает ScaledSplits с train, test и scalers."""
        df = _make_panel(n_panels=2, n_periods=24)
        train, _, test = self._split_by_date(df, "2022-01-01")
        result = scale_panel_splits((train, None, test), "article", ["sales"])
        assert result.train is not None
        assert result.test is not None
        assert result.scalers is not None

    def test_val_none_handled(self) -> None:
        """Если val=None, результирующий val тоже None."""
        df = _make_panel(n_panels=2, n_periods=24)
        train, _, test = self._split_by_date(df, "2022-01-01")
        result = scale_panel_splits((train, None, test), "article", ["sales"])
        assert result.val is None

    def test_val_is_transformed(self) -> None:
        """Если val передан, он трансформируется теми же скейлерами."""
        df = _make_panel(n_panels=2, n_periods=24)
        train, val, test = self._split_by_date(df, "2022-01-01", "2022-07-01")
        result = scale_panel_splits((train, val, test), "article", ["sales"])
        assert result.val is not None
        assert len(result.val) == len(val)

    def test_scalers_fit_on_train_only(self) -> None:
        """Скейлеры содержат запись для каждой панели из train."""
        df = _make_panel(n_panels=2, n_periods=24)
        train, _, test = self._split_by_date(df, "2022-01-01")
        result = scale_panel_splits((train, None, test), "article", ["sales"])
        for panel_id in train["article"].unique():
            assert panel_id in result.scalers["sales"]


class TestClipPanelOutliersWithVal:
    def test_val_is_clipped_too(self) -> None:
        """Если val передан, его значения тоже клиппируются."""
        df = _make_panel(n_panels=2, n_periods=60)
        df.loc[df.index[0], "sales"] = 99999.0
        train = df[df["date"] < "2023-01-01"].copy()
        val = df[(df["date"] >= "2023-01-01") & (df["date"] < "2024-01-01")].copy()
        test = df[df["date"] >= "2024-01-01"].copy()
        result = clip_panel_outliers(
            (train, val, test),
            panel_column="article",
            target_columns=["sales"],
            min_panel_size=5,
        )
        assert result.val is not None
        assert result.val["sales"].max() < 99999.0
