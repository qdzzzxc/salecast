import datetime as dt

import pandas as pd
import pytest

from src.custom_types import SplitRange, Splits
from src.model_selection import (
    generate_expanding_cv_folds,
    temporal_panel_split,
    temporal_panel_split_by_date,
    temporal_panel_split_by_size,
    temporal_panel_train_test_split,
    temporal_panel_train_val_test_split,
)


def _make_panel(n_panels: int = 3, n_periods: int = 20) -> pd.DataFrame:
    """Создаёт простой панельный датафрейм для тестов."""
    ids = [f"P{i}" for i in range(n_panels)]
    dates = pd.date_range("2021-01-01", periods=n_periods, freq="MS")
    rows = [{"id": pid, "date": d, "val": float(i)} for pid in ids for i, d in enumerate(dates)]
    return pd.DataFrame(rows)


class TestTemporalPanelTrainTestSplit:
    def test_basic_split_covers_all_rows(self) -> None:
        """Все строки попадают в train или test."""
        df = _make_panel()
        splits = temporal_panel_train_test_split(df, "id", "date")
        assert len(splits.train) + len(splits.test) == len(df)

    def test_val_is_none(self) -> None:
        """Val всегда None в train/test сплите."""
        df = _make_panel()
        splits = temporal_panel_train_test_split(df, "id", "date")
        assert splits.val is None

    def test_temporal_ordering_per_panel(self) -> None:
        """Последняя дата в train строго меньше первой даты в test для каждой панели."""
        df = _make_panel()
        splits = temporal_panel_train_test_split(df, "id", "date")
        for panel_id in df["id"].unique():
            train_dates = splits.train[splits.train["id"] == panel_id]["date"]
            test_dates = splits.test[splits.test["id"] == panel_id]["date"]
            assert train_dates.max() < test_dates.min()

    def test_raises_on_bad_ratio_above_one(self) -> None:
        """Коэффициент train_ratio > 1 вызывает ValueError."""
        df = _make_panel()
        with pytest.raises(ValueError):
            temporal_panel_train_test_split(df, "id", "date", train_ratio=1.1)

    def test_raises_on_bad_ratio_zero(self) -> None:
        """Коэффициент train_ratio = 0 вызывает ValueError."""
        df = _make_panel()
        with pytest.raises(ValueError):
            temporal_panel_train_test_split(df, "id", "date", train_ratio=0.0)

    def test_raises_when_too_little_data(self) -> None:
        """Панель из одной точки вызывает ValueError при любом train_ratio."""
        df = _make_panel(n_panels=1, n_periods=1)
        with pytest.raises(ValueError):
            temporal_panel_train_test_split(df, "id", "date", train_ratio=0.7)

    def test_returns_splits_instance(self) -> None:
        """Функция возвращает объект типа Splits."""
        df = _make_panel()
        result = temporal_panel_train_test_split(df, "id", "date")
        assert isinstance(result, Splits)


class TestTemporalPanelTrainValTestSplit:
    def test_basic_split_covers_all_rows(self) -> None:
        """Все строки попадают в train, val или test."""
        df = _make_panel()
        splits = temporal_panel_train_val_test_split(df, "id", "date")
        assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)

    def test_val_is_not_none(self) -> None:
        """Val не равен None после train/val/test сплита."""
        df = _make_panel()
        splits = temporal_panel_train_val_test_split(df, "id", "date")
        assert splits.val is not None

    def test_temporal_ordering_per_panel(self) -> None:
        """Даты идут в порядке train < val < test для каждой панели."""
        df = _make_panel()
        splits = temporal_panel_train_val_test_split(df, "id", "date")
        for panel_id in df["id"].unique():
            train_dates = splits.train[splits.train["id"] == panel_id]["date"]
            val_dates = splits.val[splits.val["id"] == panel_id]["date"]
            test_dates = splits.test[splits.test["id"] == panel_id]["date"]
            assert train_dates.max() < val_dates.min()
            assert val_dates.max() < test_dates.min()

    def test_raises_on_ratios_sum_exceeds_one(self) -> None:
        """Сумма train_ratio + val_ratio >= 1 вызывает ValueError."""
        df = _make_panel()
        with pytest.raises(ValueError):
            temporal_panel_train_val_test_split(df, "id", "date", train_ratio=0.7, val_ratio=0.4)

    def test_raises_on_zero_val_ratio(self) -> None:
        """val_ratio = 0 вызывает ValueError."""
        df = _make_panel()
        with pytest.raises(ValueError):
            temporal_panel_train_val_test_split(df, "id", "date", train_ratio=0.7, val_ratio=0.0)


class TestTemporalPanelSplit:
    def test_without_val_ratio_returns_none_val(self) -> None:
        """Без val_ratio val равен None."""
        df = _make_panel()
        splits = temporal_panel_split(df, "id", "date", train_ratio=0.7)
        assert splits.val is None

    def test_with_val_ratio_returns_val(self) -> None:
        """С val_ratio val не равен None."""
        df = _make_panel()
        splits = temporal_panel_split(df, "id", "date", train_ratio=0.7, val_ratio=0.15)
        assert splits.val is not None

    def test_with_val_ratio_covers_all_rows(self) -> None:
        """Все строки распределяются по сплитам при наличии val_ratio."""
        df = _make_panel()
        splits = temporal_panel_split(df, "id", "date", train_ratio=0.7, val_ratio=0.15)
        assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)


class TestTemporalPanelSplitByDate:
    def test_without_val_range(self) -> None:
        """Без val_range val равен None, train и test содержат строки."""
        df = _make_panel(n_periods=24)
        train_range = SplitRange(start=dt.date(2021, 1, 1), end=dt.date(2022, 6, 1))
        test_range = SplitRange(start=dt.date(2022, 7, 1), end=dt.date(2022, 12, 1))
        splits = temporal_panel_split_by_date(df, "id", "date", train_range, test_range)
        assert splits.val is None
        assert len(splits.train) > 0
        assert len(splits.test) > 0

    def test_with_val_range(self) -> None:
        """С val_range val не равен None и содержит строки."""
        df = _make_panel(n_periods=24)
        train_range = SplitRange(start=dt.date(2021, 1, 1), end=dt.date(2021, 12, 1))
        val_range = SplitRange(start=dt.date(2022, 1, 1), end=dt.date(2022, 6, 1))
        test_range = SplitRange(start=dt.date(2022, 7, 1), end=dt.date(2022, 12, 1))
        splits = temporal_panel_split_by_date(
            df, "id", "date", train_range, test_range, val_range=val_range
        )
        assert splits.val is not None
        assert len(splits.val) > 0

    def test_date_ranges_are_non_overlapping(self) -> None:
        """Даты в train не пересекаются с датами в test."""
        df = _make_panel(n_periods=24)
        train_range = SplitRange(start=dt.date(2021, 1, 1), end=dt.date(2021, 12, 1))
        test_range = SplitRange(start=dt.date(2022, 7, 1), end=dt.date(2022, 12, 1))
        splits = temporal_panel_split_by_date(df, "id", "date", train_range, test_range)
        train_dates = set(splits.train["date"].dt.date)
        test_dates = set(splits.test["date"].dt.date)
        assert train_dates.isdisjoint(test_dates)


class TestTemporalPanelSplitBySize:
    def test_basic_split_covers_all_rows(self) -> None:
        """Все строки попадают в train или test."""
        df = _make_panel()
        splits = temporal_panel_split_by_size(df, "id", "date", test_size=3)
        assert len(splits.train) + len(splits.test) == len(df)

    def test_test_size_correct(self) -> None:
        """Каждая панель имеет ровно test_size строк в test."""
        df = _make_panel(n_panels=3, n_periods=20)
        test_size = 4
        splits = temporal_panel_split_by_size(df, "id", "date", test_size=test_size)
        for panel_id in df["id"].unique():
            panel_test = splits.test[splits.test["id"] == panel_id]
            assert len(panel_test) == test_size

    def test_with_val_size(self) -> None:
        """Val не равен None и имеет val_size строк на панель при val_size > 0."""
        df = _make_panel(n_panels=3, n_periods=20)
        val_size = 3
        splits = temporal_panel_split_by_size(df, "id", "date", test_size=3, val_size=val_size)
        assert splits.val is not None
        for panel_id in df["id"].unique():
            panel_val = splits.val[splits.val["id"] == panel_id]
            assert len(panel_val) == val_size

    def test_without_val_size_val_is_none(self) -> None:
        """Без val_size val равен None."""
        df = _make_panel()
        splits = temporal_panel_split_by_size(df, "id", "date", test_size=3)
        assert splits.val is None

    def test_raises_on_test_size_zero(self) -> None:
        """test_size < 1 вызывает ValueError."""
        df = _make_panel()
        with pytest.raises(ValueError):
            temporal_panel_split_by_size(df, "id", "date", test_size=0)

    def test_raises_on_val_size_zero(self) -> None:
        """val_size < 1 вызывает ValueError."""
        df = _make_panel()
        with pytest.raises(ValueError):
            temporal_panel_split_by_size(df, "id", "date", test_size=3, val_size=0)

    def test_raises_when_not_enough_data(self) -> None:
        """Слишком большой test_size относительно данных вызывает ValueError."""
        df = _make_panel(n_panels=1, n_periods=3)
        with pytest.raises(ValueError):
            temporal_panel_split_by_size(df, "id", "date", test_size=3)


class TestGenerateExpandingCVFolds:
    def test_returns_correct_number_of_folds(self) -> None:
        df = _make_panel(n_panels=2, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        assert len(folds) == 3

    def test_val_is_none_for_all_folds(self) -> None:
        df = _make_panel(n_panels=2, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        for fold in folds:
            assert fold.val is None

    def test_train_grows_across_folds(self) -> None:
        """Train size должен расти от fold к fold."""
        df = _make_panel(n_panels=2, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        train_sizes = [len(f.train) for f in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_test_size_is_consistent(self) -> None:
        """Test size одинаковый для всех фолдов."""
        df = _make_panel(n_panels=2, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        test_sizes = [len(f.test) for f in folds]
        assert len(set(test_sizes)) == 1

    def test_no_overlap_train_test(self) -> None:
        """Даты train и test не пересекаются."""
        df = _make_panel(n_panels=2, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        for fold in folds:
            train_dates = set(fold.train["date"])
            test_dates = set(fold.test["date"])
            assert train_dates.isdisjoint(test_dates)

    def test_temporal_ordering(self) -> None:
        """Все даты train < все даты test."""
        df = _make_panel(n_panels=2, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        for fold in folds:
            assert fold.train["date"].max() < fold.test["date"].min()

    def test_all_panels_present_in_each_fold(self) -> None:
        """Каждая панель есть в train и test каждого fold."""
        df = _make_panel(n_panels=3, n_periods=20)
        folds = generate_expanding_cv_folds(df, n_folds=3, panel_column="id", time_column="date")
        expected_panels = set(df["id"].unique())
        for fold in folds:
            assert set(fold.train["id"].unique()) == expected_panels
            assert set(fold.test["id"].unique()) == expected_panels

    def test_raises_on_n_folds_less_than_2(self) -> None:
        df = _make_panel()
        with pytest.raises(ValueError):
            generate_expanding_cv_folds(df, n_folds=1, panel_column="id", time_column="date")

    def test_raises_on_bad_min_train_ratio(self) -> None:
        df = _make_panel()
        with pytest.raises(ValueError):
            generate_expanding_cv_folds(
                df, n_folds=3, panel_column="id", time_column="date", min_train_ratio=0.0
            )

    def test_raises_on_insufficient_data(self) -> None:
        """Слишком мало дат для запрошенного числа фолдов."""
        df = _make_panel(n_panels=1, n_periods=4)
        with pytest.raises(ValueError):
            generate_expanding_cv_folds(df, n_folds=5, panel_column="id", time_column="date")
