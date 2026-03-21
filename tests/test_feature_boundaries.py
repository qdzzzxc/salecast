"""Тесты корректности признаков на границах train/val/test сплитов."""

import numpy as np
import pandas as pd
import pytest

from src.classifical_features import build_ts_features
from src.configs.settings import Settings
from src.custom_types import Splits


@pytest.fixture
def two_panel_splits() -> Splits[pd.DataFrame]:
    """train=8, val=2, test=2 для двух панелей с разным масштабом."""
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
    df = pd.concat([a1, a2], ignore_index=True)
    train_parts, val_parts, test_parts = [], [], []
    for _, grp in df.groupby("article"):
        grp = grp.sort_values("date")
        train_parts.append(grp.iloc[:8])
        val_parts.append(grp.iloc[8:10])
        test_parts.append(grp.iloc[10:12])
    return Splits(
        train=pd.concat(train_parts, ignore_index=True),
        val=pd.concat(val_parts, ignore_index=True),
        test=pd.concat(test_parts, ignore_index=True),
    )


def _build_split_features(splits: Splits[pd.DataFrame]) -> pd.DataFrame:
    _SPLIT_COL = "_split"
    parts = [splits.train.copy().assign(**{_SPLIT_COL: "train"})]
    if splits.val is not None:
        parts.append(splits.val.copy().assign(**{_SPLIT_COL: "val"}))
    parts.append(splits.test.copy().assign(**{_SPLIT_COL: "test"}))
    settings = Settings()
    full = build_ts_features(pd.concat(parts, ignore_index=True), settings, disable_tqdm=True)
    return full


class TestSplitBoundaries:
    def test_lag1_at_val_boundary(self, two_panel_splits) -> None:
        full = _build_split_features(two_panel_splits)
        for pid in ["A1", "A2"]:
            panel = full[full["article"] == pid].sort_values("date")
            train_rows = panel[panel["_split"] == "train"]
            val_rows = panel[panel["_split"] == "val"]
            last_train_sales = train_rows["sales"].iloc[-1]
            first_val_lag1 = val_rows["sales_lag_1"].iloc[0]
            assert not np.isnan(first_val_lag1)
            assert first_val_lag1 == pytest.approx(last_train_sales)

    def test_lag1_at_test_boundary(self, two_panel_splits) -> None:
        full = _build_split_features(two_panel_splits)
        for pid in ["A1", "A2"]:
            panel = full[full["article"] == pid].sort_values("date")
            val_rows = panel[panel["_split"] == "val"]
            test_rows = panel[panel["_split"] == "test"]
            last_val_sales = val_rows["sales"].iloc[-1]
            first_test_lag1 = test_rows["sales_lag_1"].iloc[0]
            assert not np.isnan(first_test_lag1)
            assert first_test_lag1 == pytest.approx(last_val_sales)

    def test_rolling_at_val_boundary(self, two_panel_splits) -> None:
        full = _build_split_features(two_panel_splits)
        for pid in ["A1", "A2"]:
            panel = full[full["article"] == pid].sort_values("date")
            val_rows = panel[panel["_split"] == "val"]
            assert not np.isnan(val_rows["sales_ma_2"].iloc[0])

    def test_ema_at_val_boundary(self, two_panel_splits) -> None:
        full = _build_split_features(two_panel_splits)
        for pid in ["A1", "A2"]:
            panel = full[full["article"] == pid].sort_values("date")
            val_rows = panel[panel["_split"] == "val"]
            assert not np.isnan(val_rows["sales_ema_2"].iloc[0])

    def test_panels_independent(self, two_panel_splits) -> None:
        full = _build_split_features(two_panel_splits)
        a1 = full[full["article"] == "A1"]
        a1_lags = a1["sales_lag_1"].dropna()
        assert (a1_lags <= 120.0).all()

    def test_no_nan_in_val_test_lags(self, two_panel_splits) -> None:
        full = _build_split_features(two_panel_splits)
        val_test = full[full["_split"].isin(["val", "test"])]
        assert not val_test["sales_lag_1"].isna().any()

    def test_separate_vs_joint_features(self, two_panel_splits) -> None:
        """На val отдельно lag_1[0]=NaN, на объединённом — last train value."""
        settings = Settings()
        separate = build_ts_features(two_panel_splits.val.copy(), settings, disable_tqdm=True)
        joint = _build_split_features(two_panel_splits)
        joint_val = joint[joint["_split"] == "val"]

        for pid in ["A1", "A2"]:
            sep_panel = separate[separate["article"] == pid].sort_values("date")
            joint_panel = joint_val[joint_val["article"] == pid].sort_values("date")
            assert np.isnan(sep_panel["sales_lag_1"].iloc[0])
            assert not np.isnan(joint_panel["sales_lag_1"].iloc[0])
