"""Тесты для src/ensemble.py — ансамблирование предсказаний."""

import math

import numpy as np
import pandas as pd

from src.ensemble import (
    best_per_panel_forecasts,
    best_per_panel_predictions,
    compute_inverse_metric_weights,
    select_best_model_per_panel,
    weighted_average_forecasts,
    weighted_average_predictions,
)

# ── compute_inverse_metric_weights ──


class TestComputeInverseMetricWeights:
    def test_empty(self) -> None:
        assert compute_inverse_metric_weights({}) == {}

    def test_single_model(self) -> None:
        result = compute_inverse_metric_weights({"a": 10.0})
        assert result == {"a": 1.0}

    def test_weights_sum_to_one(self) -> None:
        result = compute_inverse_metric_weights({"a": 5.0, "b": 10.0, "c": 20.0})
        assert math.isclose(sum(result.values()), 1.0, rel_tol=1e-9)

    def test_lower_metric_higher_weight(self) -> None:
        result = compute_inverse_metric_weights({"good": 2.0, "bad": 20.0})
        assert result["good"] > result["bad"]

    def test_inf_gets_zero_weight(self) -> None:
        result = compute_inverse_metric_weights({"a": 5.0, "b": float("inf")})
        assert result["b"] == 0.0
        assert result["a"] > 0.0

    def test_nan_gets_zero_weight(self) -> None:
        result = compute_inverse_metric_weights({"a": 5.0, "b": float("nan")})
        assert result["b"] == 0.0

    def test_all_inf_equal_weights(self) -> None:
        result = compute_inverse_metric_weights({"a": float("inf"), "b": float("inf")})
        assert math.isclose(result["a"], 0.5)
        assert math.isclose(result["b"], 0.5)

    def test_negative_metric_gets_zero(self) -> None:
        result = compute_inverse_metric_weights({"a": 5.0, "b": -1.0})
        assert result["b"] == 0.0

    def test_zero_metric(self) -> None:
        result = compute_inverse_metric_weights({"a": 0.0, "b": 10.0})
        assert result["a"] > result["b"]
        assert math.isclose(sum(result.values()), 1.0, rel_tol=1e-9)


# ── weighted_average_predictions ──


def _make_pred_df(panel_id: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel_id": [panel_id] * len(values),
            "date": pd.date_range("2024-01-01", periods=len(values), freq="MS"),
            "split": ["val"] * len(values),
            "y_pred": values,
        }
    )


class TestWeightedAveragePredictions:
    def test_two_models_equal_weights(self) -> None:
        preds = {
            "a": _make_pred_df("p1", [10.0, 20.0]),
            "b": _make_pred_df("p1", [30.0, 40.0]),
        }
        result = weighted_average_predictions(preds, {"a": 0.5, "b": 0.5})
        assert len(result) == 2
        np.testing.assert_allclose(result["y_pred"].values, [20.0, 30.0])

    def test_zero_weight_excluded(self) -> None:
        preds = {
            "a": _make_pred_df("p1", [10.0]),
            "b": _make_pred_df("p1", [90.0]),
        }
        result = weighted_average_predictions(preds, {"a": 1.0, "b": 0.0})
        assert len(result) == 1
        assert result["y_pred"].iloc[0] == 10.0

    def test_empty_predictions(self) -> None:
        result = weighted_average_predictions({}, {"a": 1.0})
        assert result.empty
        assert list(result.columns) == ["panel_id", "date", "split", "y_pred"]

    def test_multiple_panels(self) -> None:
        preds = {
            "a": pd.concat([_make_pred_df("p1", [10.0]), _make_pred_df("p2", [20.0])]),
            "b": pd.concat([_make_pred_df("p1", [30.0]), _make_pred_df("p2", [40.0])]),
        }
        result = weighted_average_predictions(preds, {"a": 0.5, "b": 0.5})
        assert len(result) == 2
        p1 = result[result["panel_id"] == "p1"]["y_pred"].iloc[0]
        p2 = result[result["panel_id"] == "p2"]["y_pred"].iloc[0]
        assert math.isclose(p1, 20.0)
        assert math.isclose(p2, 30.0)

    def test_unequal_weights(self) -> None:
        preds = {
            "a": _make_pred_df("p1", [100.0]),
            "b": _make_pred_df("p1", [0.0]),
        }
        # w_a=0.75, w_b=0.25 → 100*0.75 + 0*0.25 = 75
        result = weighted_average_predictions(preds, {"a": 0.75, "b": 0.25})
        assert math.isclose(result["y_pred"].iloc[0], 75.0)


# ── select_best_model_per_panel ──


class TestSelectBestModelPerPanel:
    def test_basic(self) -> None:
        metrics = {
            "a": [{"panel_id": "p1", "val": 5.0}, {"panel_id": "p2", "val": 15.0}],
            "b": [{"panel_id": "p1", "val": 10.0}, {"panel_id": "p2", "val": 3.0}],
        }
        result = select_best_model_per_panel(metrics)
        assert result["p1"] == "a"
        assert result["p2"] == "b"

    def test_nan_val_skipped(self) -> None:
        metrics = {
            "a": [{"panel_id": "p1", "val": float("nan")}],
            "b": [{"panel_id": "p1", "val": 5.0}],
        }
        result = select_best_model_per_panel(metrics)
        assert result["p1"] == "b"

    def test_none_val_skipped(self) -> None:
        metrics = {
            "a": [{"panel_id": "p1", "val": None}],
            "b": [{"panel_id": "p1", "val": 5.0}],
        }
        result = select_best_model_per_panel(metrics)
        assert result["p1"] == "b"

    def test_empty(self) -> None:
        assert select_best_model_per_panel({}) == {}


# ── best_per_panel_predictions ──


class TestBestPerPanelPredictions:
    def test_basic(self) -> None:
        preds = {
            "a": pd.concat([_make_pred_df("p1", [10.0]), _make_pred_df("p2", [20.0])]),
            "b": pd.concat([_make_pred_df("p1", [30.0]), _make_pred_df("p2", [40.0])]),
        }
        result = best_per_panel_predictions(preds, {"p1": "a", "p2": "b"})
        assert len(result) == 2
        p1_val = result[result["panel_id"] == "p1"]["y_pred"].iloc[0]
        p2_val = result[result["panel_id"] == "p2"]["y_pred"].iloc[0]
        assert p1_val == 10.0
        assert p2_val == 40.0

    def test_missing_model(self) -> None:
        preds = {"a": _make_pred_df("p1", [10.0])}
        result = best_per_panel_predictions(preds, {"p1": "missing"})
        assert result.empty

    def test_empty(self) -> None:
        result = best_per_panel_predictions({}, {})
        assert result.empty
        assert list(result.columns) == ["panel_id", "date", "split", "y_pred"]


# ── weighted_average_forecasts ──


def _make_fc_df(panel_id: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel_id": [panel_id] * len(values),
            "date": pd.date_range("2025-01-01", periods=len(values), freq="MS"),
            "forecast": values,
        }
    )


class TestWeightedAverageForecasts:
    def test_two_models(self) -> None:
        forecasts = {
            "a": _make_fc_df("p1", [10.0, 20.0]),
            "b": _make_fc_df("p1", [30.0, 40.0]),
        }
        result = weighted_average_forecasts(forecasts, {"a": 0.5, "b": 0.5})
        assert len(result) == 2
        np.testing.assert_allclose(result["forecast"].values, [20.0, 30.0])

    def test_empty(self) -> None:
        result = weighted_average_forecasts({}, {})
        assert result.empty
        assert list(result.columns) == ["panel_id", "date", "forecast"]


# ── best_per_panel_forecasts ──


class TestBestPerPanelForecasts:
    def test_basic(self) -> None:
        forecasts = {
            "a": pd.concat([_make_fc_df("p1", [10.0]), _make_fc_df("p2", [20.0])]),
            "b": pd.concat([_make_fc_df("p1", [30.0]), _make_fc_df("p2", [40.0])]),
        }
        result = best_per_panel_forecasts(forecasts, {"p1": "b", "p2": "a"})
        assert len(result) == 2
        p1_val = result[result["panel_id"] == "p1"]["forecast"].iloc[0]
        p2_val = result[result["panel_id"] == "p2"]["forecast"].iloc[0]
        assert p1_val == 30.0
        assert p2_val == 20.0

    def test_empty(self) -> None:
        result = best_per_panel_forecasts({}, {})
        assert result.empty
        assert list(result.columns) == ["panel_id", "date", "forecast"]
