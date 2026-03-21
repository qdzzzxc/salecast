import numpy as np
import pandas as pd
import pytest

from src.configs.settings import Settings
from src.custom_types import EvaluationResults, SplitEvaluation, Splits
from src.evaluation import (
    combine_panel_results,
    compute_regression_metrics,
    evaluate_from_predictions,
    evaluate_split,
    get_panel_metrics_wide,
)


def _make_pred_df(
    panel_ids: list[str],
    n_periods: int,
    start: str = "2021-01-01",
    sales_val: float = 5.0,
    pred_val: float = 4.5,
) -> pd.DataFrame:
    """Создаёт датафрейм с предсказаниями для указанных панелей."""
    dates = pd.date_range(start, periods=n_periods, freq="MS")
    rows = [
        {"article": pid, "date": d, "sales": sales_val, "prediction": pred_val}
        for pid in panel_ids
        for d in dates
    ]
    return pd.DataFrame(rows)


def _build_evaluation_result(panel_ids: list[str], n_periods: int = 12) -> EvaluationResults:
    """Строит EvaluationResults для указанного набора панелей."""
    df = _make_pred_df(panel_ids, n_periods)
    settings = Settings()
    splits = Splits(
        train=df.copy(),
        val=None,
        test=df.copy(),
    )
    return evaluate_from_predictions(df, splits, settings)


class TestComputeRegressionMetrics:
    def test_normal_case_returns_finite_metrics(self) -> None:
        """Для нормальных данных все метрики конечны."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.mae)
        assert np.isfinite(metrics.mape)
        assert np.isfinite(metrics.r2)

    def test_perfect_prediction_r2_is_one(self) -> None:
        """При точном предсказании R2 равен 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_regression_metrics(y, y.copy())
        assert metrics.r2 == pytest.approx(1.0)

    def test_all_zero_y_true_gives_inf_mape(self) -> None:
        """Когда все y_true равны нулю, MAPE равен inf."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics.mape == float("inf")

    def test_all_zero_y_true_gives_inf_nrmse(self) -> None:
        """Когда все y_true равны нулю, nrmse равен inf."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics.nrmse == float("inf")


class TestEvaluateSplit:
    def test_returns_split_evaluation_with_correct_split_name(self) -> None:
        """evaluate_split возвращает SplitEvaluation с правильным именем сплита."""
        df = _make_pred_df(["A", "B"], n_periods=6)
        predictions = np.full(len(df), 4.5)
        result = evaluate_split(df, predictions, "article", "sales", "test")
        assert isinstance(result, SplitEvaluation)
        assert result.split_name == "test"

    def test_panel_count_matches_unique_panels(self) -> None:
        """Количество панелей в результате совпадает с числом уникальных панелей."""
        df = _make_pred_df(["A", "B", "C"], n_periods=6)
        predictions = np.full(len(df), 4.5)
        result = evaluate_split(df, predictions, "article", "sales", "train")
        assert len(result.panel_metrics) == 3

    def test_overall_metrics_are_finite(self) -> None:
        """Общие метрики конечны при нормальных данных."""
        df = _make_pred_df(["X", "Y"], n_periods=8)
        predictions = np.full(len(df), 5.0)
        result = evaluate_split(df, predictions, "article", "sales", "val")
        assert np.isfinite(result.overall_metrics.rmse)


class TestGetPanelMetricsWide:
    def test_returns_dict_with_correct_keys(self) -> None:
        """Возвращаемый словарь содержит ключи для каждого сплита."""
        df = _make_pred_df(["A", "B"], n_periods=6)
        predictions = np.full(len(df), 4.5)
        split_eval = evaluate_split(df, predictions, "article", "sales", "test")
        results = EvaluationResults(splits=[split_eval])
        panel_df = results.get_panel_metrics_df()
        wide = get_panel_metrics_wide(panel_df, sort_metric="mape")
        assert "test" in wide

    def test_sorted_by_metric_descending(self) -> None:
        """Датафрейм для каждого сплита отсортирован по метрике по убыванию."""
        df = pd.DataFrame(
            [
                {"article": "A", "date": pd.Timestamp("2021-01-01"), "sales": 1.0},
                {"article": "A", "date": pd.Timestamp("2021-02-01"), "sales": 2.0},
                {"article": "B", "date": pd.Timestamp("2021-01-01"), "sales": 10.0},
                {"article": "B", "date": pd.Timestamp("2021-02-01"), "sales": 20.0},
            ]
        )
        preds_a = np.array([5.0, 5.0])
        preds_b = np.array([10.5, 20.5])
        predictions = np.concatenate([preds_a, preds_b])
        split_eval = evaluate_split(df, predictions, "article", "sales", "test")
        results = EvaluationResults(splits=[split_eval])
        panel_df = results.get_panel_metrics_df()
        wide = get_panel_metrics_wide(panel_df, sort_metric="mape")
        mape_values = wide["test"]["mape"].values
        assert list(mape_values) == sorted(mape_values, reverse=True)


class TestCombinePanelResults:
    def test_combined_has_more_panels_than_individual(self) -> None:
        """Объединённый результат содержит больше панелей, чем каждый из входных."""
        result_a = _build_evaluation_result(["P0", "P1"])
        result_b = _build_evaluation_result(["P2", "P3"])
        combined = combine_panel_results([result_a, result_b])
        combined_panels = set()
        for split_eval in combined.splits:
            for pm in split_eval.panel_metrics:
                combined_panels.add(pm.panel_id)

        panels_a = set()
        for split_eval in result_a.splits:
            for pm in split_eval.panel_metrics:
                panels_a.add(pm.panel_id)

        assert len(combined_panels) > len(panels_a)

    def test_combined_contains_all_panels(self) -> None:
        """Объединённый результат содержит панели из обоих входных результатов."""
        result_a = _build_evaluation_result(["A1", "A2"])
        result_b = _build_evaluation_result(["B1", "B2"])
        combined = combine_panel_results([result_a, result_b])
        combined_panels = set()
        for split_eval in combined.splits:
            for pm in split_eval.panel_metrics:
                combined_panels.add(pm.panel_id)
        assert {"A1", "A2", "B1", "B2"}.issubset(combined_panels)


class TestEvaluateFromPredictions:
    def test_returns_evaluation_results(self) -> None:
        """evaluate_from_predictions возвращает объект EvaluationResults."""
        settings = Settings()
        df = _make_pred_df(["P0", "P1"], n_periods=10)
        splits = Splits(train=df.copy(), val=None, test=df.copy())
        result = evaluate_from_predictions(df, splits, settings)
        assert isinstance(result, EvaluationResults)

    def test_split_names_present(self) -> None:
        """Результат содержит сплиты train и test."""
        settings = Settings()
        df = _make_pred_df(["P0", "P1"], n_periods=10)
        splits = Splits(train=df.copy(), val=None, test=df.copy())
        result = evaluate_from_predictions(df, splits, settings)
        split_names = {s.split_name for s in result.splits}
        assert "train" in split_names
        assert "test" in split_names

    def test_panel_metrics_populated(self) -> None:
        """Метрики по панелям заполнены для каждого сплита."""
        settings = Settings()
        df = _make_pred_df(["P0", "P1", "P2"], n_periods=8)
        splits = Splits(train=df.copy(), val=None, test=df.copy())
        result = evaluate_from_predictions(df, splits, settings)
        for split_eval in result.splits:
            assert len(split_eval.panel_metrics) > 0

    def test_uses_prediction_column(self) -> None:
        """Функция корректно читает кастомную колонку предсказаний."""
        settings = Settings()
        df = _make_pred_df(["P0"], n_periods=6)
        df = df.rename(columns={"prediction": "my_pred"})
        splits = Splits(train=df.copy(), val=None, test=df.copy())
        result = evaluate_from_predictions(df, splits, settings, prediction_column="my_pred")
        assert isinstance(result, EvaluationResults)
