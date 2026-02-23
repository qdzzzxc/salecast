import numpy as np
import pandas as pd
import pytest

from src.automl.models.seasonal_naive_model import SeasonalNaiveForecastModel
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits


class TestSeasonalNaiveModel:
    def test_fit_evaluate_returns_model_result(
        self, sample_splits: Splits[pd.DataFrame], sample_settings: Settings
    ) -> None:
        """fit_evaluate возвращает ModelResult."""
        model = SeasonalNaiveForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result, ModelResult)
        assert result.name == "seasonal_naive"

    def test_evaluation_has_required_splits(
        self, sample_splits: Splits[pd.DataFrame], sample_settings: Settings
    ) -> None:
        """EvaluationResults содержит train, val и test сплиты."""
        model = SeasonalNaiveForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "val" in split_names
        assert "test" in split_names

    def test_predictions_finite(
        self, sample_splits: Splits[pd.DataFrame], sample_settings: Settings
    ) -> None:
        """Предсказания не содержат NaN или Inf."""
        model = SeasonalNaiveForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        for split_eval in result.evaluation.splits:
            assert np.all(np.isfinite(split_eval.y_pred))

    def test_unknown_panel_predicts_zero(self, sample_settings: Settings) -> None:
        """Панель, не присутствующая в train, получает предсказание 0."""
        dates_train = pd.date_range("2021-01-01", periods=24, freq="MS")
        dates_test = pd.date_range("2023-01-01", periods=3, freq="MS")
        train_df = pd.DataFrame(
            {"article": "known", "date": dates_train, "sales": np.ones(24) * 10.0}
        )
        test_df = pd.DataFrame(
            {"article": "unknown", "date": dates_test, "sales": np.ones(3) * 5.0}
        )
        splits = Splits(train=train_df, val=None, test=test_df)

        model = SeasonalNaiveForecastModel()
        result = model.fit_evaluate(splits, sample_settings)

        test_eval = next(s for s in result.evaluation.splits if s.split_name == "test")
        assert np.all(test_eval.y_pred == 0.0)
