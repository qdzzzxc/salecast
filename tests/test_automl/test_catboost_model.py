import numpy as np
import pandas as pd
import pytest

from src.automl.models.catboost_model import CatBoostForecastModel
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, ModelResult, Splits


@pytest.fixture()
def fast_params() -> CatBoostParameters:
    """Быстрые параметры CatBoost для тестов."""
    return CatBoostParameters(iterations=50, depth=3, verbose=False)


class TestCatBoostModel:
    def test_fit_evaluate_returns_model_result(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """fit_evaluate возвращает ModelResult с именем catboost."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result, ModelResult)
        assert result.name == "catboost"

    def test_evaluation_has_train_and_test_splits(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """EvaluationResults содержит train и test сплиты."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "train" in split_names
        assert "test" in split_names

    def test_predictions_are_finite(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """Предсказания не содержат NaN или Inf."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        for split_eval in result.evaluation.splits:
            assert np.all(np.isfinite(split_eval.y_pred))

    def test_params_stored_in_result(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """Параметры модели сохраняются в ModelResult."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result.params, CatBoostParameters)
        assert result.params.iterations == fast_params.iterations
