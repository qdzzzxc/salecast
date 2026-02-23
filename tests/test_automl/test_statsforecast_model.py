from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.automl.models.statsforecast_model import StatsForecastModel
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits


class TestStatsForecastModel:
    @pytest.mark.parametrize("model_type", ["autoarima", "autoets", "autotheta"])
    def test_fit_evaluate_returns_model_result(
        self,
        model_type: str,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
    ) -> None:
        """fit_evaluate возвращает ModelResult для каждого типа модели."""
        model = StatsForecastModel(model_type=model_type)
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result, ModelResult)
        assert result.name == model_type

    @pytest.mark.parametrize("model_type", ["autoarima", "autoets", "autotheta"])
    def test_predictions_are_finite(
        self,
        model_type: str,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
    ) -> None:
        """Предсказания не содержат NaN или Inf."""
        model = StatsForecastModel(model_type=model_type)
        result = model.fit_evaluate(sample_splits, sample_settings)
        for split_eval in result.evaluation.splits:
            assert np.all(np.isfinite(split_eval.y_pred))

    def test_raises_if_statsforecast_not_installed(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
    ) -> None:
        """Бросает ImportError с понятным сообщением если statsforecast не установлен."""
        model = StatsForecastModel(model_type="autoarima")
        with patch.dict("sys.modules", {"statsforecast": None}):
            with pytest.raises(ImportError, match="statsforecast не установлен"):
                model.fit_evaluate(sample_splits, sample_settings)

    def test_evaluation_has_val_and_test(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
    ) -> None:
        """EvaluationResults содержит val и test сплиты."""
        model = StatsForecastModel(model_type="autoarima")
        result = model.fit_evaluate(sample_splits, sample_settings)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "val" in split_names
        assert "test" in split_names
