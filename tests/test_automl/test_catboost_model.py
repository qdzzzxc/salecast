from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.automl.base import ModelCancelledError
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

    def test_progress_fn_called(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """progress_fn вызывается хотя бы один раз в ходе обучения."""
        progress = MagicMock()
        model = CatBoostForecastModel(params=fast_params)
        model.fit_evaluate(sample_splits, sample_settings, progress_fn=progress)
        assert progress.call_count > 0

    def test_cancel_fn_raises_model_cancelled_error(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """Если cancel_fn сразу возвращает True, бросается ModelCancelledError."""
        model = CatBoostForecastModel(params=fast_params)
        with pytest.raises(ModelCancelledError):
            model.fit_evaluate(sample_splits, sample_settings, cancel_fn=lambda: True)


class TestCatBoostForecastFuture:
    HORIZON = 2

    def test_returns_dataframe(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """forecast_future возвращает DataFrame с нужными колонками."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"panel_id", "date", "forecast"}

    def test_correct_shape(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """Количество строк = n_panels × horizon."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        n_panels = full_df[sample_settings.columns.id].nunique()
        assert len(result) == n_panels * self.HORIZON

    def test_no_negative_values(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """Прогноз не содержит отрицательных значений."""
        model = CatBoostForecastModel(params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert (result["forecast"] >= 0).all()

    def test_on_training_done_called_once(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """on_training_done вызывается ровно один раз после обучения."""
        callback = MagicMock()
        model = CatBoostForecastModel(params=fast_params)
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_training_done=callback)
        callback.assert_called_once()

    def test_on_forecast_step_called_per_horizon(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """on_forecast_step вызывается horizon раз — по одному на каждый шаг."""
        callback = MagicMock()
        model = CatBoostForecastModel(params=fast_params)
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_forecast_step=callback)
        assert callback.call_count == self.HORIZON

    def test_forecast_step_args(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
    ) -> None:
        """on_forecast_step получает (step_i, total) начиная с (1, horizon)."""
        calls = []
        model = CatBoostForecastModel(params=fast_params)
        model.forecast_future(
            full_df, self.HORIZON, sample_settings,
            on_forecast_step=lambda i, n: calls.append((i, n)),
        )
        assert calls[0] == (1, self.HORIZON)
        assert calls[-1] == (self.HORIZON, self.HORIZON)
