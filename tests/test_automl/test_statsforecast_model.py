from unittest.mock import MagicMock, patch

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


class TestStatsForecastFuture:
    HORIZON = 3

    @pytest.mark.parametrize("model_type", ["autoarima", "autoets", "autotheta"])
    def test_returns_dataframe(
        self,
        model_type: str,
        full_df: pd.DataFrame,
        sample_settings: Settings,
    ) -> None:
        """forecast_future возвращает DataFrame с нужными колонками."""
        model = StatsForecastModel(model_type=model_type)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"panel_id", "date", "forecast"}

    def test_correct_shape(
        self, full_df: pd.DataFrame, sample_settings: Settings
    ) -> None:
        """Количество строк = n_panels × horizon."""
        model = StatsForecastModel(model_type="autoets")
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        n_panels = full_df[sample_settings.columns.id].nunique()
        assert len(result) == n_panels * self.HORIZON

    def test_no_negative_values(
        self, full_df: pd.DataFrame, sample_settings: Settings
    ) -> None:
        """Прогноз не содержит отрицательных значений."""
        model = StatsForecastModel(model_type="autoets")
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert (result["forecast"] >= 0).all()

    def test_on_training_done_called_once(
        self, full_df: pd.DataFrame, sample_settings: Settings
    ) -> None:
        """on_training_done вызывается ровно один раз после обучения."""
        callback = MagicMock()
        model = StatsForecastModel(model_type="autoets")
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_training_done=callback)
        callback.assert_called_once()

    def test_on_forecast_step_not_called(
        self, full_df: pd.DataFrame, sample_settings: Settings
    ) -> None:
        """StatsForecast предсказывает одним вызовом — on_forecast_step не вызывается."""
        callback = MagicMock()
        model = StatsForecastModel(model_type="autoets")
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_forecast_step=callback)
        callback.assert_not_called()

    def test_dates_after_last_training_date(
        self, full_df: pd.DataFrame, sample_settings: Settings
    ) -> None:
        """Все даты прогноза строго после последней даты в full_df."""
        model = StatsForecastModel(model_type="autoets")
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        last_date = pd.to_datetime(full_df[sample_settings.columns.date]).max()
        forecast_dates = pd.to_datetime(result["date"])
        assert (forecast_dates > last_date).all()
