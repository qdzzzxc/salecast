from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.automl.base import ModelCancelledError
from src.automl.models.chronos_model import ChronosForecastModel, _align_chronos_predictions
from src.configs.settings import Settings
from src.custom_types import ModelResult, Splits


def _make_mock_pipeline(splits_or_df, id_col, date_col, target):
    """Создаёт мок Chronos2Pipeline с predict_df, возвращающим наивный прогноз."""
    pipeline = MagicMock()

    def mock_predict_df(context_df, prediction_length, quantile_levels, **kwargs):
        id_column = kwargs.get("id_column", "id")
        ts_column = kwargs.get("timestamp_column", "timestamp")

        rows = []
        for uid in context_df[id_column].unique():
            group = context_df[context_df[id_column] == uid].sort_values(ts_column)
            last_date = group[ts_column].max()
            last_val = group["target"].iloc[-1]
            future_dates = pd.date_range(last_date, periods=prediction_length + 1, freq="MS")[1:]
            for d in future_dates:
                rows.append({"id": uid, "timestamp": d, "0.5": float(last_val)})
        return pd.DataFrame(rows)

    pipeline.predict_df = mock_predict_df
    return pipeline


@pytest.fixture()
def mock_chronos(sample_splits, sample_settings):
    """Патчит Chronos2Pipeline.from_pretrained."""
    cols = sample_settings.columns
    pipeline = _make_mock_pipeline(
        sample_splits.train, cols.id, cols.date, cols.main_target,
    )
    with patch(
        "chronos.Chronos2Pipeline"
    ) as mock_cls:
        mock_cls.from_pretrained.return_value = pipeline
        yield pipeline


class TestChronosModel:
    def test_fit_evaluate_returns_model_result(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result, ModelResult)
        assert result.name == "chronos"

    def test_evaluation_has_val_and_test(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "val" in split_names
        assert "test" in split_names

    def test_predictions_are_finite(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        for split_eval in result.evaluation.splits:
            assert np.all(np.isfinite(split_eval.y_pred))

    def test_progress_fn_called(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        mock_chronos,
    ) -> None:
        progress = MagicMock()
        model = ChronosForecastModel()
        model.fit_evaluate(sample_splits, sample_settings, progress_fn=progress)
        assert progress.call_count > 0

    def test_cancel_fn_raises(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        with pytest.raises(ModelCancelledError):
            model.fit_evaluate(sample_splits, sample_settings, cancel_fn=lambda: True)

    def test_without_val(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        mock_chronos,
    ) -> None:
        splits_no_val = Splits(
            train=pd.concat([sample_splits.train, sample_splits.val], ignore_index=True),
            val=None,
            test=sample_splits.test,
        )
        model = ChronosForecastModel()
        result = model.fit_evaluate(splits_no_val, sample_settings)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "test" in split_names
        assert "val" not in split_names


class TestChronosForecastFuture:
    HORIZON = 2

    def test_returns_dataframe(
        self, full_df, sample_settings, mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"panel_id", "date", "forecast"}

    def test_correct_shape(
        self, full_df, sample_settings, mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        n_panels = full_df[sample_settings.columns.id].nunique()
        assert len(result) == n_panels * self.HORIZON

    def test_no_negative_values(
        self, full_df, sample_settings, mock_chronos,
    ) -> None:
        model = ChronosForecastModel()
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert (result["forecast"] >= 0).all()

    def test_on_training_done_called(
        self, full_df, sample_settings, mock_chronos,
    ) -> None:
        callback = MagicMock()
        model = ChronosForecastModel()
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_training_done=callback)
        callback.assert_called_once()


class TestAlignChronosPredictions:
    def test_aligns_by_id_and_date(self):
        pred_df = pd.DataFrame({
            "id": ["A1", "A1", "A2", "A2"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
            "0.5": [10.0, 20.0, 30.0, 40.0],
        })
        target_df = pd.DataFrame({
            "article": ["A2", "A2", "A1", "A1"],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
        })
        result = _align_chronos_predictions(pred_df, target_df, "article", "date")
        np.testing.assert_array_equal(result, [30.0, 40.0, 10.0, 20.0])

    def test_missing_predictions_filled_with_zero(self):
        pred_df = pd.DataFrame({
            "id": ["A1"],
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "0.5": [10.0],
        })
        target_df = pd.DataFrame({
            "article": ["A1", "A1"],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        result = _align_chronos_predictions(pred_df, target_df, "article", "date")
        assert result[0] == 10.0
        assert result[1] == 0.0
