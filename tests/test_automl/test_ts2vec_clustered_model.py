"""Тесты TS2VecClusteredForecastModel."""

import numpy as np
import pandas as pd
import pytest

from src.automl.models.ts2vec_clustered_model import TS2VecClusteredForecastModel
from src.automl.models.ts2vec_model import TS2VecParameters
from src.configs.settings import Settings
from src.custom_types import Splits


@pytest.fixture
def panel_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    a1 = pd.DataFrame(
        {
            "article": "A1",
            "date": dates,
            "sales": np.sin(np.arange(24) * 2 * np.pi / 12) * 50 + 100,
        }
    )
    a2 = pd.DataFrame(
        {
            "article": "A2",
            "date": dates,
            "sales": np.cos(np.arange(24) * 2 * np.pi / 12) * 30 + 200,
        }
    )
    a3 = pd.DataFrame(
        {
            "article": "A3",
            "date": dates,
            "sales": np.sin(np.arange(24) * 2 * np.pi / 6) * 40 + 150,
        }
    )
    return pd.concat([a1, a2, a3], ignore_index=True)


@pytest.fixture
def cluster_labels() -> dict[str, int]:
    return {"A1": 0, "A2": 0, "A3": 1}


@pytest.fixture
def splits(panel_df) -> Splits[pd.DataFrame]:
    parts = {"train": [], "val": [], "test": []}
    for _, grp in panel_df.groupby("article"):
        grp = grp.sort_values("date")
        parts["train"].append(grp.iloc[:18])
        parts["val"].append(grp.iloc[18:21])
        parts["test"].append(grp.iloc[21:24])
    return Splits(
        train=pd.concat(parts["train"], ignore_index=True),
        val=pd.concat(parts["val"], ignore_index=True),
        test=pd.concat(parts["test"], ignore_index=True),
    )


class TestTS2VecClusteredForecastModel:
    def test_name(self, cluster_labels) -> None:
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels)
        assert model.name == "ts2vec_clustered"

    def test_default_params(self, cluster_labels) -> None:
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels)
        assert model.params.output_dims == 320

    def test_custom_params(self, cluster_labels) -> None:
        params = TS2VecParameters(output_dims=32, n_epochs=3)
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels, params=params)
        assert model.params.output_dims == 32

    @pytest.mark.slow
    def test_fit_evaluate(self, splits, cluster_labels) -> None:
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels, params=params)
        settings = Settings()

        result = model.fit_evaluate(splits, settings)

        assert result.name == "ts2vec_clustered"
        assert result.evaluation is not None
        assert len(result.evaluation.splits) >= 1
        assert result.feature_importance is not None

        # Должны быть метрики для всех 3 панелей
        test_split = next(s for s in result.evaluation.splits if s.split_name == "test")
        panel_ids = {str(pm.panel_id) for pm in test_split.panel_metrics}
        assert panel_ids == {"A1", "A2", "A3"}

    @pytest.mark.slow
    def test_fit_evaluate_with_progress(self, splits, cluster_labels) -> None:
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels, params=params)
        settings = Settings()

        progress_calls = []
        model.fit_evaluate(
            splits, settings, progress_fn=lambda msg, pct: progress_calls.append((msg, pct))
        )
        assert len(progress_calls) > 0
        assert any("кластер" in msg for msg, _ in progress_calls)

    @pytest.mark.slow
    def test_forecast_future(self, panel_df, cluster_labels) -> None:
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels, params=params)
        settings = Settings()

        forecast = model.forecast_future(panel_df, horizon=2, settings=settings)

        assert isinstance(forecast, pd.DataFrame)
        assert set(forecast.columns) == {"panel_id", "date", "forecast"}
        assert len(forecast) == 6  # 3 panels * 2 horizon
        assert (forecast["forecast"] >= 0).all()
        assert set(forecast["panel_id"]) == {"A1", "A2", "A3"}

    def test_empty_cluster_skipped(self, splits) -> None:
        """Кластер без панелей в данных должен быть пропущен."""
        labels = {"A1": 0, "A2": 0, "A3": 1, "NONEXISTENT": 2}
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecClusteredForecastModel(cluster_labels=labels, params=params)
        settings = Settings()

        result = model.fit_evaluate(splits, settings)
        assert result.evaluation is not None

    def test_cancel(self, splits, cluster_labels) -> None:
        """Модель должна поддерживать отмену."""
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecClusteredForecastModel(cluster_labels=cluster_labels, params=params)
        settings = Settings()

        from src.automl.base import ModelCancelledError

        with pytest.raises(ModelCancelledError):
            model.fit_evaluate(splits, settings, cancel_fn=lambda: True)
