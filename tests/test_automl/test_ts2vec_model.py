"""Тесты TS2VecForecastModel."""

import numpy as np
import pandas as pd
import pytest

from src.automl.models.ts2vec_model import (
    TS2VecForecastModel,
    TS2VecParameters,
    _add_embedding_features,
    _reshape_panel_to_3d,
)
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
    return pd.concat([a1, a2], ignore_index=True)


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


class TestReshapePanelTo3d:
    def test_shape(self, panel_df) -> None:
        arr, ids = _reshape_panel_to_3d(panel_df, "article", "sales")
        assert arr.shape == (2, 24, 1)
        assert len(ids) == 2

    def test_values(self, panel_df) -> None:
        arr, ids = _reshape_panel_to_3d(panel_df, "article", "sales")
        idx = list(ids).index("A1")
        a1_vals = panel_df[panel_df["article"] == "A1"]["sales"].values
        np.testing.assert_array_almost_equal(arr[idx, :, 0], a1_vals)

    def test_empty(self) -> None:
        df = pd.DataFrame({"article": [], "sales": []})
        arr, ids = _reshape_panel_to_3d(df, "article", "sales")
        assert arr.shape[0] == 0

    def test_unequal_lengths(self) -> None:
        dates1 = pd.date_range("2023-01-01", periods=12, freq="MS")
        dates2 = pd.date_range("2023-01-01", periods=6, freq="MS")
        df = pd.concat(
            [
                pd.DataFrame({"article": "A1", "date": dates1, "sales": range(12)}),
                pd.DataFrame({"article": "A2", "date": dates2, "sales": range(6)}),
            ]
        )
        arr, ids = _reshape_panel_to_3d(df, "article", "sales")
        assert arr.shape == (2, 12, 1)
        idx_a2 = list(ids).index("A2")
        assert np.isnan(arr[idx_a2, 6:, 0]).all()


class TestAddEmbeddingFeatures:
    def test_columns_added(self, panel_df) -> None:
        embeddings = {
            "A1": np.array([1.0, 2.0, 3.0]),
            "A2": np.array([4.0, 5.0, 6.0]),
        }
        result = _add_embedding_features(panel_df, embeddings, "article", 3)
        assert "ts2vec_emb_0" in result.columns
        assert "ts2vec_emb_1" in result.columns
        assert "ts2vec_emb_2" in result.columns

    def test_values_correct(self, panel_df) -> None:
        embeddings = {
            "A1": np.array([1.0, 2.0]),
            "A2": np.array([3.0, 4.0]),
        }
        result = _add_embedding_features(panel_df, embeddings, "article", 2)
        a1_rows = result[result["article"] == "A1"]
        assert (a1_rows["ts2vec_emb_0"] == 1.0).all()
        assert (a1_rows["ts2vec_emb_1"] == 2.0).all()

    def test_missing_panel_gets_zeros(self) -> None:
        df = pd.DataFrame({"article": ["X", "X"], "sales": [1, 2]})
        embeddings = {"Y": np.array([1.0, 2.0])}
        result = _add_embedding_features(df, embeddings, "article", 2)
        assert (result["ts2vec_emb_0"] == 0.0).all()


class TestTS2VecParameters:
    def test_defaults(self) -> None:
        p = TS2VecParameters()
        assert p.output_dims == 320
        assert p.hidden_dims == 64
        assert p.depth == 10
        assert p.n_epochs == 50
        assert p.downstream == "catboost"

    def test_custom(self) -> None:
        p = TS2VecParameters(output_dims=64, n_epochs=10, batch_size=8)
        assert p.output_dims == 64
        assert p.n_epochs == 10
        assert p.batch_size == 8

    def test_serialization(self) -> None:
        p = TS2VecParameters(output_dims=128)
        d = p.model_dump()
        p2 = TS2VecParameters(**d)
        assert p2.output_dims == 128


class TestTS2VecForecastModel:
    def test_name(self) -> None:
        model = TS2VecForecastModel()
        assert model.name == "ts2vec"

    def test_custom_params(self) -> None:
        params = TS2VecParameters(output_dims=64, n_epochs=5)
        model = TS2VecForecastModel(params=params)
        assert model.params.output_dims == 64
        assert model.params.n_epochs == 5

    @pytest.mark.slow
    def test_fit_evaluate(self, splits) -> None:
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecForecastModel(params=params)
        settings = Settings()

        result = model.fit_evaluate(splits, settings)

        assert result.name == "ts2vec"
        assert result.evaluation is not None
        assert len(result.evaluation.splits) >= 1
        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0
        assert result.loss_history is not None
        assert len(result.loss_history) == 3  # n_epochs=3
        for epoch, loss in result.loss_history:
            assert isinstance(epoch, int)
            assert isinstance(loss, float)

    @pytest.mark.slow
    def test_forecast_future(self, panel_df) -> None:
        params = TS2VecParameters(output_dims=16, hidden_dims=8, depth=2, n_epochs=3, batch_size=4)
        model = TS2VecForecastModel(params=params)
        settings = Settings()

        forecast = model.forecast_future(panel_df, horizon=2, settings=settings)

        assert isinstance(forecast, pd.DataFrame)
        assert set(forecast.columns) == {"panel_id", "date", "forecast"}
        assert len(forecast) == 4
        assert (forecast["forecast"] >= 0).all()
