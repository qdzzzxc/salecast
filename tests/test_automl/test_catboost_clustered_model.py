from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.automl.base import ModelCancelledError
from src.automl.models.catboost_clustered_model import CatBoostClusteredForecastModel
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, ModelResult, Splits


@pytest.fixture
def fast_params() -> CatBoostParameters:
    """Быстрые параметры CatBoost для тестов."""
    return CatBoostParameters(iterations=50, depth=3, verbose=False)


@pytest.fixture
def cluster_labels(
    sample_splits: Splits[pd.DataFrame], sample_settings: Settings
) -> dict[str, int]:
    """Метки кластеров: 2 кластера из 5 панелей."""
    panels = sample_splits.train[sample_settings.columns.id].unique().tolist()
    # A0, A1, A2 → кластер 0; A3, A4 → кластер 1
    labels = {}
    for i, pid in enumerate(panels):
        labels[str(pid)] = 0 if i < 3 else 1
    return labels


class TestCatBoostClusteredModel:
    def test_fit_evaluate_returns_model_result(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result, ModelResult)
        assert result.name == "catboost_clustered"

    def test_evaluation_has_val_and_test(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "val" in split_names
        assert "test" in split_names

    def test_predictions_are_finite(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        for split_eval in result.evaluation.splits:
            assert np.all(np.isfinite(split_eval.y_pred))

    def test_params_stored_in_result(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert isinstance(result.params, CatBoostParameters)
        assert result.params.iterations == fast_params.iterations

    def test_feature_importance_present(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0
        # Отсортировано по убыванию важности
        importances = [v for _, v in result.feature_importance]
        assert importances == sorted(importances, reverse=True)

    def test_all_panels_covered(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        """Метрики содержат все панели из всех кластеров."""
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(sample_splits, sample_settings)
        panel_ids_in_result = {
            p.panel_id for split in result.evaluation.splits for p in split.panel_metrics
        }
        panel_ids_in_data = set(sample_splits.train[sample_settings.columns.id].astype(str))
        assert panel_ids_in_result == panel_ids_in_data

    def test_progress_fn_called(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        progress = MagicMock()
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        model.fit_evaluate(sample_splits, sample_settings, progress_fn=progress)
        assert progress.call_count > 0

    def test_cancel_fn_raises(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        with pytest.raises(ModelCancelledError):
            model.fit_evaluate(sample_splits, sample_settings, cancel_fn=lambda: True)

    def test_outliers_skipped(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        """Панели с cluster_id=-1 (выбросы HDBSCAN) пропускаются."""
        labels_with_outlier = {**cluster_labels}
        # Ставим одну панель как выброс
        first_panel = next(iter(labels_with_outlier))
        labels_with_outlier[first_panel] = -1

        model = CatBoostClusteredForecastModel(
            cluster_labels=labels_with_outlier, params=fast_params
        )
        result = model.fit_evaluate(sample_splits, sample_settings)
        panel_ids_in_result = {
            p.panel_id for split in result.evaluation.splits for p in split.panel_metrics
        }
        # Выброс не должен быть в результатах
        assert first_panel not in panel_ids_in_result

    def test_without_val(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        """Работает без val сплита."""
        splits_no_val = Splits(
            train=pd.concat([sample_splits.train, sample_splits.val], ignore_index=True),
            val=None,
            test=sample_splits.test,
        )
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.fit_evaluate(splits_no_val, sample_settings)
        assert isinstance(result, ModelResult)
        split_names = {s.split_name for s in result.evaluation.splits}
        assert "test" in split_names


class TestCatBoostClusteredForecastFuture:
    HORIZON = 2

    def test_returns_dataframe(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"panel_id", "date", "forecast"}

    def test_correct_shape(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        """Количество строк = n_panels_in_clusters × horizon."""
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        n_panels = len([pid for pid, cid in cluster_labels.items() if int(cid) >= 0])
        assert len(result) == n_panels * self.HORIZON

    def test_no_negative_values(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        assert (result["forecast"] >= 0).all()

    def test_dates_after_last_training_date(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        result = model.forecast_future(full_df, self.HORIZON, sample_settings)
        last_date = pd.to_datetime(full_df[sample_settings.columns.date]).max()
        assert (pd.to_datetime(result["date"]) > last_date).all()

    def test_on_training_done_called_once(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        callback = MagicMock()
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_training_done=callback)
        callback.assert_called_once()

    def test_on_forecast_step_called_per_cluster(
        self,
        full_df: pd.DataFrame,
        sample_settings: Settings,
        fast_params: CatBoostParameters,
        cluster_labels: dict[str, int],
    ) -> None:
        """on_forecast_step вызывается n_clusters раз."""
        callback = MagicMock()
        model = CatBoostClusteredForecastModel(cluster_labels=cluster_labels, params=fast_params)
        model.forecast_future(full_df, self.HORIZON, sample_settings, on_forecast_step=callback)
        n_clusters = len({cid for cid in cluster_labels.values() if int(cid) >= 0})
        assert callback.call_count == n_clusters
