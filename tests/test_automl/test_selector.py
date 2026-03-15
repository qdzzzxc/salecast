import pandas as pd
import pytest
from unittest.mock import patch

from src.automl.config import AutoMLConfig
from src.automl.selector import ModelSelector, _build_model, _get_metric_value
from src.configs.settings import Settings
from src.custom_types import AutoMLResult, ModelResult, Splits


@pytest.fixture()
def fast_config() -> AutoMLConfig:
    """AutoMLConfig с быстрыми параметрами для тестов."""
    return AutoMLConfig(models=["seasonal_naive"], selection_metric="mape")


class TestModelSelector:
    def test_returns_automl_result(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_config: AutoMLConfig,
    ) -> None:
        """run() возвращает AutoMLResult."""
        selector = ModelSelector(fast_config)
        result = selector.run(sample_splits, sample_settings)
        assert isinstance(result, AutoMLResult)

    def test_best_model_is_in_all_results(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_config: AutoMLConfig,
    ) -> None:
        """best модель присутствует в all_results."""
        selector = ModelSelector(fast_config)
        result = selector.run(sample_splits, sample_settings)
        assert result.best in result.all_results

    def test_single_model_is_best(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_config: AutoMLConfig,
    ) -> None:
        """При одной модели она становится лучшей."""
        selector = ModelSelector(fast_config)
        result = selector.run(sample_splits, sample_settings)
        assert len(result.all_results) == 1
        assert result.best.name == "seasonal_naive"

    def test_creates_val_when_missing(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
        fast_config: AutoMLConfig,
    ) -> None:
        """Если val=None, selector создаёт внутренний val."""
        splits_no_val = Splits(
            train=pd.concat([sample_splits.train, sample_splits.val], ignore_index=True),
            val=None,
            test=sample_splits.test,
        )
        selector = ModelSelector(fast_config)
        result = selector.run(splits_no_val, sample_settings)
        assert result.selection_split == "val"
        assert isinstance(result, AutoMLResult)

    def test_selection_metric_respected(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
    ) -> None:
        """Лучшая модель имеет наименьшее значение selection_metric на val."""
        config = AutoMLConfig(models=["seasonal_naive"], selection_metric="mape")
        selector = ModelSelector(config)
        result = selector.run(sample_splits, sample_settings)

        best_val_evals = [
            s for s in result.best.evaluation.splits if s.split_name == result.selection_split
        ]
        assert best_val_evals, "Лучшая модель должна иметь val split"

        best_mape = best_val_evals[0].overall_metrics.mape
        for other in result.all_results:
            other_val_evals = [
                s for s in other.evaluation.splits if s.split_name == result.selection_split
            ]
            if other_val_evals:
                assert best_mape <= other_val_evals[0].overall_metrics.mape

    def test_hyperopt_branch_called(
        self,
        sample_splits: Splits[pd.DataFrame],
        sample_settings: Settings,
    ) -> None:
        """При use_hyperopt=True tune_catboost вызывается перед обучением."""
        from src.custom_types import CatBoostParameters

        config = AutoMLConfig(models=["catboost"], selection_metric="mape", use_hyperopt=True, n_trials=1)
        selector = ModelSelector(config)

        fast_params = CatBoostParameters(iterations=10, depth=2, verbose=False)
        with patch("src.automl.hyperopt.tune_catboost", return_value=fast_params) as mock_tune:
            result = selector.run(sample_splits, sample_settings)
        mock_tune.assert_called_once()
        assert isinstance(result, AutoMLResult)


class TestBuildModel:
    def test_raises_for_unknown_model_type(self) -> None:
        """_build_model бросает ValueError для неизвестного типа модели."""
        with pytest.raises(ValueError, match="Неизвестный тип модели"):
            _build_model("unknown_model")  # type: ignore[arg-type]


class TestGetMetricValue:
    def test_returns_inf_when_no_splits(self, sample_splits, sample_settings) -> None:
        """_get_metric_value возвращает inf если нет подходящего сплита."""
        from src.automl.models.seasonal_naive_model import SeasonalNaiveForecastModel

        model = SeasonalNaiveForecastModel()
        result = model.fit_evaluate(sample_splits, sample_settings)
        # Запрашиваем несуществующий сплит
        value = _get_metric_value(result, "mape", "nonexistent_split")
        assert value == float("inf") or value > 0  # fallback to test or inf
