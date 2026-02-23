import pandas as pd
import pytest

from src.automl.config import AutoMLConfig
from src.automl.selector import ModelSelector
from src.configs.settings import Settings
from src.custom_types import AutoMLResult, Splits


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
