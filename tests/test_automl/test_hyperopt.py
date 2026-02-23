from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.automl.hyperopt import tune_catboost
from src.configs.settings import Settings
from src.custom_types import CatBoostParameters, Splits


class TestTuneCatboost:
    def test_raises_if_no_val(
        self, sample_splits: Splits[pd.DataFrame], sample_settings: Settings
    ) -> None:
        """Бросает ValueError если val=None."""
        splits_no_val = Splits(
            train=pd.concat([sample_splits.train, sample_splits.val], ignore_index=True),
            val=None,
            test=sample_splits.test,
        )
        with pytest.raises(ValueError, match="val split"):
            tune_catboost(splits_no_val, sample_settings, n_trials=1)

    def test_n_trials_respected(
        self, sample_splits: Splits[pd.DataFrame], sample_settings: Settings
    ) -> None:
        """Optuna запускается с заданным количеством попыток."""
        mock_study = MagicMock()
        mock_study.best_params = {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
        }
        mock_study.best_value = 0.15

        with patch("optuna.create_study", return_value=mock_study):
            tune_catboost(sample_splits, sample_settings, n_trials=5)

        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args
        assert call_kwargs.kwargs.get("n_trials") == 5 or call_kwargs.args[1] == 5

    def test_returns_catboost_parameters(
        self, sample_splits: Splits[pd.DataFrame], sample_settings: Settings
    ) -> None:
        """Возвращает CatBoostParameters с лучшими найденными значениями."""
        mock_study = MagicMock()
        mock_study.best_params = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
        }
        mock_study.best_value = 0.15

        with patch("optuna.create_study", return_value=mock_study):
            result = tune_catboost(sample_splits, sample_settings, n_trials=1)

        assert isinstance(result, CatBoostParameters)
        assert result.iterations == 300
        assert result.depth == 4
