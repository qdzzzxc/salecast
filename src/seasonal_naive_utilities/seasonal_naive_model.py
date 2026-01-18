import numpy as np
import pandas as pd


class SeasonalNaiveModel:
    """Сезонный наивный прогноз: значение из того же сезона год назад."""

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self._history: dict[str | int, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, panel_column: str, target_column: str) -> "SeasonalNaiveModel":
        """Сохраняет все значения для каждой панели."""
        for panel_id, group in df.groupby(panel_column):
            self._history[panel_id] = group[target_column].values.copy()
        return self

    def predict(
        self, df: pd.DataFrame, panel_column: str, target_column: str, is_train: bool = False
    ) -> np.ndarray:
        """Возвращает значения из соответствующего сезона."""
        predictions = []

        for panel_id, group in df.groupby(panel_column, sort=False):
            if panel_id not in self._history:
                predictions.extend([0.0] * len(group))
                continue

            history = self._history[panel_id]
            values = group[target_column].values

            if is_train:
                panel_preds = []
                offset = len(history) - len(values) - self.seasonal_period
                for i in range(len(values)):
                    lag_idx = offset + i
                    if 0 <= lag_idx < len(history):
                        panel_preds.append(history[lag_idx])
                    else:
                        panel_preds.append(values[i])
                predictions.extend(panel_preds)
            else:
                panel_preds = []
                for i in range(len(values)):
                    lag_idx = len(history) - self.seasonal_period + i
                    if 0 <= lag_idx < len(history):
                        panel_preds.append(history[lag_idx])
                    else:
                        panel_preds.append(history[-1])
                predictions.extend(panel_preds)

        return np.array(predictions)
