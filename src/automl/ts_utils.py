import pandas as pd

from src.configs.settings import TimeSeriesConfig

# Маппинг базового кода частоты → длина сезонного периода
_FREQ_TO_SEASON: dict[str, int] = {
    "D": 7,
    "B": 5,
    "W": 52,
    "MS": 12,
    "Q": 4,
    "A": 1,
}

# Маппинг базового кода частоты → список лагов для CatBoost-фичей
_FREQ_TO_LAGS: dict[str, list[int]] = {
    "D": [1, 7, 14, 30, 365],
    "B": [1, 5, 10, 21, 252],
    "W": [1, 2, 4, 8, 52],
    "MS": [1, 2, 3, 6, 12],
    "Q": [1, 2, 3, 4],
    "A": [1],
}


def _normalize_freq(freq: str) -> str:
    """Нормализует строку частоты pandas к базовому коду из _FREQ_TO_SEASON."""
    if freq.startswith("W"):
        return "W"
    if freq in ("M", "MS", "ME", "BM", "BMS"):
        return "MS"
    if freq.startswith("Q") or freq.startswith("BQ"):
        return "Q"
    if freq.startswith(("A", "Y", "AS", "YS", "AE", "YE")):
        return "A"
    return freq


def infer_ts_config(df: pd.DataFrame, date_col: str) -> TimeSeriesConfig:
    """Определяет частоту и сезонный период временного ряда.

    Использует pd.infer_freq на уникальных датах первой панели.
    При неудаче возвращает дефолтный MS/12.
    """
    try:
        dates = pd.to_datetime(df[date_col]).sort_values().drop_duplicates()
        freq = pd.infer_freq(dates)
        if freq is None:
            freq = "MS"
    except Exception:
        freq = "MS"

    base = _normalize_freq(freq)
    season_length = _FREQ_TO_SEASON.get(base, 12)
    return TimeSeriesConfig(freq=freq, season_length=season_length)


def ts_config_from_freq(freq: str) -> TimeSeriesConfig:
    """Создаёт TimeSeriesConfig по явно заданной частоте (без инференса из данных)."""
    base = _normalize_freq(freq)
    season_length = _FREQ_TO_SEASON.get(base, 12)
    return TimeSeriesConfig(freq=freq, season_length=season_length)


def get_downstream_lags(freq: str) -> list[int]:
    """Возвращает список лагов для downstream-модели в зависимости от частоты."""
    base = _normalize_freq(freq)
    return _FREQ_TO_LAGS.get(base, [1, 2, 3, 6, 12])


def next_dates(dates: pd.Series, n: int) -> list[pd.Timestamp]:
    """Генерирует n следующих дат на основе частоты ряда."""
    sorted_dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    freq = pd.infer_freq(sorted_dates)
    if freq:
        return pd.date_range(sorted_dates.iloc[-1], periods=n + 1, freq=freq)[1:].tolist()
    delta = sorted_dates.diff().dropna().median()
    return [sorted_dates.iloc[-1] + delta * (i + 1) for i in range(n)]
