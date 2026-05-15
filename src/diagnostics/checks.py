import logging

import numpy as np

from src.custom_types import CheckResult
from src.diagnostics.config import DiagnosticsConfig

logger = logging.getLogger(__name__)


def check_length(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет длину временного ряда."""
    n = len(values)
    if n < config.min_length_red:
        return CheckResult(
            name="length",
            status="red",
            message=f"Ряд слишком короткий ({n} точек) — прогноз невозможен",
            value=float(n),
        )
    if n < config.min_length_yellow:
        return CheckResult(
            name="length",
            status="yellow",
            message=f"Мало данных ({n} точек) — прогноз может быть ненадёжен",
            value=float(n),
        )
    return CheckResult(
        name="length",
        status="green",
        message=f"Длина ряда достаточная ({n} точек)",
        value=float(n),
    )


def check_zero_ratio(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет долю нулевых значений в ряду."""
    ratio = float((values == 0).mean())
    if ratio > config.max_zero_ratio_red:
        return CheckResult(
            name="zero_ratio",
            status="red",
            message=f"Слишком много нулей ({ratio:.0%}) — данные нерепрезентативны",
            value=ratio,
        )
    if ratio > config.max_zero_ratio_yellow:
        return CheckResult(
            name="zero_ratio",
            status="yellow",
            message=f"Повышенная доля нулей ({ratio:.0%})",
            value=ratio,
        )
    return CheckResult(
        name="zero_ratio",
        status="green",
        message=f"Доля нулей приемлемая ({ratio:.0%})",
        value=ratio,
    )


def check_cv(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет коэффициент вариации временного ряда."""
    mean = float(np.mean(values))
    if mean == 0:
        return CheckResult(
            name="cv",
            status="red",
            message="Среднее значение равно нулю — CV не определён",
            value=None,
        )
    cv = float(np.std(values) / mean)
    if cv > config.max_cv_red:
        return CheckResult(
            name="cv",
            status="red",
            message=f"Очень высокая волатильность (CV={cv:.2f}) — случайность доминирует над структурой",
            value=cv,
        )
    if cv > config.max_cv_yellow:
        return CheckResult(
            name="cv",
            status="yellow",
            message=f"Повышенная волатильность (CV={cv:.2f})",
            value=cv,
        )
    return CheckResult(
        name="cv",
        status="green",
        message=f"Волатильность приемлемая (CV={cv:.2f})",
        value=cv,
    )


def check_autocorrelation(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет наличие автокорреляции через тест Льюнга-Бокса."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        n = len(values)
        lags = min(10, n // 5)
        if lags < 1:
            return CheckResult(
                name="autocorrelation",
                status="yellow",
                message="Ряд слишком короткий для теста автокорреляции",
                value=None,
            )
        result = acorr_ljungbox(values, lags=[lags], return_df=True)
        p_value = float(result["lb_pvalue"].iloc[0])
        if p_value < config.alpha:
            return CheckResult(
                name="autocorrelation",
                status="green",
                message=f"Автокорреляция обнаружена (p={p_value:.3f}) — прошлое объясняет будущее",
                value=p_value,
            )
        return CheckResult(
            name="autocorrelation",
            status="yellow",
            message=f"Автокорреляция не выявлена (p={p_value:.3f}) — прошлое слабо предсказывает будущее",
            value=p_value,
        )
    except Exception:
        logger.warning("Не удалось вычислить автокорреляцию", exc_info=True)
        return CheckResult(
            name="autocorrelation",
            status="yellow",
            message="Не удалось вычислить автокорреляцию",
            value=None,
        )


def check_stationarity(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет стационарность ряда через тест Дики-Фуллера (ADF)."""
    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(values, autolag="AIC")
        p_value = float(result[1])
        if p_value < config.alpha:
            return CheckResult(
                name="stationarity",
                status="green",
                message=f"Ряд стационарен (ADF p={p_value:.3f}) — структура предсказуема",
                value=p_value,
            )
        return CheckResult(
            name="stationarity",
            status="yellow",
            message=f"Возможное случайное блуждание (ADF p={p_value:.3f}) — структура ряда слабо предсказуема",
            value=p_value,
        )
    except Exception:
        logger.warning("Не удалось вычислить тест ADF", exc_info=True)
        return CheckResult(
            name="stationarity",
            status="yellow",
            message="Не удалось вычислить тест стационарности",
            value=None,
        )


def check_seasonality(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет наличие годовой сезонности через ACF на лаге 12."""
    try:
        from statsmodels.tsa.stattools import acf

        n = len(values)
        if n <= 13:
            return CheckResult(
                name="seasonality",
                status="yellow",
                message="Ряд слишком короткий для оценки годовой сезонности (нужно > 13 точек)",
                value=None,
            )
        acf_values = acf(values, nlags=12, fft=True)
        acf_12 = float(abs(acf_values[12]))
        if acf_12 >= config.min_acf_lag12:
            return CheckResult(
                name="seasonality",
                status="green",
                message=f"Годовая сезонность обнаружена (ACF₁₂={acf_12:.2f}) — seasonal naive применим",
                value=acf_12,
            )
        return CheckResult(
            name="seasonality",
            status="yellow",
            message=f"Слабая годовая сезонность (ACF₁₂={acf_12:.2f})",
            value=acf_12,
        )
    except Exception:
        logger.warning("Не удалось вычислить ACF", exc_info=True)
        return CheckResult(
            name="seasonality",
            status="yellow",
            message="Не удалось вычислить сезонность",
            value=None,
        )


def _mann_kendall_stat(values: np.ndarray) -> tuple[float, float]:
    """Вычисляет статистику и p-value теста Манна-Кендалла."""
    from scipy.stats import norm

    n = len(values)
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    return float(z), p_value


def check_trend(values: np.ndarray, config: DiagnosticsConfig) -> CheckResult:
    """Проверяет наличие монотонного тренда тестом Манна-Кендалла."""
    try:
        _, p_value = _mann_kendall_stat(values)
        if p_value < config.alpha:
            return CheckResult(
                name="trend",
                status="yellow",
                message=f"Обнаружен устойчивый тренд (Mann-Kendall p={p_value:.3f}) — учитывайте при интерпретации прогноза",
                value=p_value,
            )
        return CheckResult(
            name="trend",
            status="green",
            message=f"Монотонный тренд не обнаружен (Mann-Kendall p={p_value:.3f})",
            value=p_value,
        )
    except Exception:
        logger.warning("Не удалось вычислить тест Манна-Кендалла", exc_info=True)
        return CheckResult(
            name="trend",
            status="yellow",
            message="Не удалось вычислить тест на тренд",
            value=None,
        )
