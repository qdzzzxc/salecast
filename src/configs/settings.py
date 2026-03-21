from datetime import date

from pydantic import BaseModel, Field

from src.automl.config import AutoMLConfig
from src.diagnostics.config import DiagnosticsConfig
from src.model_selection import SplitRange


class ColumnConfig(BaseModel):
    """Конфигурация названий колонок."""

    id: str = Field(default="article", description="Название колонки с идентификатором")
    date: str = Field(default="date", description="Название колонки с датой")
    main_target: str = Field(default="sales", description="Название главной целевой колонки")


class FiltrationConfig(BaseModel):
    min_series_length: int = 18
    min_total_sales: int = 10
    max_zero_ratio: float = 0.2
    columns: ColumnConfig = Field(default_factory=ColumnConfig)


class PreprocessingConfig(BaseModel):
    """Конфигурация препроцессинга данных."""

    apply_log: bool = Field(
        default=True, description="Применять логарифмирование к целевым колонкам"
    )


class SplitConfig(BaseModel):
    """Конфигурация временных промежутков для сплитов."""

    train: SplitRange = Field(
        default_factory=lambda: SplitRange(
            date(2023, 1, 1), date(2024, 9, 30)
        ),  # date(2024, 7, 31)),
        description="Временной промежуток для train",
    )
    val: SplitRange | None = Field(
        default_factory=lambda: None,  # SplitRange(date(2024, 8, 1), date(2024, 9, 30)),
        description="Временной промежуток для val (None если без валидации)",
    )
    test: SplitRange = Field(
        default_factory=lambda: SplitRange(date(2024, 10, 1), date(2024, 12, 31)),
        description="Временной промежуток для test",
    )

    model_config = {"arbitrary_types_allowed": True}


class TimeSeriesConfig(BaseModel):
    """Конфигурация временного ряда."""

    freq: str = Field(default="MS", description="Частота ряда (MS, D, W, Q и т.д.)")
    season_length: int = Field(default=12, description="Длина сезонного периода")


class DownstreamConfig(BaseModel):
    """Конфигурация downstream-модели."""

    lags: list[int] = Field(default=[1, 2, 3], description="Список лагов для фичей")
    windows: list[int] = Field(default=[2, 3], description="Окна для rolling-фичей")
    ema_spans: list[int] = Field(default=[2, 3], description="Окна для EMA-фичей")
    round_predictions: bool = Field(default=True, description="Округлять предсказания до целых")
    inverse: bool = Field(
        default=True, description="Применять обратное преобразование к предсказаниям"
    )
    use_trend: bool = Field(default=False, description="Добавлять признак тренда (наклон регрессии)")
    trend_window: int = Field(default=6, description="Окно для вычисления тренда (точек)")
    use_cdf: bool = Field(default=False, description="Добавлять CDF-признак (позиция в распределении)")
    cdf_decay: float = Field(default=0.9, description="Коэффициент затухания для взвешенного CDF (0-1)")
    use_mstl_seasonal: bool = Field(
        default=False, description="Добавлять сезонную компоненту MSTL как признак",
    )


class Settings(BaseModel):
    """Главный конфиг пайплайна."""

    columns: ColumnConfig = Field(default_factory=ColumnConfig, description="Конфигурация колонок")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig, description="Конфигурация препроцессинга"
    )
    split: SplitConfig = Field(default_factory=SplitConfig, description="Конфигурация сплитов")
    downstream: DownstreamConfig = Field(
        default_factory=DownstreamConfig, description="Конфигурация downstream-модели"
    )
    filtration: FiltrationConfig = FiltrationConfig()

    ts: TimeSeriesConfig = Field(
        default_factory=TimeSeriesConfig, description="Параметры временного ряда"
    )
    random_state: int = Field(default=420, description="Seed для воспроизводимости")
    automl: AutoMLConfig = Field(
        default_factory=AutoMLConfig,
        description="Конфигурация AutoML",
    )
    diagnostics: DiagnosticsConfig = Field(
        default_factory=DiagnosticsConfig,
        description="Конфигурация диагностики качества данных",
    )
