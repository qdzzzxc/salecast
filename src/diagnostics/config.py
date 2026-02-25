from pydantic import BaseModel, Field


class DiagnosticsConfig(BaseModel):
    """Конфигурация порогов для диагностики качества временных рядов."""

    min_length_red: int = Field(default=12, description="Длина ряда ниже которой — red")
    min_length_yellow: int = Field(default=24, description="Длина ряда ниже которой — yellow")

    max_zero_ratio_yellow: float = Field(default=0.2, description="Доля нулей выше которой — yellow")
    max_zero_ratio_red: float = Field(default=0.5, description="Доля нулей выше которой — red")

    max_cv_yellow: float = Field(default=0.5, description="CV выше которого — yellow")
    max_cv_red: float = Field(default=1.5, description="CV выше которого — red")

    min_acf_lag12: float = Field(
        default=0.3, description="Минимальный |ACF₁₂| для подтверждения годовой сезонности"
    )

    alpha: float = Field(default=0.05, description="Уровень значимости для статистических тестов")
