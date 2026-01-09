import re

import pandas as pd


def get_float_from_string(value: str) -> float | None:
    """Извлекает число из строки, сохраняя десятичную точку."""
    if pd.isna(value):
        return None
    match = re.search(r"\d+\.?\d*", value)
    return float(match.group()) if match else None
