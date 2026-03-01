"""Конвертирует Kaggle Store Item Demand Forecasting датасет в формат salecast.

Исходный формат: date, store, item, sales
Результат:       article, date, sales  (суммируем продажи по всем магазинам для каждого товара)

Использование:
    python examples/convert_store_item.py
"""

import pandas as pd
from pathlib import Path

SRC = Path(__file__).parent / "Store Item Demand Forecasting (Kaggle) train.csv"
DST = Path(__file__).parent / "store_item_demand.csv"


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["date"])

    # Суммируем продажи по всем магазинам для каждого (товар, дата)
    df = df.groupby(["item", "date"], as_index=False)["sales"].sum()
    df = df.rename(columns={"item": "article"})
    df = df.sort_values(["article", "date"])

    df.to_csv(DST, index=False)
    print(f"Сохранено: {DST}")
    print(f"Товаров: {df['article'].nunique()}, строк: {len(df)}")
    print(f"Период: {df['date'].min().date()} — {df['date'].max().date()}")


if __name__ == "__main__":
    main()
