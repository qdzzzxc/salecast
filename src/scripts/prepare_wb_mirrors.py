"""Подготовка данных зеркал WB для платформы Salecast.

Вход:  data/processed/mirrors_ts.csv
       Колонки: INN, NM_ID, Месяц, Выручка FBO, Выручка FBS, Выручка,
                Продажи FBO, Продажи FBS

Выход: data/wb/mirrors_ready.csv
       Колонки: nm_id, date, sales
       - nm_id  = NM_ID (строка) — идентификатор артикула (панели)
       - date   = Месяц + "-01" (YYYY-MM-DD) — первое число месяца
       - sales  = Продажи FBO + Продажи FBS — суммарные продажи

Трансформации:
  1. Агрегация по (NM_ID, Месяц): один артикул может продаваться через несколько
     продавцов (INN) → суммируем продажи по обоим каналам.
  2. Расширение до полной сетки дат: если для артикула отсутствует месяц,
     это означает 0 продаж (WB API не возвращает строки без активности).
     Заполняем нулями чтобы все ряды имели одинаковую длину (24 точки).

При загрузке в платформу указать:
  Колонка панели: nm_id
  Колонка даты:   date
  Колонка продаж: sales
"""

from pathlib import Path

import pandas as pd


INPUT = Path("data/processed/mirrors_ts.csv")
OUTPUT = Path("data/wb/mirrors_ready.csv")


def main() -> None:
    df = pd.read_csv(INPUT)

    # Шаг 1: суммируем продажи по (NM_ID, Месяц) — агрегируем по продавцам
    df["sales"] = df["Продажи FBO"] + df["Продажи FBS"]
    agg = (
        df.groupby(["NM_ID", "Месяц"], as_index=False)["sales"]
        .sum()
        .rename(columns={"NM_ID": "nm_id", "Месяц": "month"})
    )
    agg["nm_id"] = agg["nm_id"].astype(str)
    agg["date"] = pd.to_datetime(agg["month"] + "-01").dt.strftime("%Y-%m-%d")

    # Шаг 2: расширяем до полной сетки дат (пропуск = 0 продаж)
    all_dates = sorted(agg["date"].unique())
    all_ids = agg["nm_id"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_ids, all_dates], names=["nm_id", "date"]
    )
    agg_indexed = agg.set_index(["nm_id", "date"])["sales"]
    result = (
        agg_indexed.reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["nm_id", "date"])
        .reset_index(drop=True)
    )

    result["sales"] = result["sales"].astype(int)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    n_ids = result["nm_id"].nunique()
    n_dates = result["date"].nunique()
    zero_pct = (result["sales"] == 0).mean()

    print(f"Готово: {OUTPUT}")
    print(f"  Строк:    {len(result):,}  ({n_ids} артикулов × {n_dates} месяцев)")
    print(f"  Период:   {result['date'].min()} – {result['date'].max()}")
    print(f"  sales min/max: {result['sales'].min()} / {result['sales'].max()}")
    print(f"  Доля нулей: {zero_pct:.1%}")


if __name__ == "__main__":
    main()
