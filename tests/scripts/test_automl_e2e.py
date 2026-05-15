"""End-to-end тест AutoML пайплайна на реальных данных.

Запуск:
    uv run python tests/scripts/test_automl_e2e.py
    uv run python tests/scripts/test_automl_e2e.py --models seasonal_naive catboost
    uv run python tests/scripts/test_automl_e2e.py --hyperopt --n-trials 10

Каждый запуск создаёт папку tests/scripts/runs/<timestamp>/ с run.log и PNG-графиками.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.automl.config import AutoMLConfig
from src.automl.selector import ModelSelector
from src.configs.settings import Settings
from src.custom_types import AutoMLResult, EvaluationResults, ModelResult, Splits
from src.data_processing import (
    aggregate_by_panel_date,
    drop_duplicates,
    expand_to_full_panel,
    sort_panel_by_date,
)
from src.filtration import filter_time_series
from src.model_selection import temporal_panel_split_by_size
from src.visualization.visualization import (
    plot_overall_metrics_comparison,
    plot_panel_predictions,
)

RAW_DATA_PATH = ROOT / "data" / "raw" / "filtered_ts.csv"
RUNS_DIR = Path(__file__).parent / "runs"


def _setup_logging(run_dir: Path) -> None:
    fmt = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt)

    e2e_logger = logging.getLogger("e2e")
    e2e_logger.setLevel(logging.DEBUG)
    e2e_logger.handlers.clear()
    e2e_logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt))
    e2e_logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))
    e2e_logger.addHandler(file_handler)


logger = logging.getLogger("e2e")


def _sep(char: str = "─", width: int = 60) -> None:
    logger.info(char * width)


def _section(title: str) -> None:
    _sep()
    logger.info("  %s", title)
    _sep()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        logger.error("FAIL: %s", message)
        sys.exit(1)
    logger.info("  OK: %s", message)


def _preprocess_raw(path: Path, settings: Settings) -> pd.DataFrame:
    """Загружает и приводит сырой CSV к формату (article, date, sales)."""
    cols = settings.columns

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Месяц"].astype(str) + "-01")
    df["sales"] = df["Продажи FBO"].fillna(0) + df["Продажи FBS"].fillna(0)
    df = df[["NM_ID", "date", "sales"]].rename(columns={"NM_ID": cols.id})

    df = drop_duplicates(df)
    df = sort_panel_by_date(df, panel_column=cols.id, date_column=cols.date)
    df = aggregate_by_panel_date(
        df, panel_column=cols.id, date_column=cols.date, target_columns=[cols.main_target]
    )
    df = expand_to_full_panel(df, panel_column=cols.id, date_column=cols.date)
    df[cols.main_target] = df[cols.main_target].fillna(0.0)

    return df


def _log_splits_info(splits: Splits[pd.DataFrame], id_col: str, date_col: str) -> None:
    for name, df in splits.splits:
        logger.info(
            "  %-6s  %d строк  %d артикулов  %s — %s",
            name,
            len(df),
            df[id_col].nunique(),
            pd.to_datetime(df[date_col]).min().strftime("%Y-%m"),
            pd.to_datetime(df[date_col]).max().strftime("%Y-%m"),
        )


def _log_model_metrics(result: ModelResult, selection_split: str) -> None:
    logger.info("  Модель: %s", result.name)
    for split_eval in result.evaluation.splits:
        m = split_eval.overall_metrics
        marker = " ◀ selection" if split_eval.split_name == selection_split else ""
        logger.info(
            "    %-6s  mape=%.4f  rmse=%.2f  mae=%.2f  r2=%.4f  nrmse=%.4f%s",
            split_eval.split_name,
            m.mape,
            m.rmse,
            m.mae,
            m.r2,
            m.nrmse,
            marker,
        )


def _log_panel_stats(eval_results: EvaluationResults, split_name: str) -> None:
    split_eval = next((s for s in eval_results.splits if s.split_name == split_name), None)
    if split_eval is None:
        return
    mapes = pd.Series([pm.metrics.mape for pm in split_eval.panel_metrics])
    finite_mapes = mapes[np.isfinite(mapes)]
    logger.info(
        "  Панели (%s)  n=%d  медиана=%.4f  p25=%.4f  p75=%.4f  p90=%.4f  max=%.4f  inf=%d",
        split_name,
        len(mapes),
        finite_mapes.median() if len(finite_mapes) else float("nan"),
        finite_mapes.quantile(0.25) if len(finite_mapes) else float("nan"),
        finite_mapes.quantile(0.75) if len(finite_mapes) else float("nan"),
        finite_mapes.quantile(0.90) if len(finite_mapes) else float("nan"),
        finite_mapes.max() if len(finite_mapes) else float("nan"),
        int((~np.isfinite(mapes)).sum()),
    )
    logger.info("  Артикулов с mape > 1.0: %d / %d", int((finite_mapes > 1.0).sum()), len(mapes))


def _log_comparison_table(result: AutoMLResult) -> None:
    rows = []
    for mr in result.all_results:
        sel = next(
            (s for s in mr.evaluation.splits if s.split_name == result.selection_split), None
        )
        tst = next((s for s in mr.evaluation.splits if s.split_name == "test"), None)
        rows.append(
            {
                "model": mr.name,
                f"{result.selection_split}_mape": f"{sel.overall_metrics.mape:.4f}" if sel else "—",
                f"{result.selection_split}_r2": f"{sel.overall_metrics.r2:.4f}" if sel else "—",
                "test_mape": f"{tst.overall_metrics.mape:.4f}" if tst else "—",
                "test_r2": f"{tst.overall_metrics.r2:.4f}" if tst else "—",
                "best": "✓" if mr.name == result.best.name else "",
            }
        )
    col_w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    logger.info("  " + "  ".join(k.ljust(col_w[k]) for k in col_w))
    logger.info("  " + "  ".join("─" * col_w[k] for k in col_w))
    for row in rows:
        logger.info("  " + "  ".join(str(row[k]).ljust(col_w[k]) for k in col_w))


def _save_visualizations(result: AutoMLResult, run_dir: Path, n: int = 3) -> None:
    best_eval = result.best.evaluation
    panel_metrics_df = best_eval.get_panel_metrics_df()
    test_metrics = panel_metrics_df[panel_metrics_df["split"] == "test"]

    plot_overall_metrics_comparison(best_eval)
    plt.savefig(run_dir / "overall_metrics.png", bbox_inches="tight", dpi=120)
    plt.close("all")
    logger.info("  Сохранён: overall_metrics.png")

    finite_test = test_metrics[np.isfinite(test_metrics["mape"])]
    best_ids = finite_test.nsmallest(n, "mape")["panel_id"].tolist()
    worst_ids = finite_test.nlargest(n, "mape")["panel_id"].tolist()

    for tag, panel_ids in [("best", best_ids), ("worst", worst_ids)]:
        for i, panel_id in enumerate(panel_ids):
            plot_panel_predictions(panel_id, best_eval, interactive=False)
            fname = f"{tag}_panel_{i + 1}_{panel_id}.png"
            plt.savefig(run_dir / fname, bbox_inches="tight", dpi=120)
            plt.close("all")
            mape_val = finite_test[finite_test["panel_id"] == panel_id]["mape"].values[0]
            logger.info("  Сохранён: %s  (test mape=%.4f)", fname, mape_val)


def run(models: list[str], use_hyperopt: bool, n_trials: int) -> None:
    """Запускает полный E2E тест AutoML пайплайна."""
    run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)

    logger.info("Run dir: %s", run_dir)
    t_start = time.perf_counter()
    settings = Settings()

    _section("1. ЗАГРУЗКА И PREPROCESSING ДАННЫХ")
    _assert(RAW_DATA_PATH.exists(), f"Сырой файл существует: {RAW_DATA_PATH}")
    df = _preprocess_raw(RAW_DATA_PATH, settings)
    logger.info(
        "  После preprocessing: %d строк  %d артикулов", len(df), df[settings.columns.id].nunique()
    )
    logger.info(
        "  Период: %s — %s",
        pd.to_datetime(df[settings.columns.date]).min().strftime("%Y-%m"),
        pd.to_datetime(df[settings.columns.date]).max().strftime("%Y-%m"),
    )

    _section("2. ФИЛЬТРАЦИЯ ВРЕМЕННЫХ РЯДОВ")
    n_before = df[settings.columns.id].nunique()
    filtration_result = filter_time_series(df, settings.filtration)
    df_filtered = filtration_result.df
    logger.info(
        "  Артикулов до: %d  после: %d  удалено: %d",
        n_before,
        df_filtered[settings.columns.id].nunique(),
        filtration_result.total_dropped,
    )
    for step, count in filtration_result.summary().items():
        if count > 0:
            logger.info("    %s: %d dropped", step, count)
    _assert(len(df_filtered) > 0, "После фильтрации остались данные")

    _section("3. СОЗДАНИЕ СПЛИТОВ")
    splits = temporal_panel_split_by_size(
        df_filtered,
        panel_column=settings.columns.id,
        time_column=settings.columns.date,
        test_size=3,
        val_size=3,
    )
    _log_splits_info(splits, settings.columns.id, settings.columns.date)
    _assert(splits.val is not None, "val split создан")

    _section("4. КОНФИГУРАЦИЯ AutoML")
    config = AutoMLConfig(
        models=models, selection_metric="mape", use_hyperopt=use_hyperopt, n_trials=n_trials
    )
    logger.info("  models:       %s", config.models)
    logger.info("  metric:       %s", config.selection_metric)
    logger.info(
        "  use_hyperopt: %s%s",
        config.use_hyperopt,
        f"  n_trials={config.n_trials}" if use_hyperopt else "",
    )

    _section("5. ЗАПУСК ModelSelector")
    t_sel = time.perf_counter()
    selector = ModelSelector(config)
    result = selector.run(splits, settings)
    logger.info("  Завершено за %.1f сек", time.perf_counter() - t_sel)
    _assert(len(result.all_results) == len(models), "все модели обучены")

    _section("6. МЕТРИКИ ВСЕХ МОДЕЛЕЙ")
    for mr in result.all_results:
        _log_model_metrics(mr, result.selection_split)
        logger.info("")

    _section("7. СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    _log_comparison_table(result)

    _section("8. ЛУЧШАЯ МОДЕЛЬ — ДЕТАЛИ")
    logger.info(
        "  Победитель: %s  (по %s на %s)",
        result.best.name,
        result.selection_metric,
        result.selection_split,
    )
    non_defaults = result.best.params.model_dump(exclude_defaults=True)
    logger.info("  Параметры: %s", non_defaults or "(defaults)")
    _log_panel_stats(result.best.evaluation, "val")
    _log_panel_stats(result.best.evaluation, "test")

    _section("9. ПРОВЕРКИ КОРРЕКТНОСТИ")
    for mr in result.all_results:
        for split_eval in mr.evaluation.splits:
            _assert(
                bool(np.all(np.isfinite(split_eval.y_pred))),
                f"{mr.name}/{split_eval.split_name}: предсказания конечны",
            )
            _assert(
                split_eval.overall_metrics.mape < 1000,
                f"{mr.name}/{split_eval.split_name}: mape < 1000",
            )

    _section("10. СОХРАНЕНИЕ ВИЗУАЛИЗАЦИЙ")
    _save_visualizations(result, run_dir, n=3)

    total = time.perf_counter() - t_start
    _sep("═")
    logger.info("  ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ  |  %.1f сек  |  %s", total, run_dir)
    _sep("═")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E тест AutoML пайплайна")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["seasonal_naive", "catboost"],
        choices=["seasonal_naive", "catboost", "autoarima", "autoets", "autotheta"],
        metavar="MODEL",
    )
    parser.add_argument("--hyperopt", action="store_true")
    parser.add_argument("--n-trials", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(models=args.models, use_hyperopt=args.hyperopt, n_trials=args.n_trials)
