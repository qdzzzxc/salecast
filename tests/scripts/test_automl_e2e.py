"""End-to-end тест AutoML пайплайна на реальных данных.

Запуск:
    uv run python tests/scripts/test_automl_e2e.py
    uv run python tests/scripts/test_automl_e2e.py --models seasonal_naive catboost
    uv run python tests/scripts/test_automl_e2e.py --hyperopt --n-trials 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.automl.config import AutoMLConfig
from src.automl.selector import ModelSelector
from src.configs.settings import Settings
from src.custom_types import AutoMLResult, EvaluationResults, ModelResult
from src.model_selection import temporal_panel_split_by_size

DATA_PATH = ROOT / "data" / "processed" / "filtered_mirrors_ts.csv"

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("e2e")
logger.setLevel(logging.DEBUG)


def _sep(char: str = "─", width: int = 60) -> None:
    logger.info(char * width)


def _section(title: str) -> None:
    _sep()
    logger.info("  %s", title)
    _sep()


def _log_splits_info(splits: "Splits[pd.DataFrame]", id_col: str, date_col: str) -> None:  # noqa: F821
    for name, df in splits.splits:
        logger.info(
            "  %-6s  %d строк  %d артикулов  %s — %s",
            name,
            len(df),
            df[id_col].nunique(),
            df[date_col].min().strftime("%Y-%m"),
            df[date_col].max().strftime("%Y-%m"),
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
    mapes = [pm.metrics.mape for pm in split_eval.panel_metrics]
    mapes_series = pd.Series(mapes)
    logger.info(
        "  Панели (%s): медиана_mape=%.4f  p25=%.4f  p75=%.4f  p90=%.4f  max=%.4f",
        split_name,
        mapes_series.median(),
        mapes_series.quantile(0.25),
        mapes_series.quantile(0.75),
        mapes_series.quantile(0.90),
        mapes_series.max(),
    )
    n_bad = (mapes_series > 1.0).sum()
    logger.info("  Артикулов с mape > 1.0: %d / %d", n_bad, len(mapes))


def _log_comparison_table(result: AutoMLResult) -> None:
    rows = []
    for mr in result.all_results:
        sel_eval = next(
            (s for s in mr.evaluation.splits if s.split_name == result.selection_split),
            None,
        )
        test_eval = next((s for s in mr.evaluation.splits if s.split_name == "test"), None)
        rows.append({
            "model": mr.name,
            f"{result.selection_split}_mape": f"{sel_eval.overall_metrics.mape:.4f}" if sel_eval else "—",
            f"{result.selection_split}_r2": f"{sel_eval.overall_metrics.r2:.4f}" if sel_eval else "—",
            "test_mape": f"{test_eval.overall_metrics.mape:.4f}" if test_eval else "—",
            "test_r2": f"{test_eval.overall_metrics.r2:.4f}" if test_eval else "—",
            "best": "✓" if mr.name == result.best.name else "",
        })

    col_widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  " + "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    divider = "  " + "  ".join("─" * col_widths[k] for k in col_widths)
    logger.info(header)
    logger.info(divider)
    for row in rows:
        logger.info("  " + "  ".join(str(row[k]).ljust(col_widths[k]) for k in col_widths))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        logger.error("FAIL: %s", message)
        sys.exit(1)
    logger.info("  OK: %s", message)


def run(models: list[str], use_hyperopt: bool, n_trials: int) -> None:
    """Запускает полный E2E тест AutoML пайплайна."""
    t_start = time.perf_counter()

    _section("1. ЗАГРУЗКА ДАННЫХ")
    _assert(DATA_PATH.exists(), f"Файл данных существует: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("  Строк: %d  Артикулов: %d", len(df), df["article"].nunique())
    logger.info("  Период: %s — %s", df["date"].min().date(), df["date"].max().date())

    _section("2. СОЗДАНИЕ СПЛИТОВ")
    settings = Settings()
    splits = temporal_panel_split_by_size(
        df,
        panel_column=settings.columns.id,
        time_column=settings.columns.date,
        test_size=3,
        val_size=3,
    )
    _log_splits_info(splits, settings.columns.id, settings.columns.date)
    _assert(splits.val is not None, "val split создан")
    _assert(len(splits.test) > 0, "test split не пустой")

    _section("3. КОНФИГУРАЦИЯ AutoML")
    config = AutoMLConfig(
        models=models,
        selection_metric="mape",
        use_hyperopt=use_hyperopt,
        n_trials=n_trials,
    )
    logger.info("  models:           %s", config.models)
    logger.info("  selection_metric: %s", config.selection_metric)
    logger.info("  use_hyperopt:     %s", config.use_hyperopt)
    if use_hyperopt:
        logger.info("  n_trials:         %d", config.n_trials)

    _section("4. ЗАПУСК ModelSelector")
    t_sel = time.perf_counter()
    selector = ModelSelector(config)
    result = selector.run(splits, settings)
    elapsed = time.perf_counter() - t_sel
    logger.info("  Завершено за %.1f сек", elapsed)
    _assert(len(result.all_results) == len(models), "все модели обучены")
    _assert(result.best in result.all_results, "лучшая модель в all_results")

    _section("5. МЕТРИКИ ВСЕХ МОДЕЛЕЙ")
    for mr in result.all_results:
        _log_model_metrics(mr, result.selection_split)
        logger.info("")

    _section("6. СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    _log_comparison_table(result)

    _section("7. ЛУЧШАЯ МОДЕЛЬ")
    logger.info("  Победитель: %s  (по %s на %s)", result.best.name, result.selection_metric, result.selection_split)
    non_default_params = result.best.params.model_dump(exclude_defaults=True)
    logger.info("  Параметры: %s", non_default_params or "(defaults)")

    _log_panel_stats(result.best.evaluation, "val")
    _log_panel_stats(result.best.evaluation, "test")

    _section("8. ПРОВЕРКИ КОРРЕКТНОСТИ")
    for mr in result.all_results:
        for split_eval in mr.evaluation.splits:
            import numpy as np
            _assert(
                np.all(np.isfinite(split_eval.y_pred)),
                f"{mr.name}/{split_eval.split_name}: предсказания конечны",
            )
            _assert(
                split_eval.overall_metrics.mape < 1000,
                f"{mr.name}/{split_eval.split_name}: mape < 1000 (разумное значение)",
            )

    total = time.perf_counter() - t_start
    _sep("═")
    logger.info("  ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ  |  %.1f сек", total)
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
