"""Celery task для temporal cross-validation лучшей модели."""

import io
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.automl.models import (
    CatBoostForecastModel,
    CatBoostPerPanelForecastModel,
    SeasonalNaiveForecastModel,
    StatsForecastModel,
)
from src.automl.models.catboost_clustered_model import CatBoostClusteredForecastModel
from src.automl.models.chronos_model import ChronosForecastModel, ChronosParameters
from src.automl.models.ts2vec_clustered_model import TS2VecClusteredForecastModel
from src.automl.models.ts2vec_model import TS2VecForecastModel, TS2VecParameters
from src.automl.ts_utils import get_downstream_lags, infer_ts_config, ts_config_from_freq
from src.configs.settings import ColumnConfig, Settings
from src.custom_types import CatBoostParameters, ModelType, Splits
from src.ensemble import (
    best_per_panel_predictions,
    compute_inverse_metric_weights,
    select_best_model_per_panel,
    weighted_average_predictions,
)
from src.evaluation import evaluate_multiple_splits
from src.model_selection import generate_expanding_cv_folds

logger = logging.getLogger(__name__)

_user = os.getenv("POSTGRES_USER", "sales_ts_prediction")
_password = os.getenv("POSTGRES_PASSWORD", "sales_ts_prediction")
_host = os.getenv("POSTGRES_HOST", "localhost")
_port = os.getenv("POSTGRES_PORT", "5432")
_db = os.getenv("POSTGRES_DB", "sales_ts_prediction")
_SYNC_DB_URL = f"postgresql+psycopg2://{_user}:{_password}@{_host}:{_port}/{_db}"

_minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
_minio_access = os.getenv("MINIO_ROOT_USER", "sales_ts_prediction")
_minio_secret = os.getenv("MINIO_ROOT_PASSWORD", "sales_ts_prediction")
_minio_bucket = os.getenv("MINIO_BUCKET", "salecast")

_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port = int(os.getenv("REDIS_PORT", "6379"))
_redis_password = os.getenv("REDIS_PASSWORD", "")


def _get_engine():
    return create_engine(_SYNC_DB_URL)


def _get_redis():
    import redis

    return redis.Redis(
        host=_redis_host, port=_redis_port, password=_redis_password, decode_responses=True
    )


def _get_s3():
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=_minio_endpoint,
        aws_access_key_id=_minio_access,
        aws_secret_access_key=_minio_secret,
        config=Config(signature_version="s3v4"),
    )


def _load_csv(key: str) -> pd.DataFrame:
    client = _get_s3()
    resp = client.get_object(Bucket=_minio_bucket, Key=key)
    return pd.read_csv(io.BytesIO(resp["Body"].read()))


def _upload_json(key: str, data: dict) -> None:
    client = _get_s3()
    body = json.dumps(data, ensure_ascii=False).encode()
    client.put_object(
        Bucket=_minio_bucket,
        Key=key,
        Body=body,
        ContentLength=len(body),
        ContentType="application/json",
    )


def _add_step(session: Session, job, name: str, message: str) -> None:
    from api.models import Job

    steps = list(job.steps)
    steps.append(
        {"name": name, "message": message, "timestamp": datetime.now(timezone.utc).isoformat()}
    )
    session.execute(
        Job.__table__.update()
        .where(Job.__table__.c.id == job.id)
        .values(steps=steps, status="running")
    )
    session.commit()
    job.steps = steps


def _build_model(
    model_type: str,
    catboost_params: dict | None = None,
    cluster_labels: dict[str, int] | None = None,
    chronos_params: dict | None = None,
    ts2vec_params: dict | None = None,
):
    mt = ModelType(model_type)
    if mt == ModelType.seasonal_naive:
        return SeasonalNaiveForecastModel()
    if mt == ModelType.catboost:
        return CatBoostForecastModel(params=CatBoostParameters(**(catboost_params or {})))
    if mt == ModelType.catboost_per_panel:
        return CatBoostPerPanelForecastModel(params=CatBoostParameters(**(catboost_params or {})))
    if mt == ModelType.catboost_clustered:
        return CatBoostClusteredForecastModel(
            cluster_labels=cluster_labels or {},
            params=CatBoostParameters(**(catboost_params or {})),
        )
    if mt in (ModelType.autoarima, ModelType.autoets, ModelType.autotheta, ModelType.mstl):
        return StatsForecastModel(model_type=mt)
    if mt == ModelType.chronos:
        return ChronosForecastModel(params=ChronosParameters(**(chronos_params or {})))
    if mt == ModelType.ts2vec:
        return TS2VecForecastModel(params=TS2VecParameters(**(ts2vec_params or {})))
    if mt == ModelType.ts2vec_clustered:
        return TS2VecClusteredForecastModel(
            cluster_labels=cluster_labels or {},
            params=TS2VecParameters(**(ts2vec_params or {})),
        )
    raise ValueError(f"Неизвестная модель: {model_type}")


def _build_fold_predictions(
    result, fold_splits: Splits, panel_col: str, date_col: str
) -> pd.DataFrame:
    """Строит DataFrame предсказаний модели на test split фолда."""
    rows = []
    for split_eval in result.evaluation.splits:
        if split_eval.split_name != "test":
            continue
        test_df = fold_splits.test
        for pm in split_eval.panel_metrics:
            pid = str(pm.panel_id)
            panel_df = test_df[test_df[panel_col] == pid].sort_values(date_col)
            dates = pd.to_datetime(panel_df[date_col]).dt.strftime("%Y-%m-%d").tolist()
            y_pred = list(pm.y_pred) if hasattr(pm.y_pred, "__iter__") else []
            if len(dates) == len(y_pred):
                for d, p in zip(dates, y_pred):
                    rows.append({"panel_id": pid, "date": d, "split": "test", "y_pred": float(p)})
    return pd.DataFrame(rows)


def _evaluate_ensemble_fold(
    fold_splits: Splits,
    ensemble_models: list[str],
    ensemble_method: str,
    settings,
    panel_col: str,
    date_col: str,
    value_col: str,
    catboost_params: dict | None,
    cluster_labels: dict[str, int] | None,
    chronos_params: dict | None,
    ts2vec_params: dict | None,
    selection_metric: str,
) -> dict:
    """Обучает все модели ансамбля на фолде и возвращает метрики комбинированных предсказаний."""
    predictions: dict[str, pd.DataFrame] = {}
    model_val_metrics: dict[str, float] = {}
    model_panel_metrics: dict[str, list[dict]] = {}

    for m_name in ensemble_models:
        model = _build_model(m_name, catboost_params, cluster_labels, chronos_params, ts2vec_params)
        result = model.fit_evaluate(fold_splits, settings)

        pred_df = _build_fold_predictions(result, fold_splits, panel_col, date_col)
        if not pred_df.empty:
            predictions[m_name] = pred_df

        # Извлекаем val метрику для весов
        for se in result.evaluation.splits:
            if se.split_name == "val" or (se.split_name == "test" and fold_splits.val is None):
                metric_val = getattr(se.overall_metrics, selection_metric, None)
                if metric_val is not None:
                    model_val_metrics[m_name] = float(metric_val)
                # Panel metrics для best_per_panel
                model_panel_metrics[m_name] = [
                    {
                        "panel_id": str(pm.panel_id),
                        "val": float(getattr(pm.metrics, selection_metric, float("inf"))),
                    }
                    for pm in se.panel_metrics
                ]

    if len(predictions) < 2:
        return {}

    # Комбинируем
    if ensemble_method == "weighted_avg":
        weights = compute_inverse_metric_weights(model_val_metrics)
        ensemble_preds = weighted_average_predictions(predictions, weights)
    else:
        panel_best = select_best_model_per_panel(model_panel_metrics)
        ensemble_preds = best_per_panel_predictions(predictions, panel_best)

    # Метрики на test
    test_df = fold_splits.test.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    ensemble_preds["date"] = pd.to_datetime(ensemble_preds["date"])

    merged = test_df.merge(
        ensemble_preds[["panel_id", "date", "y_pred"]],
        left_on=[panel_col, date_col],
        right_on=["panel_id", "date"],
        how="inner",
    )
    if merged.empty:
        return {}

    eval_result = evaluate_multiple_splits(
        {"test": (merged, merged["y_pred"].values)},
        panel_column=panel_col,
        target_column=value_col,
    )
    fold_metrics: dict[str, float] = {}
    panel_metrics_list: list[dict] = []
    for se in eval_result.splits:
        fold_metrics = {
            "mape": float(se.overall_metrics.mape),
            "rmse": float(se.overall_metrics.rmse),
            "mae": float(se.overall_metrics.mae),
            "r2": float(se.overall_metrics.r2),
        }
        panel_metrics_list = [
            {
                "panel_id": str(pm.panel_id),
                "mape": float(pm.metrics.mape),
                "rmse": float(pm.metrics.rmse),
            }
            for pm in se.panel_metrics
        ]

    return {"metrics": fold_metrics, "panel_metrics": panel_metrics_list}


@celery.task(bind=True, name="worker.tasks.cross_validation.run_cross_validation")
def run_cross_validation(
    self,
    job_id: str,
    project_id: str,
    automl_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    model_type: str,
    n_folds: int = 5,
    freq: str | None = None,
    catboost_params: dict | None = None,
    feature_params: dict | None = None,
    chronos_params: dict | None = None,
    ts2vec_params: dict | None = None,
    ensemble_models: list[str] | None = None,
    ensemble_method: str = "weighted_avg",
) -> dict:
    """Запускает temporal cross-validation для выбранной модели."""
    from api.models import Job

    engine = _get_engine()
    redis_client = _get_redis()
    stream_key = f"stream:cv:{job_id}"

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        automl_job = session.execute(select(Job).where(Job.id == automl_job_id)).scalar_one()

        try:
            _add_step(session, job, "loading", "Загрузка данных")

            split_info = automl_job.result["split"]
            train_df = _load_csv(split_info["train_key"])
            val_df = _load_csv(split_info["val_key"])
            test_df = _load_csv(split_info["test_key"])

            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            full_df[panel_col] = full_df[panel_col].astype(str)
            full_df[date_col] = pd.to_datetime(full_df[date_col])
            full_df = full_df.sort_values([panel_col, date_col]).reset_index(drop=True)

            # Фильтруем панели (green + yellow)
            diag_panels = automl_job.result.get("diagnostics", {}).get("panels", [])
            if diag_panels:
                good_panels = {
                    str(p["panel_id"])
                    for p in diag_panels
                    if p.get("overall_status") in ("green", "yellow")
                }
                full_df = full_df[full_df[panel_col].isin(good_panels)]

            # Settings
            ts_config = ts_config_from_freq(freq) if freq else infer_ts_config(full_df, date_col)
            lags = get_downstream_lags(ts_config.freq)
            base_settings = Settings()
            downstream_update: dict = {"lags": lags}
            if feature_params:
                downstream_update.update(feature_params)
            settings = base_settings.model_copy(
                update={
                    "columns": ColumnConfig(id=panel_col, date=date_col, main_target=value_col),
                    "ts": ts_config,
                    "downstream": base_settings.downstream.model_copy(update=downstream_update),
                }
            )

            # Cluster labels если нужно
            cluster_labels: dict[str, int] | None = None
            needs_clusters = model_type in ("catboost_clustered", "ts2vec_clustered")
            if model_type == "ensemble" and ensemble_models:
                needs_clusters = any(
                    m in ("catboost_clustered", "ts2vec_clustered") for m in ensemble_models
                )
            if needs_clusters:
                clustering_info = automl_job.result.get("clustering")
                if clustering_info:
                    labels_df = _load_csv(clustering_info["labels_key"])
                    cluster_labels = dict(
                        zip(labels_df.iloc[:, 0].astype(str), labels_df["cluster_id"].astype(int))
                    )

            # Генерируем фолды
            _add_step(session, job, "generating_folds", f"Генерация {n_folds} фолдов")
            folds = generate_expanding_cv_folds(
                full_df, n_folds=n_folds, panel_column=panel_col, time_column=date_col
            )

            fold_results: list[dict] = []
            all_panel_metrics: list[dict] = []

            is_ensemble = model_type == "ensemble" and ensemble_models
            selection_metric = automl_job.result.get("automl", {}).get("selection_metric", "mape")

            for fold_i, fold_splits in enumerate(folds):
                fold_num = fold_i + 1
                redis_client.xadd(
                    stream_key,
                    {"type": "fold_start", "fold": str(fold_num), "total": str(n_folds)},
                )

                if is_ensemble:
                    _add_step(
                        session,
                        job,
                        f"fold_{fold_num}",
                        f"Fold {fold_num}/{n_folds}: ансамбль ({len(ensemble_models)} моделей)",
                    )
                    ens_result = _evaluate_ensemble_fold(
                        fold_splits,
                        ensemble_models,
                        ensemble_method,
                        settings,
                        panel_col,
                        date_col,
                        value_col,
                        catboost_params,
                        cluster_labels,
                        chronos_params,
                        ts2vec_params,
                        selection_metric,
                    )
                    fold_metrics = ens_result.get("metrics", {})
                    all_panel_metrics.extend(
                        {**pm, "fold": fold_num} for pm in ens_result.get("panel_metrics", [])
                    )
                else:
                    _add_step(
                        session,
                        job,
                        f"fold_{fold_num}",
                        f"Fold {fold_num}/{n_folds}: обучение {model_type}",
                    )
                    model = _build_model(
                        model_type,
                        catboost_params,
                        cluster_labels,
                        chronos_params,
                        ts2vec_params,
                    )
                    result = model.fit_evaluate(fold_splits, settings)

                    fold_metrics = {}
                    for split_eval in result.evaluation.splits:
                        if split_eval.split_name == "test":
                            fold_metrics = {
                                "mape": float(split_eval.overall_metrics.mape),
                                "rmse": float(split_eval.overall_metrics.rmse),
                                "mae": float(split_eval.overall_metrics.mae),
                                "r2": float(split_eval.overall_metrics.r2),
                            }
                            all_panel_metrics.extend(
                                {
                                    "fold": fold_num,
                                    "panel_id": str(pm.panel_id),
                                    "mape": float(pm.metrics.mape),
                                    "rmse": float(pm.metrics.rmse),
                                }
                                for pm in split_eval.panel_metrics
                            )

                fold_result = {
                    "fold": fold_num,
                    "train_rows": len(fold_splits.train),
                    "test_rows": len(fold_splits.test),
                    **fold_metrics,
                }
                fold_results.append(fold_result)

                redis_client.xadd(
                    stream_key,
                    {
                        "type": "fold_done",
                        "fold": str(fold_num),
                        "mape": f"{fold_metrics.get('mape', 0):.4f}",
                        "rmse": f"{fold_metrics.get('rmse', 0):.4f}",
                    },
                )
                logger.info(
                    "CV fold %d/%d: MAPE=%.4f RMSE=%.4f",
                    fold_num,
                    n_folds,
                    fold_metrics.get("mape", 0),
                    fold_metrics.get("rmse", 0),
                )

            # Агрегация
            mapes = [f["mape"] for f in fold_results if "mape" in f]
            rmses = [f["rmse"] for f in fold_results if "rmse" in f]
            maes = [f["mae"] for f in fold_results if "mae" in f]

            summary = {
                "mean_mape": float(np.mean(mapes)) if mapes else None,
                "std_mape": float(np.std(mapes)) if mapes else None,
                "mean_rmse": float(np.mean(rmses)) if rmses else None,
                "std_rmse": float(np.std(rmses)) if rmses else None,
                "mean_mae": float(np.mean(maes)) if maes else None,
                "std_mae": float(np.std(maes)) if maes else None,
            }

            cv_result = {
                "model_type": model_type,
                "n_folds": n_folds,
                "folds": fold_results,
                "summary": summary,
                "panel_metrics": all_panel_metrics,
            }

            # Сохраняем в MinIO
            cv_key = f"projects/{project_id}/cv_result.json"
            _add_step(session, job, "saving", "Сохранение результатов")
            _upload_json(cv_key, cv_result)

            result_data = {
                **automl_job.result,
                "cross_validation": {
                    "model_type": model_type,
                    "n_folds": n_folds,
                    "summary": summary,
                    "cv_key": cv_key,
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            redis_client.xadd(stream_key, {"type": "completed"})
            logger.info("CV job %s done: %s", job_id, summary)

        except Exception:
            logger.exception("CV job %s failed", job_id)
            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="failed", completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            redis_client.xadd(stream_key, {"type": "failed"})
            raise
        else:
            return result_data
