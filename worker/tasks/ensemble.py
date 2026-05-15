"""Celery task для ансамблирования предсказаний нескольких моделей."""

import io
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.ensemble import (
    best_per_panel_predictions,
    compute_inverse_metric_weights,
    select_best_model_per_panel,
    weighted_average_predictions,
)
from src.evaluation import evaluate_multiple_splits

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


def _upload_csv(key: str, df: pd.DataFrame) -> None:
    client = _get_s3()
    body = df.to_csv(index=False).encode()
    client.put_object(
        Bucket=_minio_bucket,
        Key=key,
        Body=body,
        ContentLength=len(body),
        ContentType="text/csv",
    )


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


@celery.task(bind=True, name="worker.tasks.ensemble.run_ensemble")
def run_ensemble(
    self,
    job_id: str,
    project_id: str,
    automl_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    models: list[str],
    method: str = "weighted_avg",
    selection_metric: str = "mape",
) -> dict:
    """Рассчитывает ансамбль предсказаний нескольких моделей.

    method: "weighted_avg" | "best_per_panel"
    """
    from api.models import Job

    engine = _get_engine()
    redis_client = _get_redis()
    stream_key = f"stream:ensemble:{job_id}"

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        automl_job = session.execute(select(Job).where(Job.id == automl_job_id)).scalar_one()

        try:
            # ── 1. Загрузка предсказаний ──
            _add_step(session, job, "loading", "Загрузка предсказаний моделей")
            redis_client.xadd(stream_key, {"type": "loading"})

            automl_result = automl_job.result.get("automl", {})
            model_results = automl_result.get("model_results", [])
            model_results_by_name = {mr["name"]: mr for mr in model_results}

            predictions: dict[str, pd.DataFrame] = {}
            model_val_metrics: dict[str, float] = {}
            model_test_metrics: dict[str, float] = {}
            model_panel_metrics: dict[str, list[dict]] = {}

            for model_name in models:
                mr = model_results_by_name.get(model_name)
                if mr is None:
                    logger.warning("Модель %s не найдена в automl результатах", model_name)
                    continue

                pred_key = mr.get("predictions_key")
                if not pred_key:
                    logger.warning("Нет предсказаний для модели %s", model_name)
                    continue

                try:
                    pred_df = _load_csv(pred_key)
                    predictions[model_name] = pred_df
                except Exception:
                    logger.exception("Не удалось загрузить предсказания модели %s", model_name)
                    continue

                val_metric = mr.get(f"val_{selection_metric}")
                test_metric = mr.get(f"test_{selection_metric}")
                if val_metric is not None:
                    model_val_metrics[model_name] = float(val_metric)
                if test_metric is not None:
                    model_test_metrics[model_name] = float(test_metric)

                panel_m = mr.get("panel_metrics", [])
                model_panel_metrics[model_name] = panel_m

            if len(predictions) < 2:
                raise ValueError(
                    f"Нужно минимум 2 модели с предсказаниями, найдено: {len(predictions)}"
                )

            # ── 2. Комбинирование ──
            _add_step(session, job, "computing", "Расчёт ансамбля")
            redis_client.xadd(stream_key, {"type": "computing", "method": method})

            if method == "weighted_avg":
                weights = compute_inverse_metric_weights(model_val_metrics)
                ensemble_preds = weighted_average_predictions(predictions, weights)
                method_info = {"weights": weights}
            elif method == "best_per_panel":
                panel_best = select_best_model_per_panel(model_panel_metrics)
                ensemble_preds = best_per_panel_predictions(predictions, panel_best)
                # Считаем сколько панелей "выиграла" каждая модель
                model_wins: dict[str, int] = {}
                for m in panel_best.values():
                    model_wins[m] = model_wins.get(m, 0) + 1
                method_info = {"panel_best_model": panel_best, "model_wins": model_wins}
            else:
                raise ValueError(f"Неизвестный метод ансамбля: {method}")

            # ── 3. Метрики ансамбля ──
            _add_step(session, job, "evaluating", "Вычисление метрик")

            # Загружаем true values из splits
            split_info = automl_job.result.get("split", {})
            val_df = _load_csv(split_info["val_key"])
            test_df = _load_csv(split_info["test_key"])
            val_df[panel_col] = val_df[panel_col].astype(str)
            val_df[date_col] = pd.to_datetime(val_df[date_col])
            test_df[panel_col] = test_df[panel_col].astype(str)
            test_df[date_col] = pd.to_datetime(test_df[date_col])

            ensemble_preds["panel_id"] = ensemble_preds["panel_id"].astype(str)
            ensemble_preds["date"] = pd.to_datetime(ensemble_preds["date"])

            ensemble_metrics = {}
            for split_name, true_df in [("val", val_df), ("test", test_df)]:
                split_preds = ensemble_preds[ensemble_preds["split"] == split_name]
                if split_preds.empty:
                    continue

                merged = true_df.merge(
                    split_preds[["panel_id", "date", "y_pred"]],
                    left_on=[panel_col, date_col],
                    right_on=["panel_id", "date"],
                    how="inner",
                )
                if merged.empty:
                    continue

                eval_result = evaluate_multiple_splits(
                    {split_name: (merged, merged["y_pred"].values)},
                    panel_column=panel_col,
                    target_column=value_col,
                )
                for se in eval_result.splits:
                    ensemble_metrics[split_name] = {
                        "mape": float(se.overall_metrics.mape),
                        "rmse": float(se.overall_metrics.rmse),
                        "mae": float(se.overall_metrics.mae),
                        "r2": float(se.overall_metrics.r2),
                    }

            # ── 4. Таблица сравнения ──
            comparison = [
                {
                    "name": model_name,
                    f"val_{selection_metric}": model_val_metrics.get(model_name),
                    f"test_{selection_metric}": model_test_metrics.get(model_name),
                }
                for model_name in predictions
            ]
            comparison.append(
                {
                    "name": "ensemble",
                    f"val_{selection_metric}": ensemble_metrics.get("val", {}).get(
                        selection_metric
                    ),
                    f"test_{selection_metric}": ensemble_metrics.get("test", {}).get(
                        selection_metric
                    ),
                }
            )

            # ── 5. Сохранение ──
            _add_step(session, job, "saving", "Сохранение результатов")

            # Ensemble predictions CSV
            ens_pred_key = f"projects/{project_id}/ensemble_predictions.csv"
            _upload_csv(ens_pred_key, ensemble_preds)

            # Ensemble result JSON
            ensemble_result = {
                "method": method,
                "models": list(predictions.keys()),
                "method_info": method_info,
                "metrics": ensemble_metrics,
                "comparison": comparison,
                "predictions_key": ens_pred_key,
            }
            ens_json_key = f"projects/{project_id}/ensemble_result.json"
            _upload_json(ens_json_key, ensemble_result)

            result_data = {
                **automl_job.result,
                "ensemble": {
                    "method": method,
                    "models": list(predictions.keys()),
                    "metrics": ensemble_metrics,
                    "comparison": comparison,
                    "method_info": method_info,
                    "predictions_key": ens_pred_key,
                    "result_key": ens_json_key,
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            redis_client.xadd(stream_key, {"type": "completed"})
            logger.info("Ensemble job %s done: %s", job_id, ensemble_metrics)

        except Exception:
            logger.exception("Ensemble job %s failed", job_id)
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
