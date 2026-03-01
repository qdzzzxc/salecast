import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.automl.config import AutoMLConfig
from src.automl.models import CatBoostForecastModel, SeasonalNaiveForecastModel, StatsForecastModel
from src.automl.selector import _get_metric_value
from src.configs.settings import ColumnConfig, Settings
from src.custom_types import MetricType, ModelType, Splits

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
    return redis.Redis(host=_redis_host, port=_redis_port, password=_redis_password, decode_responses=True)


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


def _add_step(session: Session, job, name: str, message: str) -> None:
    from api.models import Job
    steps = list(job.steps)
    steps.append({"name": name, "message": message, "timestamp": datetime.now(timezone.utc).isoformat()})
    session.execute(
        Job.__table__.update().where(Job.__table__.c.id == job.id).values(steps=steps, status="running")
    )
    session.commit()
    job.steps = steps


def _build_model(model_type: ModelType):
    if model_type == "seasonal_naive":
        return SeasonalNaiveForecastModel()
    if model_type == "catboost":
        return CatBoostForecastModel()
    return StatsForecastModel(model_type=model_type)


def _extract_metric(result, metric: str, split_name: str) -> float:
    """Извлекает значение метрики из ModelResult."""
    for split_eval in result.evaluation.splits:
        if split_eval.split_name == split_name:
            return float(getattr(split_eval.overall_metrics, metric, float("inf")))
    return float("inf")


def _extract_panel_metrics(result, metric: str) -> list[dict]:
    """Возвращает per-panel метрики val и test."""
    val_by_panel, test_by_panel = {}, {}
    for split_eval in result.evaluation.splits:
        if split_eval.split_name == "val":
            for pm in split_eval.panel_metrics:
                val_by_panel[str(pm.panel_id)] = float(getattr(pm.metrics, metric, float("inf")))
        elif split_eval.split_name == "test":
            for pm in split_eval.panel_metrics:
                test_by_panel[str(pm.panel_id)] = float(getattr(pm.metrics, metric, float("inf")))
    panel_ids = set(val_by_panel) | set(test_by_panel)
    return [
        {
            "panel_id": pid,
            "val": val_by_panel.get(pid, None),
            "test": test_by_panel.get(pid, None),
        }
        for pid in sorted(panel_ids)
    ]


@celery.task(bind=True, name="worker.tasks.run_automl.run_automl")
def run_automl(
    self,
    job_id: str,
    project_id: str,
    preprocessing_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    models: list[str],
    selection_metric: str,
    use_hyperopt: bool,
) -> dict:
    """Запускает AutoML: обучает модели на отфильтрованных панелях, выбирает лучшую."""
    from api.models import Job

    engine = _get_engine()
    redis_client = _get_redis()
    stream_key = f"stream:automl:{job_id}"

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        prep_job = session.execute(select(Job).where(Job.id == preprocessing_job_id)).scalar_one()

        try:
            _add_step(session, job, "loading", "Загрузка train/val/test из хранилища")
            split_info = prep_job.result["split"]
            train_df = _load_csv(split_info["train_key"])
            val_df = _load_csv(split_info["val_key"])
            test_df = _load_csv(split_info["test_key"])

            # Фильтруем только green+yellow панели
            diag_panels = prep_job.result["diagnostics"]["panels"]
            good_panels = {
                str(p["panel_id"])
                for p in diag_panels
                if p.get("overall_status") in ("green", "yellow")
            }
            train_df = train_df[train_df[panel_col].astype(str).isin(good_panels)]
            val_df = val_df[val_df[panel_col].astype(str).isin(good_panels)]
            test_df = test_df[test_df[panel_col].astype(str).isin(good_panels)]
            logger.info("AutoML: %d панелей (green+yellow)", len(good_panels))

            train_df[date_col] = pd.to_datetime(train_df[date_col])
            val_df[date_col] = pd.to_datetime(val_df[date_col])
            test_df[date_col] = pd.to_datetime(test_df[date_col])

            splits = Splits(train=train_df, val=val_df, test=test_df)
            settings = Settings().model_copy(
                update={"columns": ColumnConfig(id=panel_col, date=date_col, main_target=value_col)}
            )

            all_results = []
            for i, model_type in enumerate(models):
                _add_step(session, job, f"train_{model_type}", f"Обучение {model_type} ({i + 1}/{len(models)})")
                redis_client.xadd(stream_key, {
                    "type": "model_start",
                    "model": model_type,
                    "n": str(i + 1),
                    "total": str(len(models)),
                })

                model = _build_model(model_type)
                result = model.fit_evaluate(splits, settings)
                all_results.append(result)

                val_mape = _extract_metric(result, selection_metric, "val")
                test_mape = _extract_metric(result, selection_metric, "test")
                redis_client.xadd(stream_key, {
                    "type": "model_done",
                    "model": model_type,
                    f"val_{selection_metric}": f"{val_mape:.4f}",
                    f"test_{selection_metric}": f"{test_mape:.4f}",
                })
                logger.info("Модель %s: val_%s=%.4f test_%s=%.4f", model_type, selection_metric, val_mape, selection_metric, test_mape)

            best = min(
                all_results,
                key=lambda r: _get_metric_value(r, selection_metric, "val"),
            )
            redis_client.xadd(stream_key, {"type": "best", "model": best.name})

            _add_step(session, job, "saving", "Сохранение результатов AutoML")

            model_results = []
            for r in all_results:
                model_results.append({
                    "name": r.name,
                    f"val_{selection_metric}": _extract_metric(r, selection_metric, "val"),
                    f"test_{selection_metric}": _extract_metric(r, selection_metric, "test"),
                    "panel_metrics": _extract_panel_metrics(r, selection_metric),
                })

            result_data = {
                "split": prep_job.result["split"],
                "automl": {
                    "models_used": models,
                    "selection_metric": selection_metric,
                    "best_model": best.name,
                    "total_panels": len(good_panels),
                    "model_results": model_results,
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            redis_client.xadd(stream_key, {"type": "completed"})
            logger.info("AutoML job %s завершён, лучшая модель: %s", job_id, best.name)

        except Exception:
            logger.exception("AutoML job %s завершился с ошибкой", job_id)
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
