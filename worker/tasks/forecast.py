import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.automl.models import CatBoostForecastModel, SeasonalNaiveForecastModel, StatsForecastModel
from src.automl.ts_utils import get_downstream_lags, infer_ts_config
from src.configs.settings import ColumnConfig, Settings

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


def _upload_csv(key: str, df: pd.DataFrame) -> None:
    client = _get_s3()
    buf = df.to_csv(index=False).encode()
    client.put_object(Bucket=_minio_bucket, Key=key, Body=buf, ContentLength=len(buf))


def _add_step(session: Session, job, name: str, message: str) -> None:
    from api.models import Job
    steps = list(job.steps)
    steps.append({"name": name, "message": message, "timestamp": datetime.now(timezone.utc).isoformat()})
    session.execute(
        Job.__table__.update().where(Job.__table__.c.id == job.id).values(steps=steps, status="running")
    )
    session.commit()
    job.steps = steps


def _build_model(model_name: str):
    """Создаёт экземпляр модели по имени."""
    if model_name == "seasonal_naive":
        return SeasonalNaiveForecastModel()
    if model_name == "catboost":
        return CatBoostForecastModel()
    if model_name in ("autoarima", "autoets", "autotheta"):
        return StatsForecastModel(model_type=model_name)
    raise ValueError(f"Неизвестная модель: {model_name}")


@celery.task(bind=True, name="worker.tasks.forecast.run_forecast")
def run_forecast(
    self,
    job_id: str,
    project_id: str,
    automl_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    model_name: str,
    horizon: int,
    panel_ids: list[str],
) -> dict:
    """Обучает выбранную модель на всех данных и строит прогноз на horizon точек вперёд."""
    from api.models import Job

    engine = _get_engine()

    redis_client = None
    try:
        redis_client = _get_redis()
    except Exception:
        pass

    stream_key = f"stream:forecast:{job_id}"

    def _publish(event: dict) -> None:
        if redis_client:
            try:
                redis_client.xadd(stream_key, event)
            except Exception:
                pass

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        automl_job = session.execute(select(Job).where(Job.id == automl_job_id)).scalar_one()

        try:
            _publish({"type": "step_start", "step": "loading"})
            _add_step(session, job, "loading", "Загрузка данных")
            split_info = automl_job.result["split"]
            train_df = _load_csv(split_info["train_key"])
            val_df = _load_csv(split_info["val_key"])
            test_df = _load_csv(split_info["test_key"])

            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            full_df[date_col] = pd.to_datetime(full_df[date_col])
            full_df = full_df.sort_values([panel_col, date_col]).reset_index(drop=True)

            if panel_ids:
                full_df = full_df[full_df[panel_col].astype(str).isin(set(panel_ids))]

            ts_config = infer_ts_config(full_df, date_col)
            lags = get_downstream_lags(ts_config.freq)
            logger.info("Forecast: freq=%s, season_length=%d", ts_config.freq, ts_config.season_length)
            base_settings = Settings()
            settings = base_settings.model_copy(
                update={
                    "columns": ColumnConfig(id=panel_col, date=date_col, main_target=value_col),
                    "ts": ts_config,
                    "downstream": base_settings.downstream.model_copy(update={"lags": lags}),
                }
            )

            _publish({"type": "step_start", "step": "training"})
            _add_step(session, job, "forecasting", f"Прогноз {model_name} на {horizon} точек")

            model = _build_model(model_name)
            forecast_df = model.forecast_future(
                full_df=full_df,
                horizon=horizon,
                settings=settings,
                on_training_done=lambda: _publish({"type": "step_start", "step": "forecasting"}),
                on_forecast_step=lambda i, n: _publish({
                    "type": "forecast_step", "step_i": str(i), "total": str(n),
                }),
            )

            forecast_key = f"projects/{project_id}/forecast.csv"
            _publish({"type": "step_start", "step": "saving"})
            _add_step(session, job, "saving", "Сохранение прогноза")
            _upload_csv(forecast_key, forecast_df)

            result_data = {
                "split": automl_job.result["split"],
                "automl": automl_job.result["automl"],
                "forecast": {
                    "model": model_name,
                    "horizon": horizon,
                    "panel_count": int(forecast_df["panel_id"].nunique()),
                    "forecast_key": forecast_key,
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            logger.info("Forecast job %s done: model=%s horizon=%d", job_id, model_name, horizon)
            _publish({"type": "completed"})

        except Exception:
            logger.exception("Forecast job %s failed", job_id)
            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="failed", completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            _publish({"type": "failed"})
            raise
        else:
            return result_data
