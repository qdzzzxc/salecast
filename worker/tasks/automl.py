import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.configs.settings import FiltrationConfig, Settings
from src.diagnostics import run_diagnostics
from src.filtration import filter_time_series

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


def _get_engine():
    """Возвращает синхронный SQLAlchemy engine."""
    return create_engine(_SYNC_DB_URL)


def _download_csv(csv_key: str) -> bytes:
    """Скачивает CSV файл из MinIO синхронно."""
    import boto3
    from botocore.client import Config

    client = boto3.client(
        "s3",
        endpoint_url=_minio_endpoint,
        aws_access_key_id=_minio_access,
        aws_secret_access_key=_minio_secret,
        config=Config(signature_version="s3v4"),
    )
    response = client.get_object(Bucket=_minio_bucket, Key=csv_key)
    return response["Body"].read()


def _add_step(session: Session, job, name: str, message: str) -> None:
    """Добавляет шаг прогресса к job и сохраняет в БД."""
    from api.models import Job

    steps = list(job.steps)
    steps.append({"name": name, "message": message, "timestamp": datetime.now(timezone.utc).isoformat()})
    session.execute(
        Job.__table__.update().where(Job.__table__.c.id == job.id).values(steps=steps, status="running")
    )
    session.commit()
    job.steps = steps


@celery.task(bind=True, name="worker.tasks.automl.run_preprocessing")
def run_preprocessing(
    self,
    job_id: str,
    project_id: str,
    csv_key: str,
    panel_col: str,
    date_col: str,
    value_col: str,
) -> dict:
    """Запускает фильтрацию и диагностику временных рядов."""
    from api.models import Job

    engine = _get_engine()
    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()

        try:
            _add_step(session, job, "loading", "Загрузка данных из хранилища")
            csv_bytes = _download_csv(csv_key)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            logger.info("Загружен CSV: %d строк, колонки: %s", len(df), list(df.columns))

            _add_step(session, job, "filtration", "Фильтрация временных рядов")
            settings = Settings()
            filtration_config = FiltrationConfig(
                columns=settings.filtration.columns.model_copy(
                    update={"id": panel_col, "date": date_col, "main_target": value_col}
                )
            )
            df[date_col] = pd.to_datetime(df[date_col])
            filtration_result = filter_time_series(df, filtration_config)
            filtration_summary = filtration_result.summary()
            logger.info("Фильтрация завершена: осталось %d панелей", filtration_result.df[panel_col].nunique())

            _add_step(session, job, "diagnostics", "Диагностика качества рядов")
            diagnostics_result = run_diagnostics(
                filtration_result.df,
                panel_col=panel_col,
                date_col=date_col,
                value_col=value_col,
                config=settings.diagnostics,
            )
            diagnostics_summary = diagnostics_result.summary()
            logger.info("Диагностика завершена: %s", diagnostics_summary)

            _add_step(session, job, "saving", "Сохранение результатов")
            result = {
                "filtration": {
                    "total_before": df[panel_col].nunique(),
                    "total_after": filtration_result.df[panel_col].nunique(),
                    "steps": filtration_summary,
                },
                "diagnostics": {
                    "summary": diagnostics_summary,
                    "panels": diagnostics_result.to_df().to_dict(orient="records"),
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(
                    status="done",
                    result=result,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
            logger.info("Job %s завершён", job_id)

        except Exception:
            logger.exception("Job %s завершился с ошибкой", job_id)
            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="failed", completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            raise
        else:
            return result
