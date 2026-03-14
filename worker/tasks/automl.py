import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.automl.ts_utils import infer_ts_config
from src.configs.settings import FiltrationConfig, Settings
from src.diagnostics import run_diagnostics
from src.filtration import filter_time_series
from src.model_selection import temporal_panel_split_by_size

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


def _get_s3_client():
    """Возвращает boto3 S3 клиент для MinIO."""
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=_minio_endpoint,
        aws_access_key_id=_minio_access,
        aws_secret_access_key=_minio_secret,
        config=Config(signature_version="s3v4"),
    )


def _download_csv(csv_key: str) -> bytes:
    """Скачивает CSV файл из MinIO синхронно."""
    client = _get_s3_client()
    response = client.get_object(Bucket=_minio_bucket, Key=csv_key)
    return response["Body"].read()


def _upload_csv(key: str, df: pd.DataFrame) -> None:
    """Загружает DataFrame как CSV в MinIO."""
    client = _get_s3_client()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    client.put_object(Bucket=_minio_bucket, Key=key, Body=buf.getvalue(), ContentType="text/csv")


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


def _split_panels(
    df: pd.DataFrame,
    panel_col: str,
    date_col: str,
    value_col: str,
    val_periods: int,
    test_periods: int,
) -> tuple:
    """Разбивает панели на train/val/test, пропуская слишком короткие ряды."""
    min_len = val_periods + test_periods + 1
    panels_before = df[panel_col].nunique()

    enough = df.groupby(panel_col)[value_col].transform("count") > min_len
    df_ok = df[enough]
    dropped = panels_before - df_ok[panel_col].nunique()

    splits = temporal_panel_split_by_size(
        df_ok,
        panel_column=panel_col,
        time_column=date_col,
        val_size=val_periods,
        test_size=test_periods,
    )
    return splits, dropped, panels_before


@celery.task(bind=True, name="worker.tasks.automl.run_preprocessing")
def run_preprocessing(
    self,
    job_id: str,
    project_id: str,
    csv_key: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    val_periods: int = 6,
    test_periods: int = 6,
) -> dict:
    """Запускает фильтрацию, диагностику и разбивку временных рядов."""
    from api.models import Job

    engine = _get_engine()
    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()

        try:
            # --- Загрузка ---
            _add_step(session, job, "loading", "Загрузка данных из хранилища")
            csv_bytes = _download_csv(csv_key)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            df[date_col] = pd.to_datetime(df[date_col])
            logger.info("Загружен CSV: %d строк, колонки: %s", len(df), list(df.columns))
            ts_config = infer_ts_config(df, date_col)
            logger.info("Инференс частоты: freq=%s season_length=%d", ts_config.freq, ts_config.season_length)

            # --- Фильтрация ---
            _add_step(session, job, "filtration", "Фильтрация временных рядов")
            settings = Settings()
            filtration_config = FiltrationConfig(
                columns=settings.filtration.columns.model_copy(
                    update={"id": panel_col, "date": date_col, "main_target": value_col}
                )
            )
            filtration_result = filter_time_series(df, filtration_config)
            logger.info("Фильтрация: осталось %d панелей", filtration_result.df[panel_col].nunique())

            filtered_samples: dict = {}
            for step_report in filtration_result.steps:
                if not step_report.dropped_ids:
                    continue
                filtered_samples[step_report.step] = {
                    "reason": step_report.reason,
                    "total": len(step_report.dropped_ids),
                    "panel_ids": [str(pid) for pid in step_report.dropped_ids],
                }

            # --- Диагностика ---
            _add_step(session, job, "diagnostics", "Диагностика качества рядов")
            diagnostics_result = run_diagnostics(
                filtration_result.df,
                panel_col=panel_col,
                date_col=date_col,
                value_col=value_col,
                config=settings.diagnostics,
            )
            logger.info("Диагностика: %s", diagnostics_result.summary())

            # --- Разбивка ---
            _add_step(session, job, "split", f"Разбивка на train / val ({val_periods} п.) / test ({test_periods} п.)")
            splits, dropped_by_split, panels_before_split = _split_panels(
                filtration_result.df, panel_col, date_col, value_col, val_periods, test_periods
            )
            logger.info(
                "Split: train=%d, val=%d, test=%d строк; пропущено панелей: %d",
                len(splits.train), len(splits.val), len(splits.test), dropped_by_split,
            )

            # --- Сохранение ---
            _add_step(session, job, "saving", "Сохранение результатов")

            train_key = f"projects/{project_id}/train.csv"
            val_key = f"projects/{project_id}/val.csv"
            test_key = f"projects/{project_id}/test.csv"
            _upload_csv(train_key, splits.train)
            _upload_csv(val_key, splits.val)
            _upload_csv(test_key, splits.test)

            result = {
                "ts": {
                    "freq": ts_config.freq,
                    "season_length": ts_config.season_length,
                },
                "filtration": {
                    "total_before": df[panel_col].nunique(),
                    "total_after": filtration_result.df[panel_col].nunique(),
                    "steps": filtration_result.summary(),
                    "filtered_samples": filtered_samples,
                },
                "diagnostics": {
                    "summary": diagnostics_result.summary(),
                    "panels": diagnostics_result.to_df().to_dict(orient="records"),
                },
                "split": {
                    "val_periods": val_periods,
                    "test_periods": test_periods,
                    "panels_before": panels_before_split,
                    "panels_after": panels_before_split - dropped_by_split,
                    "panels_dropped": dropped_by_split,
                    "train_key": train_key,
                    "val_key": val_key,
                    "test_key": test_key,
                    "panel_ids": [str(p) for p in splits.train[panel_col].unique().tolist()],
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result, completed_at=datetime.now(timezone.utc))
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
