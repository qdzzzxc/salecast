import asyncio
import io
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.models import Job, Project
from api.routers.projects import JobSchema, _to_job_schema

router = APIRouter(prefix="/projects", tags=["forecast"])

_minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
_minio_access = os.getenv("MINIO_ROOT_USER", "sales_ts_prediction")
_minio_secret = os.getenv("MINIO_ROOT_PASSWORD", "sales_ts_prediction")
_minio_bucket = os.getenv("MINIO_BUCKET", "salecast")

_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port = int(os.getenv("REDIS_PORT", "6379"))
_redis_password = os.getenv("REDIS_PASSWORD", "")


async def _get_redis():
    import redis.asyncio as aioredis

    return aioredis.Redis(
        host=_redis_host, port=_redis_port, password=_redis_password, decode_responses=True
    )


def _load_forecast_csv(key: str) -> pd.DataFrame:
    import boto3
    from botocore.client import Config

    client = boto3.client(
        "s3",
        endpoint_url=_minio_endpoint,
        aws_access_key_id=_minio_access,
        aws_secret_access_key=_minio_secret,
        config=Config(signature_version="s3v4"),
    )
    resp = client.get_object(Bucket=_minio_bucket, Key=key)
    return pd.read_csv(io.BytesIO(resp["Body"].read()))


def _get_forecast_job(project: Project) -> Job | None:
    return next(
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "forecast" in j.result
        ),
        None,
    )


def _get_automl_job(project: Project) -> Job | None:
    return next(
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "automl" in j.result
        ),
        None,
    )


class ForecastRunConfig(BaseModel):
    model_name: str
    horizon: int = 6
    panel_ids: list[str] = []


class RunCVRequest(BaseModel):
    model_type: str
    n_folds: int = 5
    ensemble_models: list[str] | None = None
    ensemble_method: str = "weighted_avg"


@router.post("/{project_id}/run_forecast", response_model=JobSchema)
async def run_forecast(
    project_id: uuid.UUID,
    config: ForecastRunConfig,
    db: AsyncSession = Depends(get_db),
) -> JobSchema:
    """Запускает построение прогноза для проекта."""
    from worker.tasks.forecast import run_forecast as celery_run_forecast

    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    automl_job = _get_automl_job(project)
    if automl_job is None:
        raise HTTPException(status_code=400, detail="Сначала завершите шаг моделирования")

    job = Job(id=uuid.uuid4(), project_id=project_id, status="pending", steps=[], result=None)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    celery_run_forecast.delay(
        str(job.id),
        str(project_id),
        str(automl_job.id),
        project.panel_col,
        project.date_col,
        project.value_col,
        config.model_name,
        config.horizon,
        config.panel_ids,
    )

    return _to_job_schema(job)


@router.get("/{project_id}/forecast_data")
async def get_forecast_data(
    project_id: uuid.UUID,
    panel_ids: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[dict[str, Any]]]:
    """Возвращает прогноз по панелям: {panel_id: [{date, forecast}]}."""
    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    forecast_job = _get_forecast_job(project)
    if forecast_job is None:
        raise HTTPException(status_code=404, detail="Нет результатов прогноза")

    forecast_key = forecast_job.result["forecast"]["forecast_key"]
    requested = set(panel_ids.split(",")) if panel_ids else None

    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, _load_forecast_csv, forecast_key)
    df["panel_id"] = df["panel_id"].astype(str)

    if requested:
        df = df[df["panel_id"].isin(requested)]

    result_dict: dict[str, list[dict[str, Any]]] = {}
    for pid, group in df.groupby("panel_id"):
        result_dict[str(pid)] = group[["date", "forecast"]].to_dict("records")

    return result_dict


@router.get("/{project_id}/forecast_csv")
async def download_forecast_csv(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Скачать прогноз как CSV."""
    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    forecast_job = _get_forecast_job(project)
    if forecast_job is None:
        raise HTTPException(status_code=404, detail="Нет результатов прогноза")

    forecast_key = forecast_job.result["forecast"]["forecast_key"]

    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, _load_forecast_csv, forecast_key)
    csv_bytes = df.to_csv(index=False).encode()

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=forecast_{project_id}.csv"},
    )


@router.get("/{project_id}/forecast_progress/{job_id}")
async def get_forecast_progress(
    project_id: uuid.UUID,
    job_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Возвращает события прогресса прогноза из Redis Stream."""
    redis = await _get_redis()
    try:
        events = await redis.xrange(f"stream:forecast:{job_id}", "-", "+")
        return [fields for _, fields in events]
    finally:
        await redis.aclose()


# ─── Cross-Validation ───────────────────────────────────────────────


@router.post("/{project_id}/run_cv", response_model=JobSchema)
async def run_cv(
    project_id: uuid.UUID,
    config: RunCVRequest,
    db: AsyncSession = Depends(get_db),
) -> JobSchema:
    """Запускает temporal cross-validation для выбранной модели."""
    from worker.tasks.cross_validation import run_cross_validation as celery_run_cv

    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    automl_job = _get_automl_job(project)
    if automl_job is None:
        raise HTTPException(status_code=400, detail="Сначала завершите шаг моделирования")

    automl_info = automl_job.result.get("automl", {})

    job = Job(id=uuid.uuid4(), project_id=project_id, status="pending", steps=[], result=None)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    celery_run_cv.delay(
        str(job.id),
        str(project_id),
        str(automl_job.id),
        project.panel_col,
        project.date_col,
        project.value_col,
        config.model_type,
        config.n_folds,
        automl_info.get("ts", {}).get("freq"),
        None,  # catboost_params (use defaults)
        automl_info.get("feature_params"),
        automl_info.get("chronos_params"),
        automl_info.get("ts2vec_params"),
        config.ensemble_models,
        config.ensemble_method,
    )

    return _to_job_schema(job)


@router.get("/{project_id}/cv_progress/{job_id}")
async def get_cv_progress(
    project_id: uuid.UUID,
    job_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Возвращает события прогресса кросс-валидации из Redis Stream."""
    redis = await _get_redis()
    try:
        events = await redis.xrange(f"stream:cv:{job_id}", "-", "+")
        return [fields for _, fields in events]
    finally:
        await redis.aclose()


@router.get("/{project_id}/cv_result")
async def get_cv_result(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Возвращает результат кросс-валидации из MinIO."""
    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    # Находим последний завершённый CV job
    cv_job = next(
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "cross_validation" in j.result
        ),
        None,
    )
    if cv_job is None:
        raise HTTPException(status_code=404, detail="Нет результатов кросс-валидации")

    cv_key = cv_job.result["cross_validation"]["cv_key"]
    loop = asyncio.get_running_loop()
    import json

    def _load_json():
        import boto3
        from botocore.client import Config

        client = boto3.client(
            "s3",
            endpoint_url=_minio_endpoint,
            aws_access_key_id=_minio_access,
            aws_secret_access_key=_minio_secret,
            config=Config(signature_version="s3v4"),
        )
        resp = client.get_object(Bucket=_minio_bucket, Key=cv_key)
        return json.loads(resp["Body"].read())

    return await loop.run_in_executor(None, _load_json)
