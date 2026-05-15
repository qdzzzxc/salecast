"""API endpoints для ансамблирования моделей."""

import asyncio
import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.models import Job, Project
from api.routers.projects import JobSchema, _to_job_schema

router = APIRouter(prefix="/projects", tags=["ensemble"])

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


def _get_automl_job(project: Project) -> Job | None:
    return next(
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "automl" in j.result
        ),
        None,
    )


class RunEnsembleRequest(BaseModel):
    models: list[str]
    method: str = "weighted_avg"  # "weighted_avg" | "best_per_panel"


@router.post("/{project_id}/run_ensemble", response_model=JobSchema)
async def run_ensemble(
    project_id: uuid.UUID,
    config: RunEnsembleRequest,
    db: AsyncSession = Depends(get_db),
) -> JobSchema:
    """Запускает расчёт ансамбля предсказаний."""
    from worker.tasks.ensemble import run_ensemble as celery_run_ensemble

    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    automl_job = _get_automl_job(project)
    if automl_job is None:
        raise HTTPException(status_code=400, detail="Сначала завершите шаг моделирования")

    if len(config.models) < 2:
        raise HTTPException(status_code=400, detail="Нужно минимум 2 модели для ансамбля")

    automl_info = automl_job.result.get("automl", {})

    job = Job(id=uuid.uuid4(), project_id=project_id, status="pending", steps=[], result=None)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    celery_run_ensemble.delay(
        str(job.id),
        str(project_id),
        str(automl_job.id),
        project.panel_col,
        project.date_col,
        project.value_col,
        config.models,
        config.method,
        automl_info.get("selection_metric", "mape"),
    )

    return _to_job_schema(job)


@router.get("/{project_id}/ensemble_progress/{job_id}")
async def get_ensemble_progress(
    project_id: uuid.UUID,
    job_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Возвращает события прогресса ансамбля из Redis Stream."""
    redis = await _get_redis()
    try:
        events = await redis.xrange(f"stream:ensemble:{job_id}", "-", "+")
        return [fields for _, fields in events]
    finally:
        await redis.aclose()


@router.get("/{project_id}/ensemble_result")
async def get_ensemble_result(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Возвращает результат ансамбля из MinIO."""
    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    ens_job = next(
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "ensemble" in j.result
        ),
        None,
    )
    if ens_job is None:
        raise HTTPException(status_code=404, detail="Нет результатов ансамбля")

    result_key = ens_job.result["ensemble"]["result_key"]
    loop = asyncio.get_running_loop()

    def _load_json() -> dict:
        import boto3
        from botocore.client import Config

        client = boto3.client(
            "s3",
            endpoint_url=_minio_endpoint,
            aws_access_key_id=_minio_access,
            aws_secret_access_key=_minio_secret,
            config=Config(signature_version="s3v4"),
        )
        resp = client.get_object(Bucket=_minio_bucket, Key=result_key)
        return json.loads(resp["Body"].read())

    return await loop.run_in_executor(None, _load_json)
