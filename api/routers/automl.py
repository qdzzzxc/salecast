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
from api.routers.projects import _to_job_schema, JobSchema

router = APIRouter(prefix="/projects", tags=["automl"])

_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port = int(os.getenv("REDIS_PORT", "6379"))
_redis_password = os.getenv("REDIS_PASSWORD", "")


async def _get_redis():
    import redis.asyncio as aioredis
    return aioredis.Redis(
        host=_redis_host, port=_redis_port, password=_redis_password, decode_responses=True
    )


class AutoMLRunConfig(BaseModel):
    """Конфигурация запуска AutoML."""

    models: list[str] = ["seasonal_naive", "catboost"]
    selection_metric: str = "mape"
    use_hyperopt: bool = False


@router.post("/{project_id}/run_automl", response_model=JobSchema)
async def run_automl(
    project_id: uuid.UUID,
    config: AutoMLRunConfig = AutoMLRunConfig(),
    db: AsyncSession = Depends(get_db),
) -> JobSchema:
    """Запускает AutoML для проекта. Требует завершённого шага preprocessing."""
    from worker.tasks.run_automl import run_automl as celery_run_automl

    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    # Ищем последний preprocessing job (должен содержать split в результате)
    prep_job = next(
        (j for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
         if j.status == "done" and j.result and "split" in j.result),
        None,
    )
    if prep_job is None:
        raise HTTPException(status_code=400, detail="Сначала завершите шаг обработки данных")

    job = Job(id=uuid.uuid4(), project_id=project_id, status="pending", steps=[], result=None)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    celery_run_automl.delay(
        str(job.id),
        str(project_id),
        str(prep_job.id),
        project.panel_col,
        project.date_col,
        project.value_col,
        config.models,
        config.selection_metric,
        config.use_hyperopt,
    )

    return _to_job_schema(job)


@router.get("/{project_id}/automl_progress/{job_id}")
async def get_automl_progress(
    project_id: uuid.UUID,
    job_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Возвращает события прогресса AutoML из Redis Stream."""
    redis = await _get_redis()
    try:
        events = await redis.xrange(f"stream:automl:{job_id}", "-", "+")
        return [fields for _, fields in events]
    finally:
        await redis.aclose()
