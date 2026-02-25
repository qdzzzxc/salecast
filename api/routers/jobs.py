import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import Job

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobSchema(BaseModel):
    """Схема задачи для ответа API."""

    id: uuid.UUID
    project_id: uuid.UUID
    status: str
    steps: list[dict[str, Any]]
    result: dict[str, Any] | None
    created_at: str
    completed_at: str | None


@router.get("/{job_id}", response_model=JobSchema)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> JobSchema:
    """Возвращает статус и результат задачи по ID."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return JobSchema(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        steps=job.steps,
        result=job.result,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )
