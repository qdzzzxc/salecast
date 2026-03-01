import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.models import Job, Project
from api.storage import upload_file
from worker.tasks.automl import run_preprocessing

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    """Параметры создания проекта."""

    name: str
    panel_col: str
    date_col: str
    value_col: str


class JobSchema(BaseModel):
    """Схема задачи для ответа API."""

    id: uuid.UUID
    status: str
    steps: list[dict[str, Any]]
    result: dict[str, Any] | None
    created_at: str
    completed_at: str | None

    model_config = {"from_attributes": True}


class ProjectSchema(BaseModel):
    """Схема проекта для ответа API."""

    id: uuid.UUID
    name: str
    panel_col: str
    date_col: str
    value_col: str
    created_at: str
    latest_job: JobSchema | None

    model_config = {"from_attributes": True}


def _to_job_schema(job: Job) -> JobSchema:
    """Конвертирует ORM объект Job в схему."""
    return JobSchema(
        id=job.id,
        status=job.status,
        steps=job.steps,
        result=job.result,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


def _to_project_schema(project: Project) -> ProjectSchema:
    """Конвертирует ORM объект Project в схему."""
    sorted_jobs = sorted(project.jobs, key=lambda j: j.created_at)
    latest_job = _to_job_schema(sorted_jobs[-1]) if sorted_jobs else None
    return ProjectSchema(
        id=project.id,
        name=project.name,
        panel_col=project.panel_col,
        date_col=project.date_col,
        value_col=project.value_col,
        created_at=project.created_at.isoformat(),
        latest_job=latest_job,
    )


@router.post("", response_model=ProjectSchema)
async def create_project(
    file: UploadFile,
    name: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    db: AsyncSession = Depends(get_db),
) -> ProjectSchema:
    """Создаёт проект: сохраняет CSV в MinIO. Обработка запускается отдельно."""
    project_id = uuid.uuid4()
    csv_key = f"projects/{project_id}/data.csv"

    content = await file.read()
    await upload_file(csv_key, content, content_type="text/csv")

    project = Project(
        id=project_id,
        name=name,
        csv_key=csv_key,
        panel_col=panel_col,
        date_col=date_col,
        value_col=value_col,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    return ProjectSchema(
        id=project.id,
        name=project.name,
        panel_col=project.panel_col,
        date_col=project.date_col,
        value_col=project.value_col,
        created_at=project.created_at.isoformat(),
        latest_job=None,
    )


@router.post("/{project_id}/run", response_model=JobSchema)
async def run_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> JobSchema:
    """Запускает Celery-задачу обработки данных для проекта."""
    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    job = Job(id=uuid.uuid4(), project_id=project_id, status="pending", steps=[], result=None)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    run_preprocessing.delay(
        str(job.id),
        str(project_id),
        project.csv_key,
        project.panel_col,
        project.date_col,
        project.value_col,
    )

    return _to_job_schema(job)


@router.get("", response_model=list[ProjectSchema])
async def list_projects(db: AsyncSession = Depends(get_db)) -> list[ProjectSchema]:
    """Возвращает список всех проектов с последним job."""
    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).order_by(Project.created_at.desc())
    )
    projects = result.scalars().all()
    return [_to_project_schema(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectSchema)
async def get_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> ProjectSchema:
    """Возвращает проект по ID с последним job."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.jobs))
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return _to_project_schema(project)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Response:
    """Удаляет проект и все связанные задачи."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    await db.execute(delete(Job).where(Job.project_id == project_id))
    await db.execute(delete(Project).where(Project.id == project_id))
    await db.commit()
    return Response(status_code=204)
