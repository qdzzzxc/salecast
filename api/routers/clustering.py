import io
import os
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.models import Job, Project
from api.routers.projects import JobSchema, _to_job_schema

router = APIRouter(prefix="/projects", tags=["clustering"])

_minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
_minio_access = os.getenv("MINIO_ROOT_USER", "sales_ts_prediction")
_minio_secret = os.getenv("MINIO_ROOT_PASSWORD", "sales_ts_prediction")
_minio_bucket = os.getenv("MINIO_BUCKET", "salecast")


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


def _load_csv_from_minio(key: str) -> pd.DataFrame:
    client = _get_s3()
    resp = client.get_object(Bucket=_minio_bucket, Key=key)
    return pd.read_csv(io.BytesIO(resp["Body"].read()))


class ClusteringRunConfig(BaseModel):
    """Конфигурация запуска кластеризации."""

    n_clusters: int = 5
    method: str = "kmeans"  # "kmeans" | "hdbscan" | "kmeans_auto"
    preprocessing_job_id: str | None = None  # None → берём последний preprocessing job


@router.post("/{project_id}/run_clustering", response_model=JobSchema)
async def run_clustering(
    project_id: uuid.UUID,
    config: ClusteringRunConfig = ClusteringRunConfig(),
    db: AsyncSession = Depends(get_db),
) -> JobSchema:
    """Запускает кластеризацию панелей. Требует завершённого preprocessing."""
    from worker.tasks.clustering import run_clustering as celery_run_clustering

    result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    if config.preprocessing_job_id:
        prep_job = next(
            (j for j in project.jobs if str(j.id) == config.preprocessing_job_id),
            None,
        )
    else:
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

    celery_run_clustering.delay(
        str(job.id),
        str(project_id),
        str(prep_job.id),
        project.panel_col,
        project.date_col,
        project.value_col,
        config.n_clusters,
        config.method,
    )

    return _to_job_schema(job)


@router.get("/{project_id}/cluster_data")
async def get_cluster_data(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Возвращает данные кластеризации: UMAP-координаты + средние TS по кластерам."""
    import asyncio

    db_result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = db_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    clustering_job = next(
        (j for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
         if j.status == "done" and j.result and "clustering" in j.result),
        None,
    )
    if clustering_job is None:
        raise HTTPException(status_code=404, detail="Нет результатов кластеризации")

    clustering = clustering_job.result["clustering"]
    loop = asyncio.get_running_loop()

    umap_df, mean_ts_df = await asyncio.gather(
        loop.run_in_executor(None, _load_csv_from_minio, clustering["umap_key"]),
        loop.run_in_executor(None, _load_csv_from_minio, clustering["mean_ts_key"]),
    )

    umap_df["cluster_id"] = umap_df["cluster_id"].astype(int)
    mean_ts_df["cluster_id"] = mean_ts_df["cluster_id"].astype(int)

    response: dict = {
        "n_clusters": clustering["n_clusters"],
        "n_panels": clustering["n_panels"],
        "n_outliers": clustering.get("n_outliers", 0),
        "method": clustering["method"],
        "umap": umap_df.to_dict("records"),
        "mean_ts": mean_ts_df.to_dict("records"),
    }
    if "silhouette_scores" in clustering:
        response["silhouette_scores"] = clustering["silhouette_scores"]
        response["best_k"] = clustering.get("best_k")
    return response
