import asyncio
import io
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.models import Job, Project
from api.routers.projects import JobSchema, _to_job_schema

router = APIRouter(prefix="/projects", tags=["automl"])

_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port = int(os.getenv("REDIS_PORT", "6379"))
_redis_password = os.getenv("REDIS_PASSWORD", "")

_minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
_minio_access = os.getenv("MINIO_ROOT_USER", "sales_ts_prediction")
_minio_secret = os.getenv("MINIO_ROOT_PASSWORD", "sales_ts_prediction")
_minio_bucket = os.getenv("MINIO_BUCKET", "salecast")


def _load_model_predictions(key: str, panel_ids: set[str]) -> dict[str, list[dict]]:
    """Загружает предсказания модели из MinIO, фильтрует по panel_ids."""
    import boto3
    from botocore.client import Config

    client = boto3.client(
        "s3",
        endpoint_url=_minio_endpoint,
        aws_access_key_id=_minio_access,
        aws_secret_access_key=_minio_secret,
        config=Config(signature_version="s3v4"),
    )
    try:
        resp = client.get_object(Bucket=_minio_bucket, Key=key)
        df = pd.read_csv(io.BytesIO(resp["Body"].read()))
        df["panel_id"] = df["panel_id"].astype(str)
        df = df[df["panel_id"].isin(panel_ids)]
        result: dict[str, list[dict]] = {}
        for panel_id, group in df.groupby("panel_id"):
            result[str(panel_id)] = group[["date", "y_pred", "split"]].to_dict("records")
    except Exception:
        return {}
    else:
        return result


async def _get_redis():
    import redis.asyncio as aioredis

    return aioredis.Redis(
        host=_redis_host, port=_redis_port, password=_redis_password, decode_responses=True
    )


class SkipModelRequest(BaseModel):
    """Запрос на пропуск текущей модели."""

    job_id: str
    model_name: str


@router.post("/{project_id}/skip_model")
async def skip_model(
    project_id: uuid.UUID,
    body: SkipModelRequest,
) -> dict[str, str]:
    """Устанавливает флаг отмены для текущей обучаемой модели."""
    redis = await _get_redis()
    try:
        key = f"cancel:automl:{body.job_id}:{body.model_name}"
        await redis.set(key, "1", ex=300)
        return {"status": "ok"}
    finally:
        await redis.aclose()


class CatBoostRunParams(BaseModel):
    """Параметры CatBoost для запуска AutoML."""

    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 6


class FeatureParams(BaseModel):
    """Параметры расширенных признаков временного ряда."""

    use_trend: bool = False
    trend_window: int = 6
    use_cdf: bool = False
    cdf_decay: float = 0.9
    use_mstl_seasonal: bool = False


class ChronosRunParams(BaseModel):
    """Параметры Chronos-2 для запуска AutoML."""

    context_length: int | None = None
    cross_learning: bool = False
    batch_size: int = 256


class TS2VecRunParams(BaseModel):
    """Параметры TS2Vec для запуска AutoML."""

    output_dims: int = 320
    hidden_dims: int = 64
    depth: int = 10
    n_epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 16


class PatchTSTRunParams(BaseModel):
    """Параметры PatchTST для запуска AutoML."""

    input_size: int = 24
    max_steps: int = 200
    hidden_size: int = 64
    n_heads: int = 4


class AutoMLRunConfig(BaseModel):
    """Конфигурация запуска AutoML."""

    models: list[str] = ["seasonal_naive", "catboost"]
    selection_metric: str = "mape"
    use_hyperopt: bool = False
    n_trials: int = 30
    hyperopt_timeout: int | None = None
    freq: str | None = None
    catboost_params: CatBoostRunParams = CatBoostRunParams()
    chronos_params: ChronosRunParams = ChronosRunParams()
    ts2vec_params: TS2VecRunParams = TS2VecRunParams()
    patchtst_params: PatchTSTRunParams = PatchTSTRunParams()
    autoarima_approximation: bool = True
    feature_params: FeatureParams = FeatureParams()
    hyperopt_ranges: dict | None = None


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
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "split" in j.result
        ),
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
        config.freq,
        config.n_trials,
        config.hyperopt_timeout,
        config.catboost_params.model_dump(),
        config.autoarima_approximation,
        config.feature_params.model_dump(),
        config.chronos_params.model_dump(),
        config.ts2vec_params.model_dump(),
        config.patchtst_params.model_dump(),
        config.hyperopt_ranges,
    )

    return _to_job_schema(job)


@router.get("/{project_id}/automl_result")
async def get_automl_result(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Возвращает агрегированный результат проекта (split + automl + clustering)."""
    db_result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = db_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    result: dict[str, Any] = {}
    for j in sorted(project.jobs, key=lambda j: j.created_at):
        if j.status == "done" and j.result:
            result.update(j.result)
    return {"project_id": str(project_id), "name": project.name, "result": result}


@router.get("/{project_id}/automl_predictions")
async def get_automl_predictions(
    project_id: uuid.UUID,
    panel_ids: str,
    models: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Возвращает предсказания моделей для набора панелей.

    panel_ids — через запятую, models — через запятую (все если не указано).
    Ответ: {model: {panel_id: [{date, y_pred, split}, ...]}}
    """
    db_result = await db.execute(
        select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
    )
    project = db_result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    automl_job = next(
        (
            j
            for j in sorted(project.jobs, key=lambda j: j.created_at, reverse=True)
            if j.status == "done" and j.result and "automl" in j.result
        ),
        None,
    )
    if automl_job is None:
        raise HTTPException(status_code=404, detail="Нет результатов AutoML")

    model_results = automl_job.result["automl"]["model_results"]
    requested_models = set(models.split(",")) if models else {mr["name"] for mr in model_results}
    requested_panel_ids = set(panel_ids.split(","))

    loop = asyncio.get_running_loop()
    response: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for mr in model_results:
        if mr["name"] not in requested_models:
            continue
        pred_key = mr.get("predictions_key")
        if not pred_key:
            continue
        panel_preds = await loop.run_in_executor(
            None, _load_model_predictions, pred_key, requested_panel_ids
        )
        response[mr["name"]] = panel_preds

    return response


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
