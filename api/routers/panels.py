import io
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import Project
from api.storage import download_file

router = APIRouter(prefix="/projects", tags=["panels"])


@router.get("/{project_id}/preview")
async def get_project_preview(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Возвращает статистику по сырым панелям без запуска обработки."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    csv_bytes = await download_file(project.csv_key)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df[project.date_col] = pd.to_datetime(df[project.date_col])
    df[project.panel_col] = df[project.panel_col].astype(str)

    panels = (
        df.groupby(project.panel_col)
        .agg(
            rows=(project.value_col, "count"),
            date_min=(project.date_col, "min"),
            date_max=(project.date_col, "max"),
        )
        .reset_index()
        .rename(columns={project.panel_col: "panel_id"})
    )
    panels["date_min"] = panels["date_min"].dt.strftime("%Y-%m-%d")
    panels["date_max"] = panels["date_max"].dt.strftime("%Y-%m-%d")

    return {
        "panel_count": len(panels),
        "row_count": len(df),
        "date_min": df[project.date_col].min().strftime("%Y-%m-%d"),
        "date_max": df[project.date_col].max().strftime("%Y-%m-%d"),
        "panels": panels.to_dict(orient="records"),
    }


@router.get("/{project_id}/panels")
async def get_panels_data(
    project_id: uuid.UUID,
    ids: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Возвращает временные ряды для указанных панелей (ids через запятую)."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")

    panel_ids = [pid.strip() for pid in ids.split(",") if pid.strip()]

    csv_bytes = await download_file(project.csv_key)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df[project.date_col] = pd.to_datetime(df[project.date_col])
    df[project.panel_col] = df[project.panel_col].astype(str)

    filtered = df[df[project.panel_col].isin(panel_ids)].sort_values(
        [project.panel_col, project.date_col]
    )

    panels = []
    for panel_id, group in filtered.groupby(project.panel_col):
        panels.append({
            "panel_id": panel_id,
            "dates": group[project.date_col].dt.strftime("%Y-%m-%d").tolist(),
            "values": group[project.value_col].tolist(),
        })
    return panels
