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
            "dates": group[project.date_col].dt.strftime("%Y-%m").tolist(),
            "values": group[project.value_col].tolist(),
        })
    return panels
