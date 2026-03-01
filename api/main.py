from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.database import Base, engine
from api.routers import automl, jobs, panels, projects
from api.storage import ensure_bucket


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Инициализирует БД и MinIO бакет при старте приложения."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_bucket()
    yield


app = FastAPI(title="Salecast API", lifespan=lifespan)

app.include_router(projects.router)
app.include_router(jobs.router)
app.include_router(panels.router)
app.include_router(automl.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверяет доступность API."""
    return {"status": "ok"}
