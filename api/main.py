from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Создаёт таблицы при старте приложения."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Sales TS Prediction API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверяет доступность API."""
    return {"status": "ok"}
