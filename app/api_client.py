import asyncio
import os
from typing import Any

import aiohttp

_API_URL = os.getenv("API_URL", "http://localhost:8000")


def _run(coro):
    """Запускает корутину синхронно."""
    return asyncio.run(coro)


async def _create_project(
    name: str,
    file_bytes: bytes,
    filename: str,
    panel_col: str,
    date_col: str,
    value_col: str,
) -> dict[str, Any]:
    """Создаёт проект через API."""
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("file", file_bytes, filename=filename, content_type="text/csv")
        async with session.post(
            f"{_API_URL}/projects",
            data=form,
            params={"name": name, "panel_col": panel_col, "date_col": date_col, "value_col": value_col},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def _list_projects() -> list[dict[str, Any]]:
    """Получает список проектов из API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{_API_URL}/projects") as resp:
            resp.raise_for_status()
            return await resp.json()


async def _get_job(job_id: str) -> dict[str, Any]:
    """Получает статус задачи из API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{_API_URL}/jobs/{job_id}") as resp:
            resp.raise_for_status()
            return await resp.json()


def create_project(
    name: str,
    file_bytes: bytes,
    filename: str,
    panel_col: str,
    date_col: str,
    value_col: str,
) -> dict[str, Any]:
    """Создаёт проект через API (синхронная обёртка)."""
    return _run(_create_project(name, file_bytes, filename, panel_col, date_col, value_col))


def list_projects() -> list[dict[str, Any]]:
    """Получает список проектов (синхронная обёртка)."""
    return _run(_list_projects())


def get_job(job_id: str) -> dict[str, Any]:
    """Получает статус задачи (синхронная обёртка)."""
    return _run(_get_job(job_id))
