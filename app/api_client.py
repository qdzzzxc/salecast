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


async def _get_project_preview(project_id: str) -> dict[str, Any]:
    """Получает статистику по сырым панелям проекта."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{_API_URL}/projects/{project_id}/preview") as resp:
            resp.raise_for_status()
            return await resp.json()


def get_project_preview(project_id: str) -> dict[str, Any]:
    """Получает превью проекта (синхронная обёртка)."""
    return _run(_get_project_preview(project_id))


async def _run_project(project_id: str, val_periods: int, test_periods: int) -> dict[str, Any]:
    """Запускает обработку проекта через API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_API_URL}/projects/{project_id}/run",
            json={"val_periods": val_periods, "test_periods": test_periods},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


def run_project(project_id: str, val_periods: int, test_periods: int) -> dict[str, Any]:
    """Запускает обработку проекта (синхронная обёртка)."""
    return _run(_run_project(project_id, val_periods, test_periods))


async def _delete_project(project_id: str) -> None:
    """Удаляет проект через API."""
    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{_API_URL}/projects/{project_id}") as resp:
            resp.raise_for_status()


def delete_project(project_id: str) -> None:
    """Удаляет проект (синхронная обёртка)."""
    _run(_delete_project(project_id))


async def _run_automl(project_id: str, models: list[str], selection_metric: str, use_hyperopt: bool) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_API_URL}/projects/{project_id}/run_automl",
            json={"models": models, "selection_metric": selection_metric, "use_hyperopt": use_hyperopt},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


def run_automl(project_id: str, models: list[str], selection_metric: str, use_hyperopt: bool) -> dict[str, Any]:
    return _run(_run_automl(project_id, models, selection_metric, use_hyperopt))


async def _get_automl_progress(project_id: str, job_id: str) -> list[dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{_API_URL}/projects/{project_id}/automl_progress/{job_id}") as resp:
            resp.raise_for_status()
            return await resp.json()


def get_automl_progress(project_id: str, job_id: str) -> list[dict[str, Any]]:
    return _run(_get_automl_progress(project_id, job_id))


async def _get_panels_data(project_id: str, ids: list[str]) -> list[dict]:
    """Загружает данные панелей из API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{_API_URL}/projects/{project_id}/panels",
            params={"ids": ",".join(ids)},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


def get_panels_data(project_id: str, ids: list[str]) -> list[dict]:
    """Загружает данные панелей (синхронная обёртка)."""
    return _run(_get_panels_data(project_id, ids))


async def _get_automl_predictions(
    project_id: str,
    panel_ids: list[str],
    models: list[str] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    params: dict = {"panel_ids": ",".join(panel_ids)}
    if models:
        params["models"] = ",".join(models)
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{_API_URL}/projects/{project_id}/automl_predictions",
            params=params,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


def get_automl_predictions(
    project_id: str,
    panel_ids: list[str],
    models: list[str] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """Возвращает предсказания моделей для набора панелей (синхронная обёртка)."""
    return _run(_get_automl_predictions(project_id, panel_ids, models))
