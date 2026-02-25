import logging

from celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="worker.tasks.automl.run_automl")
def run_automl(self, job_id: str, project_id: str) -> dict:
    """Запускает AutoML пайплайн для проекта."""
    logger.info("Запуск AutoML: job_id=%s project_id=%s", job_id, project_id)
    self.update_state(state="STARTED", meta={"step": "init", "message": "Инициализация"})
    return {"job_id": job_id, "status": "done"}
