import os

from celery import Celery

redis_password = os.getenv("REDIS_PASSWORD", "sales_ts_prediction")
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", "6379")

_redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"

celery = Celery(
    "sales_ts_prediction",
    broker=_redis_url,
    backend=_redis_url,
    include=["worker.tasks.automl"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
