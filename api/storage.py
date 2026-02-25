import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aioboto3
from botocore.exceptions import ClientError

_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "sales_ts_prediction")
_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "sales_ts_prediction")
_BUCKET = os.getenv("MINIO_BUCKET", "salecast")

_session = aioboto3.Session()


@asynccontextmanager
async def _s3_client() -> AsyncIterator:
    """Контекстный менеджер для S3 клиента."""
    async with _session.client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
    ) as client:
        yield client


async def ensure_bucket() -> None:
    """Создаёт бакет если не существует."""
    async with _s3_client() as client:
        try:
            await client.head_bucket(Bucket=_BUCKET)
        except ClientError:
            await client.create_bucket(Bucket=_BUCKET)


async def upload_file(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Загружает файл в MinIO."""
    async with _s3_client() as client:
        await client.put_object(Bucket=_BUCKET, Key=key, Body=data, ContentType=content_type)


async def download_file(key: str) -> bytes:
    """Скачивает файл из MinIO."""
    async with _s3_client() as client:
        response = await client.get_object(Bucket=_BUCKET, Key=key)
        return await response["Body"].read()
