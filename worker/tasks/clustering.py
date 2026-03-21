import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.clustering import (
    cluster_panels,
    cluster_panels_auto,
    compute_cluster_mean_ts,
    compute_umap_embedding,
    extract_panel_features,
)

logger = logging.getLogger(__name__)

_user = os.getenv("POSTGRES_USER", "sales_ts_prediction")
_password = os.getenv("POSTGRES_PASSWORD", "sales_ts_prediction")
_host = os.getenv("POSTGRES_HOST", "localhost")
_port = os.getenv("POSTGRES_PORT", "5432")
_db = os.getenv("POSTGRES_DB", "sales_ts_prediction")
_SYNC_DB_URL = f"postgresql+psycopg2://{_user}:{_password}@{_host}:{_port}/{_db}"

_minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
_minio_access = os.getenv("MINIO_ROOT_USER", "sales_ts_prediction")
_minio_secret = os.getenv("MINIO_ROOT_PASSWORD", "sales_ts_prediction")
_minio_bucket = os.getenv("MINIO_BUCKET", "salecast")


def _get_engine():
    return create_engine(_SYNC_DB_URL)


def _get_s3():
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=_minio_endpoint,
        aws_access_key_id=_minio_access,
        aws_secret_access_key=_minio_secret,
        config=Config(signature_version="s3v4"),
    )


def _load_csv(key: str) -> pd.DataFrame:
    client = _get_s3()
    resp = client.get_object(Bucket=_minio_bucket, Key=key)
    return pd.read_csv(io.BytesIO(resp["Body"].read()))


def _upload_csv(key: str, df: pd.DataFrame) -> None:
    client = _get_s3()
    buf = df.to_csv(index=False).encode()
    client.put_object(Bucket=_minio_bucket, Key=key, Body=buf, ContentLength=len(buf))


def _add_step(session: Session, job, name: str, message: str) -> None:
    from api.models import Job
    steps = list(job.steps)
    steps.append({"name": name, "message": message, "timestamp": datetime.now(timezone.utc).isoformat()})
    session.execute(
        Job.__table__.update().where(Job.__table__.c.id == job.id).values(steps=steps, status="running")
    )
    session.commit()
    job.steps = steps


@celery.task(bind=True, name="worker.tasks.clustering.run_clustering")
def run_clustering(
    self,
    job_id: str,
    project_id: str,
    preprocessing_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    n_clusters: int = 5,
    method: str = "kmeans",
    use_mstl: bool = False,
    feature_mode: str = "all",
    freq: str = "MS",
) -> dict:
    """Кластеризует панели по TS-признакам, сохраняет результаты в MinIO."""
    from api.models import Job

    engine = _get_engine()

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        prep_job = session.execute(select(Job).where(Job.id == preprocessing_job_id)).scalar_one()

        try:
            _add_step(session, job, "loading", "Загрузка данных из хранилища")
            split_info = prep_job.result["split"]
            train_df = _load_csv(split_info["train_key"])
            train_df[date_col] = pd.to_datetime(train_df[date_col])

            if feature_mode == "seasonal":
                _add_step(session, job, "features", "MSTL-декомпозиция → сезонные вектора")
                from src.mstl_features import extract_seasonal_vectors
                raw_features = extract_seasonal_vectors(
                    train_df, panel_col, value_col, freq=freq,
                )
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled = scaler.fit_transform(raw_features.values)
                features_df = pd.DataFrame(
                    scaled, index=raw_features.index, columns=raw_features.columns,
                )
            else:
                feat_msg = "Извлечение признаков TS"
                if use_mstl:
                    feat_msg += " + MSTL"
                _add_step(session, job, "features", feat_msg)
                features_df = extract_panel_features(
                    train_df, panel_col, value_col,
                    use_mstl=use_mstl, freq=freq,
                )

            silhouette_scores: dict[str, float] | None = None
            best_k: int | None = None

            if method == "kmeans_auto":
                _add_step(
                    session, job, "clustering",
                    f"Кластеризация (KMeans auto, k от 2 до {n_clusters})",
                )
                labels, raw_scores, best_k = cluster_panels_auto(
                    features_df, max_k=n_clusters,
                )
                silhouette_scores = {str(k): v for k, v in raw_scores.items()}
            else:
                _add_step(
                    session, job, "clustering",
                    f"Кластеризация ({method}, n={n_clusters})",
                )
                labels = cluster_panels(features_df, n_clusters=n_clusters, method=method)

            n_actual = int(labels[labels >= 0].nunique())
            n_outliers = int((labels == -1).sum())

            _add_step(session, job, "umap", "Вычисление UMAP-проекции")
            embedding = compute_umap_embedding(features_df)

            _add_step(session, job, "mean_ts", "Расчёт средних рядов по кластерам")
            mean_ts_df = compute_cluster_mean_ts(train_df, panel_col, date_col, value_col, labels)

            _add_step(session, job, "saving", "Сохранение результатов")

            labels_key = f"projects/{project_id}/cluster_labels.csv"
            umap_key = f"projects/{project_id}/cluster_umap.csv"
            mean_ts_key = f"projects/{project_id}/cluster_mean_ts.csv"

            labels_df = labels.reset_index()
            labels_df.columns = [panel_col, "cluster_id"]
            _upload_csv(labels_key, labels_df)

            umap_df = pd.DataFrame(
                {"x": embedding[:, 0], "y": embedding[:, 1], "cluster_id": labels.values},
                index=features_df.index,
            ).reset_index().rename(columns={"index": panel_col})
            _upload_csv(umap_key, umap_df)

            _upload_csv(mean_ts_key, mean_ts_df)

            clustering_meta: dict = {
                "n_clusters": n_actual,
                "method": method,
                "n_panels": len(labels),
                "n_outliers": n_outliers,
                "labels_key": labels_key,
                "umap_key": umap_key,
                "mean_ts_key": mean_ts_key,
            }
            if silhouette_scores is not None:
                clustering_meta["silhouette_scores"] = silhouette_scores
                clustering_meta["best_k"] = best_k

            result_data = {**prep_job.result, "clustering": clustering_meta}

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            logger.info("Кластеризация завершена: %d кластеров, %d панелей", n_actual, len(labels))

        except Exception:
            logger.exception("Кластеризация job %s завершилась с ошибкой", job_id)
            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="failed", completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            raise
        else:
            return result_data
