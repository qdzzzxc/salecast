import io
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.configs.settings import ColumnConfig, Settings

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


def _next_dates(dates: pd.Series, n: int) -> list[pd.Timestamp]:
    """Генерирует n следующих дат на основе частоты ряда."""
    sorted_dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    freq = pd.infer_freq(sorted_dates)
    if freq:
        return pd.date_range(sorted_dates.iloc[-1], periods=n + 1, freq=freq)[1:].tolist()
    delta = sorted_dates.diff().dropna().median()
    return [sorted_dates.iloc[-1] + delta * (i + 1) for i in range(n)]


def _infer_freq(dates: pd.Series) -> str:
    sorted_dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    return pd.infer_freq(sorted_dates) or "MS"


def _forecast_seasonal_naive(
    full_df: pd.DataFrame, panel_col: str, date_col: str, value_col: str, horizon: int
) -> pd.DataFrame:
    from src.seasonal_naive_utilities.seasonal_naive_model import SeasonalNaiveModel

    model = SeasonalNaiveModel()
    model.fit(full_df, panel_col, value_col)

    rows = []
    for panel_id, group in full_df.groupby(panel_col):
        future_dates = _next_dates(group[date_col], horizon)
        rows.extend({panel_col: panel_id, date_col: d, value_col: np.nan} for d in future_dates)

    future_df = pd.DataFrame(rows)
    preds = model.predict(future_df, panel_col, value_col, is_train=False)
    future_df["forecast"] = np.maximum(preds, 0)
    future_df["panel_id"] = future_df[panel_col].astype(str)
    future_df["date"] = pd.to_datetime(future_df[date_col]).dt.strftime("%Y-%m-%d")
    return future_df[["panel_id", "date", "forecast"]]


def _forecast_statsmodels(
    full_df: pd.DataFrame,
    model_type: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    horizon: int,
) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    _cls_map = {"autoarima": AutoARIMA, "autoets": AutoETS, "autotheta": AutoTheta}
    freq = _infer_freq(full_df[date_col])

    sf_df = full_df.rename(columns={panel_col: "unique_id", date_col: "ds", value_col: "y"})
    sf_df["unique_id"] = sf_df["unique_id"].astype(str)
    sf_df["ds"] = pd.to_datetime(sf_df["ds"])

    sf = StatsForecast(models=[_cls_map[model_type](season_length=12)], freq=freq, verbose=False)
    sf.fit(sf_df)
    forecast = sf.predict(h=horizon).reset_index()

    pred_col = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]
    result = pd.DataFrame({
        "panel_id": forecast["unique_id"].astype(str),
        "date": pd.to_datetime(forecast["ds"]).dt.strftime("%Y-%m-%d"),
        "forecast": forecast[pred_col].clip(lower=0).astype(float),
    })
    return result


def _forecast_catboost(
    full_df: pd.DataFrame,
    panel_col: str,
    date_col: str,
    value_col: str,
    horizon: int,
    settings: Settings,
) -> pd.DataFrame:
    from src.catboost_utilities.train import train_catboost
    from src.classifical_features import build_monthly_features
    from src.custom_types import CatBoostParameters

    # Отключаем скейлинг для простоты — прогноз в исходном масштабе
    forecast_settings = settings.model_copy(
        update={"downstream": settings.downstream.model_copy(update={"scale": False})}
    )
    target = forecast_settings.columns.main_target
    id_col = forecast_settings.columns.id
    date = forecast_settings.columns.date

    features_df = build_monthly_features(full_df, forecast_settings, disable_tqdm=True)
    model = train_catboost(features_df, None, CatBoostParameters(), forecast_settings)

    drop_cols = {target, id_col, date}
    feature_cols = [c for c in features_df.columns if c not in drop_cols]

    running_df = full_df.copy()
    all_preds = []

    for _ in range(horizon):
        next_rows = []
        for pid, group in running_df.groupby(panel_col):
            nd = _next_dates(group[date_col], 1)[0]
            next_rows.append({panel_col: pid, date_col: nd, value_col: 0.0})

        next_df = pd.DataFrame(next_rows)
        extended = pd.concat([running_df, next_df], ignore_index=True)
        feat = build_monthly_features(extended, forecast_settings, disable_tqdm=True)

        future_feat = feat.tail(len(next_rows))[feature_cols].reset_index(drop=True)
        preds = model.predict(future_feat)
        preds = np.maximum(preds, 0)

        next_df = next_df.reset_index(drop=True)
        next_df[value_col] = preds

        for i, row in next_df.iterrows():
            all_preds.append({
                "panel_id": str(row[panel_col]),
                "date": pd.Timestamp(row[date_col]).strftime("%Y-%m-%d"),
                "forecast": float(preds[i]),
            })

        running_df = pd.concat([running_df, next_df], ignore_index=True)

    return pd.DataFrame(all_preds)


@celery.task(bind=True, name="worker.tasks.forecast.run_forecast")
def run_forecast(
    self,
    job_id: str,
    project_id: str,
    automl_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    model_name: str,
    horizon: int,
    panel_ids: list[str],
) -> dict:
    """Обучает выбранную модель на всех данных и строит прогноз на horizon точек вперёд."""
    from api.models import Job

    engine = _get_engine()

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        automl_job = session.execute(select(Job).where(Job.id == automl_job_id)).scalar_one()

        try:
            _add_step(session, job, "loading", "Загрузка данных")
            split_info = automl_job.result["split"]
            train_df = _load_csv(split_info["train_key"])
            val_df = _load_csv(split_info["val_key"])
            test_df = _load_csv(split_info["test_key"])

            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            full_df[date_col] = pd.to_datetime(full_df[date_col])
            full_df = full_df.sort_values([panel_col, date_col]).reset_index(drop=True)

            if panel_ids:
                full_df = full_df[full_df[panel_col].astype(str).isin(set(panel_ids))]

            settings = Settings().model_copy(
                update={"columns": ColumnConfig(id=panel_col, date=date_col, main_target=value_col)}
            )

            _add_step(session, job, "forecasting", f"Прогноз {model_name} на {horizon} точек")

            if model_name == "seasonal_naive":
                forecast_df = _forecast_seasonal_naive(full_df, panel_col, date_col, value_col, horizon)
            elif model_name in ("autoarima", "autoets", "autotheta"):
                forecast_df = _forecast_statsmodels(full_df, model_name, panel_col, date_col, value_col, horizon)
            elif model_name == "catboost":
                forecast_df = _forecast_catboost(full_df, panel_col, date_col, value_col, horizon, settings)
            else:
                raise ValueError(f"Неизвестная модель: {model_name}")

            forecast_key = f"projects/{project_id}/forecast.csv"
            _add_step(session, job, "saving", "Сохранение прогноза")
            _upload_csv(forecast_key, forecast_df)

            result_data = {
                "split": automl_job.result["split"],
                "automl": automl_job.result["automl"],
                "forecast": {
                    "model": model_name,
                    "horizon": horizon,
                    "panel_count": int(forecast_df["panel_id"].nunique()),
                    "forecast_key": forecast_key,
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            logger.info("Forecast job %s done: model=%s horizon=%d", job_id, model_name, horizon)

        except Exception:
            logger.exception("Forecast job %s failed", job_id)
            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="failed", completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            raise
        else:
            return result_data
