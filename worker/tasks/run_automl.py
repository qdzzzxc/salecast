import concurrent.futures
import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from celery_app import celery
from src.automl.base import ModelCancelledError
from src.automl.hyperopt import tune_catboost
from src.automl.models import (
    CatBoostForecastModel,
    CatBoostPerPanelForecastModel,
    SeasonalNaiveForecastModel,
    StatsForecastModel,
)
from src.automl.models.catboost_clustered_model import CatBoostClusteredForecastModel
from src.automl.models.chronos_model import ChronosForecastModel, ChronosParameters
from src.automl.models.ts2vec_clustered_model import TS2VecClusteredForecastModel
from src.automl.models.ts2vec_model import TS2VecForecastModel, TS2VecParameters
from src.automl.selector import _get_metric_value
from src.automl.ts_utils import get_downstream_lags, infer_ts_config, ts_config_from_freq
from src.configs.settings import ColumnConfig, Settings
from src.custom_types import CatBoostParameters, ModelType, Splits

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

_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port = int(os.getenv("REDIS_PORT", "6379"))
_redis_password = os.getenv("REDIS_PASSWORD", "")


def _get_engine():
    return create_engine(_SYNC_DB_URL)


def _get_redis():
    import redis

    return redis.Redis(
        host=_redis_host, port=_redis_port, password=_redis_password, decode_responses=True
    )


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
    """Загружает DataFrame как CSV в MinIO."""
    client = _get_s3()
    buf = df.to_csv(index=False).encode()
    client.put_object(Bucket=_minio_bucket, Key=key, Body=buf, ContentLength=len(buf))


def _build_predictions_df(result, splits: Splits, panel_col: str, date_col: str) -> pd.DataFrame:
    """Строит DataFrame с предсказаниями модели (val + test) по всем панелям."""
    split_dfs = {}
    if splits.val is not None:
        split_dfs["val"] = splits.val
    if splits.test is not None:
        split_dfs["test"] = splits.test

    rows = []
    for split_eval in result.evaluation.splits:
        split_name = split_eval.split_name
        split_df = split_dfs.get(split_name)
        if split_df is None:
            continue
        for pm in split_eval.panel_metrics:
            panel_id = str(pm.panel_id)
            panel_df = split_df[split_df[panel_col] == panel_id].sort_values(date_col)
            dates = pd.to_datetime(panel_df[date_col]).dt.strftime("%Y-%m-%d").tolist()
            y_pred = list(pm.y_pred) if hasattr(pm.y_pred, "__iter__") else []
            if len(dates) == len(y_pred):
                for d, p in zip(dates, y_pred):
                    rows.append(
                        {"panel_id": panel_id, "date": d, "split": split_name, "y_pred": float(p)}
                    )
    return pd.DataFrame(rows)


def _add_step(session: Session, job, name: str, message: str) -> None:
    from api.models import Job

    steps = list(job.steps)
    steps.append(
        {"name": name, "message": message, "timestamp": datetime.now(timezone.utc).isoformat()}
    )
    session.execute(
        Job.__table__.update()
        .where(Job.__table__.c.id == job.id)
        .values(steps=steps, status="running")
    )
    session.commit()
    job.steps = steps


def _build_model(
    model_type: ModelType,
    catboost_params: dict | None = None,
    autoarima_approximation: bool = True,
    cluster_labels: dict[str, int] | None = None,
    chronos_params: dict | None = None,
    ts2vec_params: dict | None = None,
):
    """Создаёт модель заданного типа с переданными параметрами."""
    if model_type == ModelType.seasonal_naive:
        return SeasonalNaiveForecastModel()
    if model_type == ModelType.catboost:
        params = CatBoostParameters(**(catboost_params or {}))
        return CatBoostForecastModel(params=params)
    if model_type == ModelType.catboost_per_panel:
        params = CatBoostParameters(**(catboost_params or {}))
        return CatBoostPerPanelForecastModel(params=params)
    if model_type == ModelType.catboost_clustered:
        params = CatBoostParameters(**(catboost_params or {}))
        return CatBoostClusteredForecastModel(cluster_labels=cluster_labels or {}, params=params)
    if model_type == ModelType.autoarima:
        return StatsForecastModel(model_type=model_type, approximation=autoarima_approximation)
    if model_type == ModelType.chronos:
        return ChronosForecastModel(params=ChronosParameters(**(chronos_params or {})))
    if model_type == ModelType.ts2vec:
        return TS2VecForecastModel(params=TS2VecParameters(**(ts2vec_params or {})))
    if model_type == ModelType.ts2vec_clustered:
        return TS2VecClusteredForecastModel(
            cluster_labels=cluster_labels or {},
            params=TS2VecParameters(**(ts2vec_params or {})),
        )
    return StatsForecastModel(model_type=model_type)


def _extract_metric(result, metric: str, split_name: str) -> float:
    """Извлекает значение метрики из ModelResult."""
    for split_eval in result.evaluation.splits:
        if split_eval.split_name == split_name:
            return float(getattr(split_eval.overall_metrics, metric, float("inf")))
    return float("inf")


def _extract_panel_metrics(result, metric: str) -> list[dict]:
    """Возвращает per-panel метрики val и test."""
    val_by_panel, test_by_panel = {}, {}
    for split_eval in result.evaluation.splits:
        if split_eval.split_name == "val":
            for pm in split_eval.panel_metrics:
                val_by_panel[str(pm.panel_id)] = float(getattr(pm.metrics, metric, float("inf")))
        elif split_eval.split_name == "test":
            for pm in split_eval.panel_metrics:
                test_by_panel[str(pm.panel_id)] = float(getattr(pm.metrics, metric, float("inf")))
    panel_ids = set(val_by_panel) | set(test_by_panel)
    return [
        {
            "panel_id": pid,
            "val": val_by_panel.get(pid, None),
            "test": test_by_panel.get(pid, None),
        }
        for pid in sorted(panel_ids)
    ]


@celery.task(bind=True, name="worker.tasks.run_automl.run_automl")
def run_automl(
    self,
    job_id: str,
    project_id: str,
    preprocessing_job_id: str,
    panel_col: str,
    date_col: str,
    value_col: str,
    models: list[str],
    selection_metric: str,
    use_hyperopt: bool,
    freq: str | None = None,
    n_trials: int = 30,
    hyperopt_timeout: int | None = None,
    catboost_params: dict | None = None,
    autoarima_approximation: bool = True,
    feature_params: dict | None = None,
    chronos_params: dict | None = None,
    ts2vec_params: dict | None = None,
) -> dict:
    """Запускает AutoML: обучает модели на отфильтрованных панелях, выбирает лучшую."""
    from api.models import Job

    engine = _get_engine()
    redis_client = _get_redis()
    stream_key = f"stream:automl:{job_id}"

    with Session(engine) as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        prep_job = session.execute(select(Job).where(Job.id == preprocessing_job_id)).scalar_one()

        try:
            _add_step(session, job, "loading", "Загрузка train/val/test из хранилища")
            split_info = prep_job.result["split"]
            train_df = _load_csv(split_info["train_key"])
            val_df = _load_csv(split_info["val_key"])
            test_df = _load_csv(split_info["test_key"])

            # Фильтруем только green+yellow панели
            diag_panels = prep_job.result["diagnostics"]["panels"]
            good_panels = {
                str(p["panel_id"])
                for p in diag_panels
                if p.get("overall_status") in ("green", "yellow")
            }
            train_df[panel_col] = train_df[panel_col].astype(str)
            val_df[panel_col] = val_df[panel_col].astype(str)
            test_df[panel_col] = test_df[panel_col].astype(str)
            train_df = train_df[train_df[panel_col].isin(good_panels)]
            val_df = val_df[val_df[panel_col].isin(good_panels)]
            test_df = test_df[test_df[panel_col].isin(good_panels)]
            logger.info("AutoML: %d панелей (green+yellow)", len(good_panels))

            train_df[date_col] = pd.to_datetime(train_df[date_col])
            val_df[date_col] = pd.to_datetime(val_df[date_col])
            test_df[date_col] = pd.to_datetime(test_df[date_col])

            splits = Splits(train=train_df, val=val_df, test=test_df)

            ts_config = ts_config_from_freq(freq) if freq else infer_ts_config(train_df, date_col)
            lags = get_downstream_lags(ts_config.freq)
            logger.info(
                "AutoML: freq=%s season_length=%d lags=%s (источник: %s)",
                ts_config.freq,
                ts_config.season_length,
                lags,
                "явно задан" if freq else "автоопределение",
            )
            base_settings = Settings()
            downstream_update: dict = {"lags": lags}
            if feature_params:
                downstream_update.update(feature_params)
            settings = base_settings.model_copy(
                update={
                    "columns": ColumnConfig(id=panel_col, date=date_col, main_target=value_col),
                    "ts": ts_config,
                    "downstream": base_settings.downstream.model_copy(update=downstream_update),
                }
            )

            # Загружаем cluster_labels если нужна clustered-модель
            cluster_labels: dict[str, int] | None = None
            _clustered_models = {"catboost_clustered", "ts2vec_clustered"}
            if _clustered_models & set(models):
                clustering_info = prep_job.result.get("clustering")
                if clustering_info:
                    labels_df = _load_csv(clustering_info["labels_key"])
                    cluster_labels = dict(
                        zip(labels_df.iloc[:, 0].astype(str), labels_df["cluster_id"].astype(int))
                    )
                    logger.info("Загружены cluster_labels: %d панелей", len(cluster_labels))
                else:
                    logger.warning(
                        "clustered-модель запрошена, но результатов кластеризации нет"
                    )
                    models = [m for m in models if m not in _clustered_models]

            if use_hyperopt and "catboost" in models:
                timeout_label = f", timeout {hyperopt_timeout}s" if hyperopt_timeout else ""
                _add_step(
                    session,
                    job,
                    "hyperopt",
                    f"Hyperopt CatBoost ({n_trials} trials{timeout_label})",
                )
                redis_client.xadd(stream_key, {"type": "hyperopt_start", "n_trials": str(n_trials)})
                try:
                    best_cb_params = tune_catboost(
                        splits, settings, n_trials=n_trials, timeout=hyperopt_timeout
                    )
                    catboost_params = best_cb_params.model_dump()
                    redis_client.xadd(stream_key, {"type": "hyperopt_done"})
                    logger.info("Hyperopt завершён: %s", catboost_params)
                except Exception:
                    logger.exception(
                        "Hyperopt завершился с ошибкой, используем дефолтные параметры"
                    )
                    redis_client.xadd(stream_key, {"type": "hyperopt_failed"})

            all_results = []
            for i, model_type in enumerate(models):
                _add_step(
                    session,
                    job,
                    f"train_{model_type}",
                    f"Обучение {model_type} ({i + 1}/{len(models)})",
                )
                redis_client.xadd(
                    stream_key,
                    {
                        "type": "model_start",
                        "model": model_type,
                        "n": str(i + 1),
                        "total": str(len(models)),
                    },
                )

                model = _build_model(
                    model_type,
                    catboost_params,
                    autoarima_approximation,
                    cluster_labels,
                    chronos_params,
                    ts2vec_params,
                )

                cancel_key = f"cancel:automl:{job_id}:{model_type}"

                def _progress_fn(
                    message: str, pct: float | None = None, _mt: str = model_type
                ) -> None:
                    # Парсим расширенный формат: "text||key=val||key2=val2"
                    display_msg = message
                    extra: dict[str, str] = {}
                    if "||" in message:
                        parts = message.split("||")
                        display_msg = parts[0]
                        for part in parts[1:]:
                            if "=" in part:
                                k, v = part.split("=", 1)
                                extra[k] = v

                    payload: dict = {
                        "type": "model_progress",
                        "model": _mt,
                        "message": display_msg,
                    }
                    if pct is not None:
                        payload["pct"] = f"{pct:.1f}"
                    payload.update(extra)
                    redis_client.xadd(stream_key, payload)

                def _cancel_fn(_key: str = cancel_key) -> bool:
                    return bool(redis_client.get(_key))

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            model.fit_evaluate, splits, settings, _progress_fn, _cancel_fn
                        )
                        result = future.result(timeout=300)
                except ModelCancelledError:
                    logger.info("Модель %s пропущена пользователем", model_type)
                    redis_client.xadd(stream_key, {"type": "model_skipped", "model": model_type})
                    continue
                except concurrent.futures.TimeoutError:
                    logger.error("Модель %s превысила таймаут (300s), пропускаем", model_type)
                    redis_client.xadd(stream_key, {"type": "model_timeout", "model": model_type})
                    continue
                except Exception:
                    logger.exception("Модель %s упала с ошибкой, пропускаем", model_type)
                    redis_client.xadd(stream_key, {"type": "model_skipped", "model": model_type})
                    continue
                all_results.append(result)

                # Сохраняем предсказания (val + test) в MinIO
                pred_key = f"projects/{project_id}/automl_predictions/{model_type}.csv"
                try:
                    pred_df = _build_predictions_df(result, splits, panel_col, date_col)
                    _upload_csv(pred_key, pred_df)
                except Exception:
                    logger.exception("Не удалось сохранить предсказания для модели %s", model_type)
                    pred_key = None

                val_mape = _extract_metric(result, selection_metric, "val")
                test_mape = _extract_metric(result, selection_metric, "test")
                redis_client.xadd(
                    stream_key,
                    {
                        "type": "model_done",
                        "model": model_type,
                        f"val_{selection_metric}": f"{val_mape:.4f}",
                        f"test_{selection_metric}": f"{test_mape:.4f}",
                    },
                )
                logger.info(
                    "Модель %s: val_%s=%.4f test_%s=%.4f",
                    model_type,
                    selection_metric,
                    val_mape,
                    selection_metric,
                    test_mape,
                )

            best = min(
                all_results,
                key=lambda r: _get_metric_value(r, selection_metric, "val"),
            )
            redis_client.xadd(stream_key, {"type": "best", "model": best.name})

            _add_step(session, job, "saving", "Сохранение результатов AutoML")

            pred_keys = {
                r.name: f"projects/{project_id}/automl_predictions/{r.name}.csv"
                for r in all_results
            }

            model_results = [
                {
                    "name": r.name,
                    f"val_{selection_metric}": _extract_metric(r, selection_metric, "val"),
                    f"test_{selection_metric}": _extract_metric(r, selection_metric, "test"),
                    "panel_metrics": _extract_panel_metrics(r, selection_metric),
                    "predictions_key": pred_keys.get(r.name),
                    "feature_importance": r.feature_importance,
                    "loss_history": r.loss_history,
                }
                for r in all_results
            ]

            result_data = {
                **prep_job.result,
                "automl": {
                    "models_used": models,
                    "selection_metric": selection_metric,
                    "best_model": best.name,
                    "total_panels": len(good_panels),
                    "model_results": model_results,
                    "ts": {
                        "freq": ts_config.freq,
                        "season_length": ts_config.season_length,
                        "freq_source": "manual" if freq else "auto",
                    },
                    "feature_params": feature_params or {},
                    "chronos_params": chronos_params or {},
                    "ts2vec_params": ts2vec_params or {},
                },
            }

            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="done", result=result_data, completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            redis_client.xadd(stream_key, {"type": "completed"})
            logger.info("AutoML job %s завершён, лучшая модель: %s", job_id, best.name)

        except Exception:
            logger.exception("AutoML job %s завершился с ошибкой", job_id)
            session.execute(
                Job.__table__.update()
                .where(Job.__table__.c.id == job.id)
                .values(status="failed", completed_at=datetime.now(timezone.utc))
            )
            session.commit()
            redis_client.xadd(stream_key, {"type": "failed"})
            raise
        else:
            return result_data
