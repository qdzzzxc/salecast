# Salecast

Система прогнозирования временных рядов продаж на основе методов машинного обучения.

Выпускная квалификационная работа. Тема: **Разработка системы прогнозирования временных рядов продаж на основе методов машинного обучения**.
Финансовый университет при Правительстве РФ, направление «Прикладная математика и информатика», группа ПМ22-1.

---

## Что делает система

Пользователь загружает CSV с историей продаж по артикулам (товарным позициям). Система автоматически:

1. **Оценивает качество данных** — фильтрует ряды слишком короткие, нулевые или аномальные
2. **Кластеризует ряды** — UMAP + HDBSCAN или TS2Vec эмбеддинги, кластеры используются как признак для моделей
3. **Запускает AutoML** — обучает набор моделей (от бейзлайнов до нейросетевых), сравнивает их по метрикам и выбирает лучшую
4. **Делает кросс-валидацию** — rolling / expanding window на исторических данных
5. **Строит ансамбли** — взвешенная комбинация лучших моделей
6. **Прогнозирует** — выдаёт предсказания на заданный горизонт с возможностью экспорта в CSV

Все тяжёлые вычисления асинхронные — задачи выполняются в фоне через Celery, интерфейс отображает прогресс в реальном времени через Redis Streams.

---

## Архитектура

```
┌─────────────┐     HTTP      ┌─────────────┐     Celery tasks     ┌─────────────┐
│  Streamlit  │ ────────────▶ │   FastAPI   │ ──────────────────▶  │   Worker    │
│  (app/)     │               │   (api/)    │                       │ (worker/)   │
└─────────────┘               └──────┬──────┘                       └──────┬──────┘
                                     │                                      │
                              ┌──────▼──────┐                       ┌──────▼──────┐
                              │ PostgreSQL  │                       │    Redis    │
                              │  (jobs,     │                       │  (broker +  │
                              │  projects)  │                       │   results)  │
                              └─────────────┘                       └─────────────┘
                                                                           │
                                                                    ┌──────▼──────┐
                                                                    │    MinIO    │
                                                                    │  (CSV files,│
                                                                    │  predictions│
                                                                    │  forecasts) │
                                                                    └─────────────┘
```

| Компонент | Технология | Назначение |
|---|---|---|
| Web-интерфейс | Streamlit | Upload, Quality, Clustering, AutoML, CV, Ensemble, Forecast |
| API | FastAPI + asyncpg | REST API, управление проектами и задачами |
| Очередь задач | Celery + Redis | Асинхронное выполнение тяжёлых вычислений |
| Прогресс задач | Redis Streams | Стрим прогресса AutoML/CV/Forecast в UI |
| База данных | PostgreSQL | Проекты, задачи (jobs), статусы |
| Файловое хранилище | MinIO (S3-совместимый) | Сырые данные, сплиты, предсказания, прогнозы, артефакты моделей |
| Мониторинг задач | Flower | Web-UI для Celery |

---

## Модели

| Модель | Тип | Описание |
|---|---|---|
| **Seasonal Naive** | Baseline | Прогноз = значение той же сезонной точки сезон назад |
| **AutoARIMA** | Статистическая | StatsForecast, автоматический подбор порядков |
| **AutoETS** | Статистическая | Error-Trend-Seasonal с автовыбором компонент |
| **AutoTheta** | Статистическая | Theta-метод с автоматической декомпозицией |
| **MSTL** | Статистическая | Multiple Seasonal-Trend декомпозиция + AutoETS на остатке |
| **CatBoost (global)** | Gradient Boosting | Одна модель на все панели; лаговые/скользящие/EMA/календарные + trend/CDF (Пасканов); Optuna hyperopt |
| **CatBoost (per-panel)** | Gradient Boosting | Отдельная модель на каждый ряд |
| **CatBoost Clustered** | Gradient Boosting | Отдельная модель на каждый кластер рядов (UMAP+HDBSCAN) |
| **TS2Vec → MLP** | Self-supervised + DL | Contrastive энкодер временных рядов + MLP-голова на эмбеддингах |
| **TS2Vec Clustered** | Self-supervised + GB | TS2Vec эмбеддинги + кластеризация + CatBoost |
| **PatchTST** | Transformer | Патч-токенизация, neuralforecast |
| **Chronos** | Foundation model | Pretrained LLM-подобная модель для временных рядов (Amazon) |
| **Ensemble** | Мета | Взвешенная комбинация топ-N моделей |

ModelSelector выбирает лучшую модель по метрике на валидационной выборке (по умолчанию — MAPE).
Для CatBoost доступна оптимизация гиперпараметров через **Optuna** с настраиваемым числом trials и диапазонами.

---

## Пайплайн

```
CSV с продажами
      │
      ▼
[1. Upload] → маппинг колонок (ID, дата, значение)
      │
      ▼
[2. Preprocessing] → агрегация дубликатов, заполнение пропусков, временной сплит (train / val / test)
      │
      ▼
[3. Quality] → фильтрация рядов:
               • минимальная длина
               • доля нулей
               • edge-zero фильтр
      │
      ▼
[4. Clustering] (опционально) → UMAP → HDBSCAN или TS2Vec эмбеддинги,
                                кластер используется как признак в моделях
      │
      ▼
[5. AutoML] → для каждой модели: fit на train → evaluate на val/test
              → сохранение предсказаний в MinIO
              → ModelSelector выбирает лучшую по val-метрике
      │
      ▼
[6. Cross-validation] (опционально) → rolling/expanding window поверх выбранных моделей,
                                       с возможностью hyperopt параметров
      │
      ▼
[7. Ensemble] (опционально) → взвешенная комбинация топ-N моделей
      │
      ▼
[8. Forecast] → прогноз на N точек вперёд, сохранение forecast.csv в MinIO
```

---

## Метрики качества

Для каждого ряда и в агрегате:

| Метрика | Описание |
|---|---|
| **MAPE** | Средняя абсолютная процентная ошибка |
| **RMSE** | Среднеквадратичная ошибка |
| **MAE** | Средняя абсолютная ошибка |
| **R²** | Коэффициент детерминации |
| **nRMSE** | RMSE нормированная на std ряда |

---

## Быстрый старт

```bash
# Клонировать репозиторий
git clone <repo>
cd salecast

# Запустить все сервисы
docker compose up -d

# Открыть интерфейс
open http://localhost:8501
```

Веб-интерфейсы сервисов:

| Сервис | URL |
|---|---|
| Streamlit (приложение) | http://localhost:8501 |
| FastAPI Swagger | http://localhost:8000/docs |
| Flower (Celery) | http://localhost:5555 |
| MinIO Console | http://localhost:9001 |
| RedisInsight | http://localhost:5540 |
| Adminer (PostgreSQL) | http://localhost:8080 |

---

## Структура проекта

```
salecast/
├── src/                         # Библиотека алгоритмов
│   ├── automl/                  # AutoML: ModelSelector, Optuna hyperopt, конфиги
│   │   ├── models/              # CatBoost(+clustered), TS2Vec(+clustered), Chronos,
│   │   │                        # PatchTST, SeasonalNaive, StatsForecast
│   │   └── ts2vec/              # Реализация TS2Vec энкодера
│   ├── catboost_utilities/      # Обучение и оценка CatBoost
│   ├── seasonal_naive_utilities/ # Seasonal Naive модель
│   ├── clustering.py            # UMAP + HDBSCAN кластеризация рядов
│   ├── ensemble.py              # Взвешенные ансамбли моделей
│   ├── classifical_features.py  # Lag / rolling / EMA / calendar / trend / CDF признаки
│   ├── mstl_features.py         # MSTL-декомпозиция как признак
│   ├── ts_features.py           # TS2Vec эмбеддинги как признак
│   ├── filtration.py            # Фильтрация временных рядов
│   ├── data_processing.py       # Скейлинг, агрегация, expand, сплиты
│   ├── model_selection.py       # Временные сплиты (ratio, date, tail, CV)
│   ├── evaluation.py            # MAPE, RMSE, MAE, R², nRMSE
│   ├── visualization/           # Графики для отчётов и UI
│   ├── diagnostics/             # Диагностика рядов и моделей
│   └── custom_types.py          # Общие типы: Splits, ModelResult, AutoMLResult
│
├── api/                         # FastAPI приложение
│   └── routers/                 # projects, jobs, panels, automl, clustering,
│                                # cross_validation, ensemble, forecast
│
├── worker/                      # Celery задачи
│   └── tasks/                   # automl, clustering, cross_validation, ensemble, forecast
│
├── app/                         # Streamlit интерфейс
│   └── views/                   # upload, quality, clustering, automl,
│                                # cross_validation, ensemble, forecast
│
├── notebooks/                   # Исследовательские ноутбуки (EDA, кластеризация, эксперименты)
├── tests/                       # 390 тестов, 84% покрытие
├── docker-compose.yml           # Полный стек: api, worker, streamlit, postgres, redis, minio
└── pyproject.toml
```

---

## Разработка

```bash
# Зависимости
uv sync

# Линтер
uv run ruff check src/
uv run ruff check --fix src/
uv run ruff format src/

# Тесты
uv run pytest tests/

# E2E тест AutoML на реальных данных
uv run python tests/scripts/test_automl_e2e.py
```

**Стек:** Python 3.10, [uv](https://github.com/astral-sh/uv), Docker, ruff, pytest.
**Основные библиотеки:** pandas, catboost, statsforecast, neuralforecast, hdbscan, umap-learn, torch (TS2Vec), streamlit, fastapi, celery, sqlalchemy, pydantic.
