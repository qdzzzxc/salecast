# Salecast

Система прогнозирования временных рядов продаж на основе методов машинного обучения.

Дипломный проект. Тема: **Разработка системы прогнозирования временных рядов продаж на основе методов машинного обучения**.

---

## Что делает система

Пользователь загружает CSV с историей продаж по артикулам (товарным позициям). Система автоматически:

1. **Оценивает качество данных** — фильтрует ряды слишком короткие, нулевые или аномальные
2. **Запускает AutoML** — обучает несколько моделей, сравнивает их по метрикам (MAPE, RMSE, MAE, R²) и выбирает лучшую
3. **Строит прогноз** — выдаёт предсказания на заданный горизонт вперёд с возможностью экспорта в CSV

Все вычисления асинхронные — задачи выполняются в фоне, интерфейс отображает прогресс в реальном времени.

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
| Web-интерфейс | Streamlit | Upload, Quality, AutoML, Forecast |
| API | FastAPI + asyncpg | REST API, управление проектами и задачами |
| Очередь задач | Celery + Redis | Асинхронное выполнение тяжёлых вычислений |
| База данных | PostgreSQL | Проекты, задачи (jobs), статусы |
| Файловое хранилище | MinIO (S3-совместимый) | Сырые данные, сплиты, предсказания, прогнозы |
| Мониторинг задач | Flower | Web-UI для Celery |

---

## Модели

| Модель | Тип | Описание |
|---|---|---|
| **Seasonal Naive** | Baseline | Прогноз = значение той же сезонной точки год назад |
| **CatBoost** | Gradient Boosting | Лаговые, скользящие, EMA и календарные признаки; поддержка Optuna hyperopt |
| **AutoARIMA** | Статистическая | Автоматический подбор порядков ARIMA, statsforecast |
| **AutoETS** | Статистическая | Error-Trend-Seasonal модель с автовыбором компонент |
| **AutoTheta** | Статистическая | Theta-метод с автоматической декомпозицией |

Выбор лучшей модели производится по метрике на валидационной выборке (по умолчанию — MAPE).
Для CatBoost доступна оптимизация гиперпараметров через **Optuna** (30 trials).

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
[4. AutoML] → для каждой модели:
              • fit на train
              • evaluate на val + test
              • сохранение предсказаний в MinIO
              ↓
              выбор лучшей модели по val-метрике
      │
      ▼
[5. Forecast] → прогноз на N точек вперёд
                • SeasonalNaive — прямо
                • StatsModels — sf.predict(h=N)
                • CatBoost — рекурсивно
                ↓
                сохранение forecast.csv в MinIO
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
│   ├── automl/                  # AutoML: ModelSelector, Optuna hyperopt
│   │   └── models/              # CatBoost, SeasonalNaive, StatsForecast обёртки
│   ├── catboost_utilities/      # Обучение и оценка CatBoost
│   ├── seasonal_naive_utilities/ # Seasonal Naive модель
│   ├── filtration.py            # Фильтрация временных рядов
│   ├── data_processing.py       # Скейлинг, агрегация, expand, сплиты
│   ├── classifical_features.py  # Lag / rolling / EMA / calendar признаки
│   ├── evaluation.py            # MAPE, RMSE, MAE, R², nRMSE
│   ├── model_selection.py       # Временные сплиты (ratio, date, tail)
│   └── custom_types.py          # Общие типы: Splits, ModelResult, AutoMLResult
│
├── api/                         # FastAPI приложение
│   └── routers/                 # projects, jobs, panels, automl, forecast
│
├── worker/                      # Celery задачи
│   └── tasks/                   # run_automl.py, forecast.py
│
├── app/                         # Streamlit интерфейс
│   └── views/                   # upload, quality, automl, forecast
│
├── notebooks/niches/            # Исследовательские ноутбуки (EDA, кластеризация, эксперименты)
├── tests/                       # Unit и E2E тесты (101 тест, ~76% покрытие)
├── docker-compose.yml
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

**Требования:** Python 3.10, [uv](https://github.com/astral-sh/uv), Docker.
