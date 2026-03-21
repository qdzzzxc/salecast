# TODO — Salecast

Живой список задач по проекту. Приоритет сверху вниз внутри каждого раздела.

---

## 🔴 Критично — баги моделирования

### 1. Разрывы на границах train/val/test
**Симптом:** первая точка val/test на графике резко падает вниз относительно реального ряда.
**Вероятные причины:**
- Скейлинг: scaler fit на train, поэтому первое предсказание val скейлится по-другому
- Off-by-one: предсказание строится для `last_train + 1`, но ряд может иметь пропуск дат
- Lag-фичи: лаг-фичи на границе используют нули (expanded panel) вместо реальных значений

**Что проверить:**
- `src/data_processing.py` — как применяется `expand_to_full_panel` и fillna перед фичами
- `src/classifical_features.py` — какие лаги строятся на первых точках val/test
- `src/catboost_utilities/evaluate.py` — как собираются предсказания

**Фикс-гипотеза:** передавать весь ряд (train+val+test) при построении фичей, обрезать по split-границам уже после. Это даст корректные лаги на границах.

---

## 🟡 Модели и алгоритмы

### 2. Ансамблирование моделей
**Идея:** объединить предсказания нескольких лучших моделей.
**Подходы (выбрать при реализации):**
- Weighted average по val MAPE
- Stacking: метамодель (линрег или CatBoost) на предсказаниях базовых моделей
- Best-of-N per panel: для каждой панели — модель с лучшим индивидуальным val-MAPE

**Когда:** после того как все целевые модели готовы.

---

### 3. CatBoost отдельно под каждую панель ✅ ГОТОВО
- `CatBoostPerPanelForecastModel` в `src/automl/models/catboost_model.py`
- В UI: чекбокс `catboost_per_panel` с caption "Медленно"

### 4. Кластеризация панелей ✅ ГОТОВО
- `src/clustering.py` — extract_panel_features, cluster_panels (kmeans/hdbscan), cluster_panels_auto (silhouette), UMAP, mean TS
- `worker/tasks/clustering.py` — Celery task
- `api/routers/clustering.py` — POST run_clustering, GET cluster_data
- `app/views/clustering.py` — UI: форма (3 метода), UMAP scatter, distribution bar, silhouette chart, mean TS grid
- Опциональный шаг пайплайна между Quality и AutoML

### 5. CatBoost на кластеры панелей ✅ ГОТОВО
- `CatBoostClusteredForecastModel` в `src/automl/models/catboost_clustered_model.py`
- Принимает готовые `cluster_labels` из шага кластеризации
- Интегрирован в selector, run_automl, forecast
- В UI: чекбокс disabled если нет результата кластеризации

### 6. TS2Vec → CatBoost/MLP
**Что:** контрастивное обучение представлений ВР → embeddings → downstream regressor.
**Источник кода (уже изучен):**
- Ядро: `/home/nikita/projects/sales_ts_prediction/repos/ts2vec/`
- Обёртки: `/home/nikita/projects/sales_ts_prediction/src/ts2vec_utilities/`
- MLP head: `/home/nikita/projects/sales_ts_prediction/src/mlp_utilities/`
- Regression features: `/home/nikita/projects/sales_ts_prediction/src/get_features.py`
**Реализация:** скопировать ядро в `src/automl/ts2vec/`, адаптировать utilities под `BaseForecastModel`.
**Зависимости:** `uv add torch --optional neural`

### 7. Chronos (Amazon)
Foundation model (T5-based, pretrained на ~1B точек). Zero-shot прогнозирование.
`uv add chronos-forecasting --optional neural`.
Реализуем самостоятельно по документации: `ChronosForecastModel` в `src/automl/models/chronos_model.py`.

### 8. AutoGluon
Исследование: запустить `TimeSeriesPredictor` на демо-данных, оценить качество vs размер (~2–4 ГБ) vs скорость.
Откладываем до завершения остальных моделей.

### 9. Feature importance — анализ важности признаков
**Что:** после обучения CatBoost показать какие фичи реально важны (lag_1, trend, cdf, calendar и т.д.).
**Зачем:** много кастомных признаков (lags, rolling, EMA, trend, CDF, calendar) — нужно понимать какие работают, какие шум.
**Реализация:**
- Извлечь `model.get_feature_importance()` после `train_catboost`
- Сохранить top-N фичей в `ModelResult` (новое поле `feature_importance: list[tuple[str, float]]`)
- В UI AutoML: bar chart с top-20 фичами для каждой CatBoost-модели
- Опционально: в Forecast UI показать importance выбранной модели
**Где:** `src/catboost_utilities/train.py`, `src/custom_types.py`, `app/views/automl.py`

### 10. MSTL-сезонность — выделение и использование
**Что:** декомпозиция ряда через MSTL (Multiple Seasonal-Trend decomposition using Loess).
**Применения:**
- **Признак для кластеризации:** сила сезонности (seasonality strength) как фича в `extract_panel_features` — панели с похожей сезонностью в один кластер
- **Признак для CatBoost:** seasonal component как дополнительная фича в `build_ts_features`
- **Отдельная модель:** MSTL + downstream forecaster (AutoETS/CES) — уже частично описано в п.11
**Источник:** `statsforecast.models.MSTL` или `statsmodels.tsa.seasonal.MSTL`
**Реализация:**
- `src/mstl_features.py` — `extract_mstl_components(series, season_lengths)` → trend, seasonal, residual; `seasonality_strength(series, season_lengths) -> float`
- Интеграция в `src/clustering.py` → `extract_panel_features` добавляет seasonality_strength
- Интеграция в `src/classifical_features.py` → опциональная seasonal-фича в `build_ts_features`
- Toggle в Settings: `use_mstl_features`, `mstl_season_lengths`

### 11. Мультисезонность для дневных данных (MSTL)
**Проблема:** при freq=D сейчас передаётся `season_length=7` — только недельная сезонность.
Дневные данные имеют 3 уровня: 7 (неделя), ~30 (месяц), 365 (год).

**Фикс:**
- Добавить тип модели `mstl` в `StatsForecastModel`:
  ```python
  from statsforecast.models import MSTL, AutoETS
  MSTL(season_length=[7, 365], trend_forecaster=AutoETS())
  ```
- Для D: `season_length=[7, 365]`, для W: `[52]`, для MS: `[12]`
- Требует ≥ 2–3 года данных для надёжной оценки 365-периода

---

## 🟡 Тесты

### 4. E2E тест полного пайплайна через API
Покрыть сценарий:
1. POST `/projects` (загрузить CSV)
2. POST `/projects/{id}/run` (split)
3. Poll `/jobs/{id}` до `status=done`
4. POST `/projects/{id}/run_automl` (models=["seasonal_naive"])
5. Poll до done
6. POST `/projects/{id}/run_forecast` (horizon=3)
7. Poll до done
8. GET `/projects/{id}/forecast_data` → проверить структуру ответа

Файл: `tests/test_e2e_api.py` (pytest + httpx, запускать против живого стека)

### 5. E2E тест моделирования (уже частично есть)
`tests/scripts/test_automl_e2e.py` — расширить:
- Добавить сравнение: прогноз на known holdout vs реальные значения, MAPE должна быть разумной
- Проверить что forecast CSV содержит ровно `n_panels × horizon` строк
- Тест на синтетике с известным паттерном: seasonal naive должен воспроизвести сезонность с MAPE < 15%

---

## 🟡 Надёжность сервисов

### 6. Проверка взаимодействия сервисов (нет мусора)
Проверить что при удалении проекта:
- [ ] Job записи в PostgreSQL удаляются (CASCADE или явно в роутере)
- [ ] Redis: ключи Celery результатов протухают (TTL настроен?)

Проверить что при падении воркера в середине задачи:
- [ ] Не остаётся partial-файлов в MinIO

Проверить что при перезапуске API:
- [ ] Pending job'ы не зависают навсегда (нужен механизм timeout или статус-проверки при старте)

---

## 🔵 Проверка на реальных данных

### 11. Прогон CatBoost per-panel на реальных данных WB
Запустить `catboost_per_panel` через UI на `data/raw/filtered_ts.csv`, сравнить с глобальным CatBoost:
- [ ] Метрики (val MAPE, test MAPE) — лучше или хуже глобального?
- [ ] Время обучения — насколько медленнее?
- [ ] Проверить что progress-бар отображает прогресс по панелям корректно

---

## 🔵 Финальная валидация

### 10. Прогон на данных WB
- Запустить все модели на `data/raw/filtered_ts.csv`
- Сравнить метрики в таблице (val MAPE, test MAPE, время обучения)
- Зафиксировать best model и типичные значения MAPE

### 12. Кросс-валидация итоговой модели + отчёт
**Что:** после выбора лучшей модели через AutoML — проверить её стабильность через temporal CV.
**Зачем:** один split может быть удачным. CV покажет mean ± std метрик → подтвердит качество.
**Реализация:**
- Temporal CV: expanding window, N фолдов (3–5), каждый раз сдвигаем границу train/test
- Для каждого fold: обучение с теми же параметрами (из AutoML result) → метрики по панелям
- Агрегация: mean/std MAPE, RMSE по фолдам + box plot разброса по панелям
- Celery task `run_cross_validation` — тяжёлая операция
- Результат сохраняется в MinIO как JSON, отображается в UI
**UI:**
- Кнопка "Кросс-валидация" на странице Forecast (или отдельная вкладка "Отчёт")
- Таблица: fold → метрики, итоговая строка mean ± std
- Box plot MAPE по панелям для каждого fold
- Опционально: экспорт отчёта (PDF или markdown)
**Приоритет:** после реализации всех моделей, перед README

---

## 🟢 Качество и UX

### 7. Проверить Optuna (hyperopt) в продакшне
- [ ] Убедиться что `tune_catboost` не падает при `use_hyperopt=True` через реальный UI (end-to-end)
- [ ] Проверить что прогресс в UI не зависает пока идёт hyperopt (он блокирующий внутри worker'а — таймаут 300s может не хватить при большом n_trials)

### 8. Настройки для AutoETS/AutoTheta
**Отсутствуют настройки для:**
- **AutoETS** — параметр `model` (например `"ZZZ"`, `"AAA"`) и `season_length` override
- **AutoTheta** — `decomposition_type` (`"multiplicative"` / `"additive"`)
- **Seasonal Naive** — показать текущий `season_length` как информационное поле (read-only)

**Приоритет:** низкий — AutoETS/AutoTheta редко тюнятся вручную.

### 9. Ноутбуки — убрать лишнее, оставить суть
- [ ] Очистить все outputs (`nbconvert --clear-output` или вручную)
- [ ] Убрать мусорные/экспериментальные ячейки
- [ ] Добавить markdown-заголовки (что делает ноутбук, какой вход/выход)
- [ ] Проверить что запускаются от начала до конца без ошибок

### 10. README.md
Сделать в самом конце, когда всё остальное готово.
Структура:
- Краткое описание + скриншоты
- Архитектура (схема сервисов)
- Быстрый старт: `docker compose up`
- Описание пайплайна (Upload → Quality → AutoML → Forecast)
- Используемые технологии

---

## ✅ Готово
- Backend: FastAPI + Celery + Redis + PostgreSQL + MinIO
- Streamlit: Upload, Quality, AutoML (Моделирование), Forecast
- AutoML: SeasonalNaive, CatBoost, AutoARIMA, AutoETS, AutoTheta
- Hyperopt (Optuna) — вызывается через worker
- Docker Compose с полным стеком
- Настройки моделей в UI (CatBoost: iterations/lr/depth, AutoARIMA: approximation, Hyperopt: n_trials)
- YAML import/export конфига AutoML (через popover)
- Фикс: вкладка «Качество данных» пропадала после AutoML
- Фикс: очистка MinIO при удалении проекта
- Фикс: YAML import/export layout (перенесён в popover)
- Фикс: демо-проекты пересоздавались при F5
- Фикс: баг с лаговыми фичами при прогнозе CatBoost
- Прогресс-бар для прогноза (loading/training/forecasting/saving + sub-progress для CatBoost)
- Рефакторинг: `forecast_future()` в каждом классе модели, убран дублирующий код из воркера
- Trend и CDF признаки в `build_ts_features` + переименование `build_monthly_features`
- Кластеризация панелей: kmeans/kmeans_auto/hdbscan, UMAP, silhouette score, distribution chart
- CatBoost clustered: отдельная модель на кластер, интеграция в весь пайплайн
