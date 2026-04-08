# TODO — Salecast

Живой список задач по проекту. Приоритет сверху вниз внутри каждого раздела.

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

### 6. TS2Vec → CatBoost ✅ ГОТОВО
- `TS2VecForecastModel` в `src/automl/models/ts2vec_model.py`
- Ядро TS2Vec (dilated conv encoder + contrastive loss) в `src/automl/ts2vec/`
- TS2Vec encoder → full_series embeddings → CatBoost downstream
- GPU автодетекция, прогресс-бар, feature importance
- Параметры (output_dims, n_epochs, batch_size) через весь стек

### 7. Chronos-2 (Amazon) ✅ ГОТОВО
- `ChronosForecastModel` в `src/automl/models/chronos_model.py`
- Chronos-2 (120M параметров), zero-shot через `predict_df` pandas API
- GPU автодетекция (cuda/cpu), подпись "GPU" в UI
- `docker-compose.gpu.yml` для проброса GPU в worker

### 8. PatchTST ⚠️ ИЗВЕСТНЫЕ ПРОБЛЕМЫ
- Сезонность не улавливается — прогноз плоский. Возможно underfitting (мало `max_steps`)
- Автоподбор `input_size`/`patch_len` по `season_length` не проверен на реальных данных
- Модель реализована и интегрирована, но качество требует тюнинга дефолтов

### 9. Feature importance ✅ ГОТОВО
- bar chart top-20 для всех CatBoost-моделей в UI AutoML
- Кластерные модели: взвешенное среднее по размеру кластера (вместо простого)
- ts2vec_clustered: embedding-фичи исключены (разные энкодеры → несопоставимы)
- catboost_clustered: importance корректно нормализован по всем кластерам

### 10. MSTL-сезонность ✅ ГОТОВО
- Декомпозиция, признак CatBoost, визуализация на экране Качество

### 11. Мультисезонность для дневных данных (MSTL) ✅ ГОТОВО
- `StatsForecastModel` поддерживает `model_type="mstl"` — MSTL + AutoETS(ZZN) trend forecaster
- Мультисезонность из коробки: D→[7,365], W→[52], MS→[12]
- MSTL-декомпозиция для просмотра на экране «Качество данных»
- MSTL seasonal как признак CatBoost (toggle `use_mstl_seasonal`)

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

### 12. Кросс-валидация итоговой модели ✅ ГОТОВО
- `generate_expanding_cv_folds()` в `src/model_selection.py`
- Celery task `run_cross_validation` в `worker/tasks/cross_validation.py`
- API: `run_cv`, `cv_progress`, `cv_result` в `api/routers/forecast.py`
- UI: секция на странице Forecast — выбор модели, кол-во фолдов, прогресс, таблица + box plot

---

## 🟢 Качество и UX

### 7. Optuna (hyperopt) ✅ ГОТОВО
- Прогресс по trial'ам в реальном времени (trial N/M, best MAPE)
- `HyperoptResult` — лучшие параметры, история trial'ов
- Визуализация: optimization history + parallel coordinates
- Результаты сохраняются в job result и отображаются на экране AutoML

### 8. Настройки для AutoETS/AutoTheta
**Отсутствуют настройки для:**
- **AutoETS** — параметр `model` (например `"ZZZ"`, `"AAA"`) и `season_length` override
- **AutoTheta** — `decomposition_type` (`"multiplicative"` / `"additive"`)
- **Seasonal Naive** — показать текущий `season_length` как информационное поле (read-only)

**Приоритет:** низкий — AutoETS/AutoTheta редко тюнятся вручную.

### TS2Vec loss chart — сохранение после обучения ✅ ГОТОВО
- `ModelResult.loss_history` — список `(epoch, loss)`
- TS2Vec записывает loss каждой эпохи, сохраняется в job result
- На экране AutoML: секция «Loss обучения» с Plotly графиком (expander)

### 9. Прогресс-бары — детализация этапов
- [ ] **Кластеризация (kmeans_auto):** нет прогресса внутри этапа, пользователь не видит что происходит (перебор k, silhouette)
- [ ] **CatBoost per-panel / clustered:** прогресс появляется только когда пошли панели/кластеры, но этап подготовки данных (build_ts_features, scaling) молчит
- [ ] **UMAP:** долгий шаг без обратной связи (скорее всего нельзя сделать прогресс-бар — UMAP не даёт callback, но можно хотя бы показать «Вычисление UMAP...»)

### 11. Скачивание прогнозов со страницы Моделирование ✅ ГОТОВО
- Кнопка «Скачать предсказания (CSV)» на вкладке Моделирование

### 10. Ноутбуки — убрать лишнее, оставить суть
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
- Feature importance: bar chart top-20 для всех CatBoost-моделей
- MSTL: модель (декомпозиция + AutoETS), признак для CatBoost, визуализация на экране Качество
- Chronos-2: zero-shot foundation model, GPU/CPU, pandas API
- Фикс: разрывы на границах train/val/test (лаг-фичи) + 32 интеграционных теста feature engineering
- ChronosParameters через весь стек (API → worker → UI)
- Multi-stage Dockerfile: neural deps только в worker
- ruff + mypy: все ошибки исправлены
- TS2Vec + CatBoost: contrastive encoder → embeddings → downstream CatBoost, GPU/CPU, 14 тестов
