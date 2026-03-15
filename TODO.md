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

### 3. CatBoost отдельно под каждую панель
**Идея:** вместо одной глобальной CatBoost-модели на все панели — обучать отдельную модель для каждой панели.
**Плюсы:** модель специализируется на поведении конкретного ряда, нет смешения разных паттернов.
**Минусы:** очень медленно при большом числе панелей; требует достаточно точек в каждом ряду (min ~24–36 точек).
**Реализация:**
- Новый класс `CatBoostPerPanelForecastModel` в `src/automl/models/catboost_model.py`
- В `fit_evaluate`: для каждой панели отдельный `train_catboost` + `evaluate_catboost`
- В `forecast_future`: для каждой панели отдельный `train_catboost` + `predict`
- Добавить в UI как новую опцию модели `"catboost_per_panel"` с предупреждением о скорости
- Можно добавить порог: если панелей < N, то per-panel, иначе предупреждение

### 4. CatBoost на кластеры панелей
**Идея:** кластеризовать панели по TS-признакам (ts_features.py), обучить одну CatBoost-модель на каждый кластер.
**Зависимость:** `src/ts_features.py` уже есть.

### 5. TS2Vec
Контрастивное обучение представлений ВР → embeddings → forecasting head.
Код будет предоставлен пользователем, затем адаптируем под `BaseForecastModel`.

### 6. Chronos (Amazon)
Foundation model (T5-based, pretrained). Zero-shot или fine-tuned.
`pip install chronos-forecasting`. Код будет предоставлен пользователем.

### 7. AutoGluon
Исследование: запустить `TimeSeriesPredictor` на демо-данных, оценить качество vs размер vs скорость.
Решить: обёртка как одна модель или отдельный AutoML-backend.

### 8. Статфичи от Пасканова
Расширенные признаки для CatBoost (Fourier, entropy, trend strength и др.).
Код будет предоставлен пользователем, интегрируем в `classifical_features.py` или отдельным модулем.

### 9. Мультисезонность для дневных данных (MSTL)
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

## 🔵 Финальная валидация

### 10. Прогон на данных WB
- Запустить все модели на `data/raw/filtered_ts.csv`
- Сравнить метрики в таблице (val MAPE, test MAPE, время обучения)
- Зафиксировать best model и типичные значения MAPE

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
