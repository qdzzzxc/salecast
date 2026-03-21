FROM python:3.10-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Только lock-файлы — этот слой стабилен при изменении кода
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- лёгкий образ: api, streamlit, flower ----------
FROM deps AS app
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# ---------- тяжёлые зависимости worker (стабильный слой) ----------
FROM deps AS worker-deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra neural

# ---------- worker: код поверх тяжёлых deps ----------
FROM worker-deps AS worker
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
