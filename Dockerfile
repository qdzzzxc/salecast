FROM python:3.10-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# ---------- лёгкий образ: api, streamlit, flower ----------
FROM base AS app

# ---------- тяжёлый образ: worker (+ chronos / CUDA) ----------
FROM base AS worker

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra neural
