# --- base: общие зависимости (api, streamlit, flower) ---
FROM python:3.10-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/

ENV UV_LINK_MODE=copy
WORKDIR /app

COPY requirements/base.txt requirements/base.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && uv pip install --no-deps -r requirements/base.txt

# --- app: api, streamlit, flower ---
FROM deps AS app
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# --- worker-deps: + neural (torch, chronos, neuralforecast) ---
FROM deps AS worker-deps
COPY requirements/neural.txt requirements/neural.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -r requirements/neural.txt

# --- worker ---
FROM worker-deps AS worker
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
