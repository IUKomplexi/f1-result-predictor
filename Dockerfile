# syntax=docker/dockerfile:1
# Two-stage image for the F1 result predictor dashboard.
#
# Stage 1 builds the React SPA (f1web/ui/dist is gitignored, so it is
# compiled here); stage 2 installs the Python app with uv (locked) and bakes
# in the cached data, reports, and config for a fully offline, turnkey runtime.
#
# Build:  docker build -t f1-result-predictor .
# Run:    docker run -p 8080:8080 f1-result-predictor
#         -> http://127.0.0.1:8080/  (dashboard)

# ---- Stage 1: build the SPA -------------------------------------------------
FROM node:22-slim AS ui
WORKDIR /build
COPY f1web/ui/package.json f1web/ui/package-lock.json ./
RUN npm ci
COPY f1web/ui/ ./
RUN npm run build

# ---- Stage 2: Python runtime ------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# uv gives a locked, reproducible install from uv.lock (no compiler needed:
# all deps ship manylinux wheels).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Copy the project sources and install editable (uv sync installs the current
# project editable into /app/.venv), so the app resolves data/, reports/ and
# config.toml from /app at runtime.
COPY pyproject.toml uv.lock ./
COPY f1core/ f1core/
COPY f1data/ f1data/
COPY f1weather/ f1weather/
COPY features/ features/
COPY model/ model/
# Explicit file list (not a bare `COPY f1web/`): ui/ is built in stage 1 and
# copied as dist below, and ui/node_modules must not enter the runtime image.
# New modules in the f1web package must be added here (e.g. f1web/jobs.py).
COPY f1web/__init__.py f1web/app.py f1web/jobs.py f1web/
RUN uv sync --frozen --no-dev --extra web

# Bake the cached raw API + dataset + model checkpoints and the reports.
# config.toml is optional (code defaults match it); copied so overrides stick.
COPY data/ data/
COPY reports/ reports/
COPY config.toml ./

# Built SPA from the ui stage (app.py serves it from f1web/ui/dist).
COPY --from=ui /build/dist f1web/ui/dist/

# Run unprivileged; /app/reports stays writable for generated snapshots.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

ENTRYPOINT ["f1", "web", "--host", "0.0.0.0", "--port", "8080"]
