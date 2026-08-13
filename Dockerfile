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
# npm cache persists across builds via the BuildKit cache mount (and never
# lands in an image layer), so `npm ci` re-runs are download-free.
RUN --mount=type=cache,target=/root/.npm npm ci
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
#
# Only the Python modules are copied before `uv sync`; the SPA sources
# (f1web/ui) are build-time only and never enter the runtime image (the built
# dist is copied from stage 1 below). This keeps UI edits from busting the
# expensive uv layer, and keeps the image free of node sources.
COPY pyproject.toml uv.lock ./
COPY f1core/ f1core/
COPY f1data/ f1data/
COPY features/ features/
COPY model/ model/
COPY f1web/*.py f1web/
# The uv cache lives in a BuildKit cache mount: it persists across rebuilds
# (download-free re-runs) but never lands in an image layer.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra web

# Bake the cached raw API + dataset + model checkpoints and the reports.
# config.toml is optional (code defaults match it); copied so overrides stick.
COPY data/ data/
COPY reports/ reports/
COPY config.toml ./

# Built SPA from the ui stage (app.py serves it from f1web/ui/dist).
COPY --from=ui /build/dist f1web/ui/dist/

# Run unprivileged; only data/ and reports/ are written at runtime (dataset
# builds, prediction cache, job snapshots) — .venv and the sources stay
# root-owned read-only, so the chown layer stays small instead of duplicating
# the whole virtualenv (a `chown -R /app` layer would be ~0.5GB).
RUN useradd --create-home --uid 10001 app && chown -R app:app /app/data /app/reports
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

ENTRYPOINT ["f1", "web", "--host", "0.0.0.0", "--port", "8080"]
