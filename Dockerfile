# syntax=docker/dockerfile:1
# Multi-stage image for the F1 result predictor web dashboard.
#
# Stage 1 builds the React SPA (f1web/ui/dist is gitignored, so it is
# compiled here); stage 2 installs the Python app and bakes in the cached
# data, reports, and config for a fully offline, turnkey runtime.
#
# Build:  docker build -t f1-result-predictor .
# Run:    docker run -p 8080:8080 f1-result-predictor
#         -> http://127.0.0.1:8080/dashboard

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
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=5 \
    PIP_TIMEOUT=60

WORKDIR /app

# Install the project (all deps ship manylinux wheels; no compiler needed).
# Editable install: the app must resolve from /app, where data/, reports/,
# config.toml and f1web/ui/dist are baked in — a site-packages copy would
# not see them (mirrors requirements.txt's `-e .[test]`).
COPY pyproject.toml ./
COPY f1data/ f1data/
COPY f1weather/ f1weather/
COPY features/ features/
COPY model/ model/
COPY f1web/__init__.py f1web/app.py f1web/
COPY f1web/templates/ f1web/templates/
COPY config.py predict.py reporting.py httpclient.py ./
RUN pip install -e ".[web]"

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

ENTRYPOINT ["f1-web", "--host", "0.0.0.0", "--port", "8080"]
