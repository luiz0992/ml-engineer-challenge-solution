# syntax=docker/dockerfile:1.7
#
# Multi-stage build producing two images from one Dockerfile:
#
#   --target api     the FastAPI service
#   --target worker  the Celery batch worker
#
# Both share the `runtime` stage, so the dependency layers are built once and
# reused. They differ only in entrypoint and health check, which is the honest
# relationship between them: the same code, invoked differently.
#
# Layer ordering is deliberate. Dependency manifests are copied and installed
# before application source, so editing a Python file reuses the cached
# dependency layer instead of reinstalling several hundred megabytes.
#
# Size: the training stack (torch, torchvision, CUDA) is never installed. The
# serving path uses ONNX Runtime and reimplements preprocessing in NumPy and
# Pillow precisely so this image can stay small.

# ---------------------------------------------------------------------------
# Builder: resolve and install dependencies into a self-contained virtualenv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# uv resolves and installs an order of magnitude faster than pip and honours
# the committed lockfile, so image builds match local development exactly.
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependency manifests only: this layer is invalidated by dependency changes,
# not by application edits.
COPY pyproject.toml uv.lock README.md ./

# --no-install-project installs dependencies without the local package, so the
# cached layer survives changes to our own source.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Runtime: the common base for both services.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp1 is required by ONNX Runtime's threading layer; curl is used by the
# container health checks. Installed with --no-install-recommends and the apt
# lists removed in the same layer, so the cache is never committed to the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. A container process running as root that escapes
# its namespace is root on the host; there is no reason for a web service to
# need it.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home --shell /usr/sbin/nologin app

ENV VIRTUAL_ENV=/build/.venv \
    PATH="/build/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    # Model artifacts are mounted at runtime rather than baked into the image.
    ARTIFACTS_DIR=/app/models/artifacts

COPY --from=builder --chown=app:app /build/.venv /build/.venv

WORKDIR /app

# Application source last: the layer most likely to change is the cheapest to
# rebuild.
COPY --chown=app:app api/ ./api/
COPY --chown=app:app worker/ ./worker/

# Mount point for the artifacts volume. Created here so it exists with the
# right ownership even when no volume is attached.
RUN mkdir -p /app/models/artifacts && chown -R app:app /app

USER app

EXPOSE 8000

# ---------------------------------------------------------------------------
# API service
# ---------------------------------------------------------------------------
FROM runtime AS api

# Readiness, not liveness: the container is only healthy once a model is
# loaded, so an orchestrator withholds traffic during the startup window
# instead of routing requests that would fail.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health/ready || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. The shell form
# would make /bin/sh PID 1, which does not forward signals, and every deploy
# would end in a 10-second SIGKILL instead of a graceful drain.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-graceful-shutdown", "30", \
     "--no-access-log"]

# ---------------------------------------------------------------------------
# Celery worker
# ---------------------------------------------------------------------------
FROM runtime AS worker

# `celery inspect ping` asks this worker to respond over the broker, which
# proves both that the process is alive and that it can reach Redis. A bare
# process check would report a worker that is running but disconnected as
# healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD celery -A worker.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# --concurrency=2: inference is CPU- and memory-heavy, and each worker process
# loads its own copy of the model. Oversubscribing would multiply memory use
# and cause the processes to contend for the same cores.
# --max-tasks-per-child recycles workers periodically, bounding the impact of
# any slow memory growth in native inference libraries.
CMD ["celery", "-A", "worker.celery_app", "worker", \
     "--loglevel=info", \
     "--concurrency=2", \
     "--max-tasks-per-child=100"]
