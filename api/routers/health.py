"""Health, readiness, model metadata, and metrics endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.config import Settings, get_settings
from api.dependencies import get_cache_service, get_model_service
from api.middleware.monitoring import REGISTRY
from api.models.responses import (
    ComponentHealth,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from api.services.cache_service import CacheService
from api.services.model_service import ModelService

router = APIRouter(tags=["operations"])

#: Process start, used to report uptime.
_STARTED_AT = time.monotonic()

API_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(
    request: Request = None,  # type: ignore[assignment]
    models: ModelService = Depends(get_model_service),
    cache: CacheService = Depends(get_cache_service),
) -> HealthResponse:
    """Report aggregate health and the state of each dependency.

    Distinguishes ``degraded`` from ``unhealthy`` deliberately. If the cache is
    down but models still serve, the service is usable and must stay in the
    load balancer; reporting that as unhealthy would take down a working
    service. Only the loss of inference itself is ``unhealthy``.
    """
    components: list[ComponentHealth] = []

    loaded = models.list_models()
    components.append(
        ComponentHealth(
            name="models",
            healthy=bool(loaded),
            detail=(
                f"{len(loaded)} model(s) loaded"
                if loaded
                else "no models loaded; inference unavailable"
            ),
        )
    )

    started = time.perf_counter()
    cache_healthy, cache_detail = await cache.ping()
    components.append(
        ComponentHealth(
            name="cache",
            healthy=cache_healthy,
            detail=cache_detail or "connected",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    )

    database_healthy = True
    engine = getattr(getattr(request, "app", None), "state", None)
    engine = getattr(engine, "db_engine", None) if engine is not None else None
    if engine is not None:
        from api.db.session import check_connection

        started = time.perf_counter()
        database_healthy, database_detail = await check_connection(engine)
        components.append(
            ComponentHealth(
                name="database",
                healthy=database_healthy,
                detail=database_detail or "connected",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )

    degraded = [m for m in loaded if m.degraded_from is not None]
    if degraded:
        components.append(
            ComponentHealth(
                name="inference_backend",
                healthy=True,
                detail=(
                    "running on a fallback backend: "
                    + ", ".join(
                        f"{m.key} using {m.backend.value} "
                        f"(requested {m.degraded_from.value if m.degraded_from else '?'})"
                        for m in degraded
                    )
                ),
            )
        )

    if not loaded:
        status = "unhealthy"
    elif not cache_healthy or degraded or not database_healthy:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        version=API_VERSION,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 1),
        components=components,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Returns 200 whenever the process is running.",
)
async def liveness() -> dict[str, str]:
    """Liveness probe for the orchestrator.

    Deliberately checks nothing. Liveness answers "should this container be
    restarted"; making it depend on the cache or database would restart a
    healthy process because something downstream failed, turning a partial
    outage into a crash loop. Dependency checks belong in readiness.
    """
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness(
    models: ModelService = Depends(get_model_service),
    response: Response = None,  # type: ignore[assignment]
) -> dict[str, object]:
    """Readiness probe: is this instance able to serve traffic?

    Returns 503 until a model is loaded, so the orchestrator withholds traffic
    during the startup window rather than routing requests that would fail.
    """
    ready = models.is_ready
    if response is not None and not ready:
        response.status_code = 503
    return {"ready": ready, "models_loaded": len(models.list_models())}


@router.get("/models", response_model=ModelsResponse, summary="Registered models")
async def list_models(
    models: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    """List every registered model version and its metadata."""
    return ModelsResponse(
        models=[
            ModelInfo(
                name=model.name,
                version=model.version,
                task=model.task,
                backend=model.backend.value,
                loaded=True,
                is_active=models.is_active(model),
                num_classes=model.num_classes,
                input_size=model.image_size,
                artifact_size_mb=model.artifact_size_mb,
                metrics=model.metrics,
                loaded_at=model.loaded_at,
            )
            for model in models.list_models()
        ],
        default_backend=settings.inference_backend.value,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    responses={200: {"content": {CONTENT_TYPE_LATEST: {}}}},
)
async def metrics() -> Response:
    """Expose metrics in Prometheus text format.

    Unauthenticated, which is the convention for a scrape endpoint: Prometheus
    does not carry bearer tokens, and the endpoint is expected to be reachable
    only from inside the deployment network. The Compose and nginx
    configurations do not expose it publicly.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
