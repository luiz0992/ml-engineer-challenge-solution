"""FastAPI application entrypoint.

Wires configuration, logging, middleware, dependencies, and routers together,
and manages the lifecycle of the services that must exist before the first
request arrives.

Startup is deliberately fail-fast on the model and fail-open on the cache.
Serving without a model is pointless, so a model that cannot load stops the
process and the orchestrator does not route traffic to it. A cache that cannot
connect is logged and the service continues uncached, because a Redis outage
should not take down inference.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import AppEnv, Settings, UserTier, get_settings
from api.db.session import (
    check_connection,
    create_all,
    create_engine,
    create_session_factory,
)
from api.exceptions import APIError, RateLimitExceededError
from api.logging_config import configure_logging, get_correlation_id, get_logger
from api.middleware.auth import APIKeyStore
from api.middleware.monitoring import (
    CorrelationMiddleware,
    MetricsMiddleware,
    record_model_loaded,
)
from api.middleware.rate_limit import RateLimiter
from api.routers import auth, batch, classification, detection, health, similarity
from api.services.audit_service import AuditService
from api.services.cache_service import CacheService
from api.services.detection_service import DetectionService
from api.services.experiment_service import ExperimentRegistry
from api.services.inference_service import InferenceService
from api.services.model_service import ModelService, discover_versions
from api.services.similarity_service import SimilarityIndex, SimilarityService
from api.utils.validators import configure_pillow_limits

logger = get_logger(__name__)

DESCRIPTION = """
Multi-model computer vision API serving image classification, object detection,
and batch inference.

## Authentication

Exchange an API key for a bearer token at `POST /api/v1/auth/token`, then send
`Authorization: Bearer <token>` on every request.

## Rate limits

Applied per user tier and reported in `X-RateLimit-Limit` and
`X-RateLimit-Remaining` on every response. A rejected request returns 429 with
`Retry-After`.

| Tier | Requests / minute |
| --- | --- |
| free | 10 |
| pro | 120 |
| enterprise | 1200 |

## Correlation

Every response carries `X-Correlation-ID`. Supply your own to have it adopted;
quote it when reporting a problem.
"""


async def _create_redis(settings: Settings) -> Redis | None:
    """Connect to Redis, returning ``None`` if it is unavailable.

    Connectivity is verified with a ping at startup rather than deferred to the
    first request, so a misconfiguration appears in the startup logs instead of
    as a puzzling latency spike later.
    """
    try:
        client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        await client.ping()
    except (RedisError, OSError) as exc:
        logger.warning(
            "redis_unavailable",
            error=str(exc),
            impact="result caching and rate limiting are disabled",
        )
        return None

    logger.info("redis_connected", url=settings.redis_url)
    return client


async def _create_database(
    settings: Settings,
) -> tuple[AsyncEngine | None, async_sessionmaker[AsyncSession] | None]:
    """Connect to Postgres, returning ``(None, None)`` when unavailable.

    Unlike the model, the database is not required to serve traffic: inference
    works without an audit trail. Failing startup here would take down a
    service that is otherwise perfectly capable of answering requests, so the
    failure is logged loudly and the service continues without auditing.
    """
    try:
        engine = create_engine(settings)
        healthy, detail = await check_connection(engine)
        if not healthy:
            raise RuntimeError(detail or "connection check failed")

        # Convenience for local development and tests. Production schema
        # changes go through Alembic; see migrations/.
        if not settings.is_production:
            await create_all(engine)

        logger.info("database_connected", host=settings.postgres_host)
        return engine, create_session_factory(engine)

    except Exception as exc:
        logger.warning(
            "database_unavailable",
            error=str(exc),
            impact="inference audit logging is disabled",
        )
        return None, None


def _seed_api_keys(settings: Settings) -> APIKeyStore:
    """Create the API key store.

    Development seeds one key per tier so the API is usable immediately. In
    production keys come from the database, and seeding is skipped entirely —
    shipping a known key to production would be an unauthenticated back door.
    """
    store = APIKeyStore()

    if settings.app_env is AppEnv.PRODUCTION:
        logger.info("api_key_store_initialised", seeded=False, source="database")
        return store

    for tier in UserTier:
        store.add(f"dev-key-{tier.value}", user_id=f"dev-{tier.value}", tier=tier)

    logger.warning(
        "api_key_store_seeded_for_development",
        keys=len(store),
        detail="dev-key-free / dev-key-pro / dev-key-enterprise; never enabled in production",
    )
    return store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build services on startup and release them on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_format == "json")
    configure_pillow_limits(settings.max_image_pixels)

    logger.info(
        "starting",
        env=settings.app_env.value,
        backend=settings.inference_backend.value,
        artifacts=str(settings.artifacts_dir),
    )

    redis = await _create_redis(settings)
    cache_service = CacheService(redis, ttl_seconds=settings.cache_ttl_seconds)
    rate_limiter = RateLimiter(redis, settings)

    engine, session_factory = await _create_database(settings)
    audit_service = AuditService(session_factory)
    await audit_service.start()

    model_service = ModelService(settings)
    started = time.perf_counter()
    # Not guarded: a model that cannot load is a fatal misconfiguration, and
    # failing startup keeps the instance out of the load balancer.
    model = model_service.load_classifier()
    record_model_loaded(
        model=model.name,
        version=model.version,
        backend=model.backend.value,
        degraded=model.degraded_from is not None,
    )

    # Any additional versions present on disk are loaded alongside the default,
    # so a second version is a deployment concern rather than a code change.
    # They are not made active: a new version becomes reachable by pinning or
    # by an experiment, never by the mere act of shipping it.
    for extra in discover_versions(settings.artifacts_dir / "onnx"):
        if extra == model.version:
            continue
        try:
            alternate = model_service.load_classifier(version=extra, make_active=False)
        except Exception as exc:
            logger.warning("model_version_unavailable", version=extra, error=str(exc))
            continue
        record_model_loaded(
            model=alternate.name,
            version=alternate.version,
            backend=alternate.backend.value,
            degraded=alternate.degraded_from is not None,
        )
    # The detector is optional: a deployment may serve classification only, and
    # failing startup over it would take down a working service.
    detection_service = None
    try:
        detector = model_service.load_detector()
        record_model_loaded(
            model=detector.name,
            version=detector.version,
            backend=detector.backend.value,
            degraded=detector.degraded_from is not None,
        )
        detection_service = DetectionService(model_service, settings)
    except Exception as exc:
        logger.warning(
            "detector_unavailable",
            error=str(exc),
            impact="POST /api/v1/detect will return 503",
        )

    # Similarity search is optional in the same way as detection: a deployment
    # without an index degrades one endpoint rather than failing startup.
    similarity_service = None
    try:
        embedder = model_service.load_embedder()
        index = SimilarityIndex.load(settings.artifacts_dir)
        similarity_service = SimilarityService(model_service, index, settings, embedder.class_names)
        record_model_loaded(
            model=embedder.name,
            version=embedder.version,
            backend=embedder.backend.value,
            degraded=embedder.degraded_from is not None,
        )
    except Exception as exc:
        logger.warning(
            "similarity_unavailable",
            error=str(exc),
            impact="POST /api/v1/similar will return 503",
        )

    logger.info("models_ready", duration_ms=round((time.perf_counter() - started) * 1000, 1))

    app.state.settings = settings
    app.state.redis = redis
    app.state.cache_service = cache_service
    app.state.rate_limiter = rate_limiter
    app.state.model_service = model_service
    app.state.db_engine = engine
    app.state.audit_service = audit_service
    app.state.inference_service = InferenceService(
        model_service, cache_service, settings, audit_service
    )
    # Experiments are resolved against loaded models, so a variant naming an
    # unloaded version is rejected at startup rather than 404-ing that share of
    # traffic.
    available_versions: dict[str, set[str]] = {}
    for loaded in model_service.list_models():
        available_versions.setdefault(loaded.name, set()).add(loaded.version)

    app.state.experiment_registry = ExperimentRegistry.load(
        settings.artifacts_dir / "experiments.json",
        available_versions=available_versions,
    )
    app.state.detection_service = detection_service
    app.state.similarity_service = similarity_service
    app.state.api_key_store = _seed_api_keys(settings)

    try:
        yield
    finally:
        # Drain the audit queue before closing the engine, so records buffered
        # at the moment of a rolling deploy are persisted rather than lost --
        # audit gaps would otherwise cluster exactly around deploys.
        #
        # Guarded because it runs first: `stop()` waits on the writer task, and
        # anything that propagated out of it would skip the two releases below
        # and leak a connection pool on every shutdown. Losing the tail of the
        # audit trail is the lesser failure.
        try:
            await audit_service.stop()
        except Exception:
            logger.exception("audit_shutdown_failed", impact="buffered records may be lost")

        if engine is not None:
            await engine.dispose()
        if redis is not None:
            await redis.aclose()
        logger.info("shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can construct
    independent instances with their own configuration.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Multi-Model Computer Vision API",
        description=DESCRIPTION,
        version=health.API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware executes in reverse registration order, so correlation is
    # registered last and therefore runs first: every metric and log line
    # emitted downstream already carries the correlation ID.
    app.add_middleware(
        MetricsMiddleware,
        exclude_paths=frozenset({f"{settings.api_v1_prefix}/metrics", "/health/live"}),
    )
    app.add_middleware(CorrelationMiddleware)

    if not settings.is_production:
        # Permissive CORS for local development only. A production deployment
        # sets explicit origins at the gateway; "*" with credentials is a
        # standing invitation to cross-origin abuse.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    prefix = settings.api_v1_prefix
    app.include_router(auth.router, prefix=prefix)
    app.include_router(classification.router, prefix=prefix)
    app.include_router(detection.router, prefix=prefix)
    app.include_router(similarity.router, prefix=prefix)
    app.include_router(batch.router, prefix=prefix)
    app.include_router(health.router, prefix=prefix)
    # Unprefixed aliases, so orchestrator probes need not know the API version.
    app.include_router(health.router)

    _register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "Multi-Model Computer Vision API",
            "version": health.API_VERSION,
            "docs": "/docs",
            "health": f"{prefix}/health",
        }

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Install handlers that render every error in one consistent envelope."""

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        correlation_id = get_correlation_id()
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitExceededError):
            headers["Retry-After"] = str(exc.retry_after_seconds)

        log = logger.warning if exc.status_code < 500 else logger.error
        log(
            "request_failed",
            code=exc.code,
            status_code=exc.status_code,
            path=request.url.path,
            detail=exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(correlation_id),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Render Pydantic validation failures in the standard envelope.

        FastAPI's default body has a different shape from every other error
        this API returns, which would force clients to parse two formats.
        """
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request payload failed validation.",
                    "details": {"fields": _summarise_validation(exc)},
                    "correlation_id": get_correlation_id(),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": str(exc.detail),
                    "correlation_id": get_correlation_id(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for unanticipated failures.

        The exception and traceback go to the logs under the correlation ID;
        the client receives a generic message. Returning the exception text
        would leak file paths, library versions, and sometimes configuration.
        """
        correlation_id = get_correlation_id()
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            exception_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred. Quote the correlation ID "
                    "when reporting this.",
                    "correlation_id": correlation_id,
                }
            },
        )


def _summarise_validation(exc: RequestValidationError) -> list[dict[str, str]]:
    """Reduce Pydantic errors to a compact, caller-safe list."""
    summary: list[dict[str, str]] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        summary.append({"field": location or "body", "message": str(error.get("msg", "invalid"))})
    return summary


app = create_app()
